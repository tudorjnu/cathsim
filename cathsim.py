from stable_baselines3 import PPO
import cv2
from collections import deque
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from gym import utils
from gym.wrappers import TimeLimit, FrameStack
from stable_baselines3.common.env_checker import check_env
from utils import ALGOS
from tqdm import trange
import mujoco_env

TARGETS = {1: {"bca": np.array([-0.029918, 0.035143, 1.0431]),
               "lcca": np.array([0.003474, 0.035143, 1.0357])},
           2: {'bca': np.array([-0.013049, -0.077002, 1.0384]),
               'lcca': np.array([0.019936, -0.048568, 1.0315])}}

DEFAULT_CAMERA_CONFIG = {
    "pos": [0.007738, - 0.029034, 1.550]
}


class CathSimEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 scene: int = 1,
                 target: str = "bca",
                 obs_type: str = "internal",
                 image_size: int = 128,
                 delta: float = 0.008,
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 n_frames: int = 3):

        self.scene = scene
        self.target = TARGETS[scene][target]
        self.obs_type = obs_type
        self.image_size = image_size
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward

        utils.EzPickle.__init__(self)

        xml_file = f'scene_{scene}.xml'
        self.image_size = image_size

        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        for i in range(n_frames):
            self.frames.append(np.zeros(shape=(image_size, image_size)))

        """ Inherits from MujocoEnv """

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, image_size)

    @property
    def force_image(self):
        """ computes the force and maps it to an image """
        image_range = 2

        data = self.sim.data
        force_image = np.zeros(
            shape=(self.image_size, self.image_size, 1))

        # for all available contacts
        for i in range(data.ncon):
            contact = data.contact[i]

            geom_1 = [contact.geom1,
                      self.sim.model.geom_id2name(contact.geom1)]
            geom_2 = [contact.geom2,
                      self.sim.model.geom_id2name(contact.geom2)]
            if geom_1[1] is None or geom_2[1] is None:
                collision_pos = self.point2pixel(point=contact.pos,
                                                 camera_matrix=self.top_camera_matrix)
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(
                    self.sim.model, data, i, c_array)
                collision_force = np.linalg.norm(c_array[:3])
                for i in range(collision_pos[0]-image_range,
                               collision_pos[0]+image_range):
                    for j in range(collision_pos[1]-image_range,
                                   collision_pos[1]+image_range):
                        if (0 <= i <= self.image_size and
                                0 <= j <= self.image_size):
                            force_image[j, i] = collision_force

        return force_image

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        Computes the reward for the given achieved goal and desired goal.
        """
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = bool(distance <= self.delta)

        if self.dense_reward:
            reward = self.success_reward if success else -distance
        else:
            reward = self.success_reward if success else -1.0

        return reward, success, distance

    @property
    def head_pos(self):
        """head_pos."""
        head_pos = self.sim.data.get_body_xpos("head")
        head_pos = np.array(head_pos)
        return head_pos

    def step(self, a):
        """ Performs the simulation step
        """

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        head_pos = self.head_pos

        # compute the reward
        reward, done, distance = self.compute_reward(achieved_goal=head_pos,
                                                     desired_goal=self.target)

        # check if the episode is done
        info = {"success": done,
                "distance": distance,
                "head_pos": head_pos,
                "target_pos": self.target}

        return obs, reward, done, info

    def _get_obs(self):
        """_get_obs."""

        if self.obs_type == "image":
            image = self.get_image("top_view")
            self.frames.append(image)
            obs = np.array(self.frames)
        else:
            data = self.sim.data

            position = data.qpos.flat.copy()
            velocity = data.qvel.flat.copy()

            com_inertia = data.cinert.flat.copy()
            com_velocity = data.cvel.flat.copy()

            actuator_forces = data.qfrc_actuator.flat.copy()

            obs = np.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                )
            )

        return obs

    def reset(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nv),
        )

        return self._get_obs()

    def point2pixel(self, point, camera_matrix):
        """Transforms from world coordinates to pixel coordinates."""
        x, y, z = point
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

        return round(xs/s), round(ys/s)

    def get_image(self, camera_name, mode="rgb"):
        image = self.render("rgb_array", camera_name=camera_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def print_info(self, link=-1):
        print("Position: ", self.data.qpos[link])
        print("Velocity: ", self.data.qvel[link])
        print("COM Inertia: ", self.data.cinert[link])
        print("COM Velocity: ", self.data.cvel[link])
        print("Actuator Forces: ", self.data.qfrc_actuator[link].shape)


if __name__ == "__main__":
    env = CathSimEnv(scene=1,
                     obs_type="image",
                     target="lcca",
                     image_size=128,
                     n_frames=3)
    env = TimeLimit(env, max_episode_steps=2000)
    print(env.observation_space.shape)
    model = PPO(
        "CnnPolicy", env,
        verbose=1,
        device="cuda",
        learning_rate=5.05041e-05,
        n_steps=512,
        clip_range=0.1,
        ent_coef=0.000585045,
        n_epochs=20,
        max_grad_norm=1,
        vf_coef=0.871923,
        batch_size=32,
        seed=42)

    model.learn(total_timesteps=1000000)
    exit()

    for _ in trange(2):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)
            print(reward)

            cv2.imshow("Image", obs[0])
            cv2.waitKey(1)

    env.close()
