import cv2
from tqdm import trange
from collections import deque
import mujoco_py
import numpy as np
from gym import utils
from gym.wrappers import TimeLimit
from stable_baselines3.common.env_checker import check_env
import mujoco_env
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed

TARGETS = {1: {"bca": np.array([-0.029918, 0.035143, 1.0431]),
               "lcca": np.array([0.003474, 0.035143, 1.0357])},
           2: {'bca': np.array([-0.013049, -0.077002, 1.0384]),
               'lcca': np.array([0.019936, -0.048568, 1.0315])}}


class CathSimEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 scene: int = 1,
                 target: str = "bca",
                 obs_type: str = "internal",
                 image_size: int = 128,
                 delta: float = 0.008,
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 n_frames: int = 4,
                 compute_force: bool = False):

        self.scene = scene
        self.target = TARGETS[scene][target]
        self.obs_type = obs_type
        self.image_size = image_size
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.compute_force = compute_force

        xml_file = f'scene_{scene}.xml'
        self.image_size = image_size

        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)
        for i in range(n_frames):
            self.frames.append(np.zeros(shape=(image_size, image_size)))

        utils.EzPickle.__init__(self)

        """ Inherits from MujocoEnv """

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5, image_size)

    @ property
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
                for i in range(collision_pos[0] - image_range,
                               collision_pos[0] + image_range):
                    for j in range(collision_pos[1] - image_range,
                                   collision_pos[1] + image_range):
                        if (0 <= i <= self.image_size and 0 <= j <= self.image_size):
                            force_image[j, i] = collision_force

        return force_image

    @ property
    def forces(self):
        """ computes the force and maps it to an image """
        data = self.sim.data

        # for all available contacts
        forces = []
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
                forces.append(
                    [collision_pos[0], collision_pos[1], collision_force])

        return forces

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
        if self.compute_force:
            info["force_image"] = self.force_image

        return obs, reward, done, info

    def _get_obs(self):
        """_get_obs."""

        if self.obs_type == "image":
            image = self.get_image("top_camera")
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

        if self.obs_type == "image":
            self.frames.clear()
            for i in range(self.n_frames):
                self.frames.append(
                    np.zeros(shape=(self.image_size, self.image_size)))
        return self._get_obs()

    def point2pixel(self, point, camera_matrix):
        """Transforms from world coordinates to pixel coordinates."""
        x, y, z = point
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

        return round(xs / s), round(ys / s)

    def get_image(self, camera_name, mode="gray"):
        image = self.render("rgb_array", camera_name=camera_name)
        if mode != "rgb":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        return image

    def print_info(self, link=-1):
        print("Position: ", self.data.qpos[link])
        print("Velocity: ", self.data.qvel[link])
        print("COM Inertia: ", self.data.cinert[link])
        print("COM Velocity: ", self.data.cvel[link])
        print("Actuator Forces: ", self.data.qfrc_actuator[link].shape)


def make_env(rank, scene, target, obs_type, image_size, n_frames, seed):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = CathSimEnv(scene=scene, target=target, obs_type=obs_type,
                         image_size=image_size, n_frames=n_frames)
        env = TimeLimit(env, max_episode_steps=2000)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


if __name__ == "__main__":

    env = CathSimEnv(scene=1,
                     obs_type="internal",
                     target="lcca",
                     image_size=128,
                     n_frames=4)
    env = TimeLimit(env, max_episode_steps=2000)
    check_env(env)
    print(env.observation_space.shape)
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        image = env.render("rgb_array", width=1080, height=1080, camera_name="top_camera")
        cv2.imshow("image", image)
        cv2.waitKey(1)
        # env.render()
