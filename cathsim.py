import cv2
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from gym import utils
from gym.wrappers import TimeLimit
from stable_baselines3.common.env_checker import check_env
from utils import ALGOS
from tqdm import trange
import mujoco_env

TARGETS = {1: {"bca": [-0.029918, 0.055143, 1.0431],
               "lcca": [0.003474, 0.055143, 1.0357]},
           2: {'bca': [-0.013049, -0.077002, 1.0384],
               'lcca': [0.019936, -0.048568, 1.0315]}}

DEFAULT_CAMERA_CONFIG = {
    "pos": [0.007738, - 0.029034, 1.550]
}


def test_env(env):
    """test_env.

    :param env:
    """
    check_env(env, warn=True)
    print("Observation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    print("Action space:", env.action_space)
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("obs_shape:", obs.shape, "\nreward:", reward, "\ndone:", done)
    for key, value in info.items():
        if key != "force_image":
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value.shape}')
    print("Degrees of Freedom:", env.sim.model.nv)
    obs = env.reset()


class CathSimEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """CathSimEnv."""

    def __init__(self,
                 scene: int = 1,
                 target: str = "bca",
                 obs_type: str = "internal",
                 ep_length: int = 2000,
                 image_size: int = 128,
                 delta: float = 0.008,
                 dense_reward: bool = True,
                 success_reward: float = 10.0):

        self.scene = scene
        self.target = np.array(TARGETS[scene][target])
        self.obs_type = obs_type
        self.ep_length = ep_length
        self.image_size = image_size
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward

        utils.EzPickle.__init__(self)

        xml_file = f'scene_{scene}.xml'
        self.image_size = image_size

        self.current_step = 1
        self.num_resets = -1

        if self.obs_type == "image_time":
            self.obs = np.zeros(
                shape=(image_size, image_size, 3))

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
        success = (distance < self.delta)

        if self.dense_reward:
            reward = self.success_reward if success else -distance
        else:
            reward = self.success_reward if success else -1.0

        return reward, success, distance

    @property
    def head_pos(self):
        """head_pos."""
        head_pos = self.sim.data.get_body_xpos("B99")
        head_pos = np.array(head_pos)
        return head_pos

    def step(self, a):
        """ Performs the simulation step
        """

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # compute the reward
        reward, done, distance = self.compute_reward(achieved_goal=self.head_pos,
                                                     desired_goal=self.target)

        # check if the episode is done
        info = {"success": done, "distance": distance}
        done = self.current_step >= self.ep_length
        self.done = done
        self.current_step += 1

        return obs, reward, done, info

    def _get_obs(self):
        """_get_obs."""

        if self.obs_type == "image":
            obs = self.get_image("top_view", mode="gray")
        elif self.obs_type == "image_time":
            image = self.get_image("top_view", mode="gray")
            np.append(self.obs, image, axis=-1)
            np.delete(self.obs, 0, -1)
            return self.obs
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
        """reset."""
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nv),
        )
        self.current_step = 1
        self.num_resets += 1

        return self._get_obs()

    def point2pixel(self, point, camera_matrix):
        """Transforms from world coordinates to pixel coordinates."""
        x, y, z = point
        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

        return round(xs/s), round(ys/s)

    def get_image(self, camera_name, mode="rgb"):
        """get_image.

        :param camera_name:
        :param mode:
        """
        image = self.render("rgb_array", camera_name=camera_name)
        if mode == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        return image

    def print_info(self, link=-1):
        """print_info.

        :param link:
        """
        print("Position: ", self.data.qpos[link])
        print("Velocity: ", self.data.qvel[link])
        print("COM Inertia: ", self.data.cinert[link])
        print("COM Velocity: ", self.data.cvel[link])
        print("Actuator Forces: ", self.data.qfrc_actuator[link].shape)


if __name__ == "__main__":
    env = CathSimEnv(scene=1,
                     obs_type="internal",
                     ep_length=2000,
                     target="lcca",
                     image_size=128)
    print(env.observation_space)
    env = TimeLimit(env, max_episode_steps=2000)
    obs = env.reset()

    check_env(env)
    # exit()

    print("Degrees of Freedom:", env.model.nv)
    print("Number of joints:", env.model.nq)
    print("Number of actuators:", env.model.na)
    print("Number of bodies:", env.model.nbody)

    print("Position:", env.sim.data.qpos.shape)
    print(env.sim.data.qpos[0])
    print("Velocity:", env.sim.data.qvel.shape)
    print(env.sim.data.qvel[0])

    print("COM inertia:", env.sim.data.cinert.shape)
    print(env.sim.data.cinert[0])
    print("COM velocity:", env.sim.data.cvel.shape)
    print(env.sim.data.cvel[0])

    for _ in trange(2):
        obs = env.reset()
        done = False
        while not done:
            # env.print_info()
            env.render()
            action = env.action_space.sample()
            obs, reward, done, info = env.step(action)

    env.close()
