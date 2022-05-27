import mujoco_env
from tqdm import trange
from utils import ALGOS
from stable_baselines3.common.env_checker import check_env
from gym import utils
import numpy as np
import mujoco_py
import cv2
from collections import OrderedDict
from gym.envs.registration import EnvSpec
from stable_baselines3.common.noise import NormalActionNoise
import random
from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3


IMAGES_SAVING_INTERVAL = 500
EPISODES = 10
STEPS = 20000
DEFAULT_IMAGE_SIZE = 256
TIMESTEPS = 300000
EP_LENGTH = 3075
OBS_TYPE = "internal"
TARGETS = {1: {"bca": [-0.029918, 0.055143, 1.0431],
               "lcca": [0.003474, 0.055143, 1.0357]},
           2: {'bca': [-0.013049, -0.077002, 1.0384],
               'lcca': [0.019936, -0.048568, 1.0315]}}

DEFAULT_CAMERA_CONFIG = {
    "pos": [0.007738, - 0.029034, 1.550]
}


class CathSimEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self,
                 scene: int = 1,
                 obs_type: str = "internal"):

        self.spec = EnvSpec("CathSimEnv-v0")

        utils.EzPickle.__init__(self)

        self.scene = scene
        xml_file = f'scene_{scene}.xml'

        self.obs_type = obs_type
        self.desired_goal = np.array(TARGETS[scene]["bca"])

        if self.obs_type == "image_time":
            self.obs = np.zeros(
                shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))

        """ Inherits from MujocoEnv """

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.top_camera_matrix = self.get_camera_matrix("top_view")

    @ property
    def force_image(self):
        """ computes the force and maps it to an image """
        image_range = 2

        data = self.sim.data
        force_image = np.zeros(
            shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1))

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
                        if (0 <= i <= 255 and
                                0 <= j <= 255):
                            force_image[j, i] = collision_force

        return force_image

    @ property
    def head_pos(self):
        head_pos = self.sim.data.get_body_xpos("B99")
        return head_pos

    def step(self, a):
        """ Performs the simulation step
        """

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        obs['achieved_goal'] = self.head_pos

        reward = self.compute_reward(obs['achieved_goal'], obs['desired_goal'])

        return obs, reward, self.done, {}

    def compute_reward(self, achieved_goal, desired_goal, info=None, delta=0.008, dense=True):
        """ Computes the reward """

        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = distance <= delta
        self.done = bool(success)

        if dense:
            reward = 1.0 if success else -distance
        else:
            reward = 1.0 if success else 0.0

        return reward

    def _get_obs(self):

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

            actuator_forces = data.qfrc_actuator.flat.copy()

            obs = np.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    actuator_forces,
                )
            )

        obs = OrderedDict([
            ('observation', obs),
            ('desired_goal', self.desired_goal),
            ('achieved_goal', self.head_pos),
        ])

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

        targets = ["bca", "lcca"]
        desired_goal = TARGETS[self.scene][targets[random.randint(0, 1)]]
        self.desired_goal = np.array(desired_goal)

        return self._get_obs()

    def point2pixel(self, point, camera_matrix):
        """Transforms from world coordinates to pixel coordinates."""

        x, y, z = point

        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))
        return round(xs/s), round(ys/s)

    def get_image(self, camera_name, mode="rgb"):
        image = self.render("rgb_array", camera_name=camera_name)
        if mode == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        return image

    def print_info(self, link=-1):
        print("Position: ", self.data.qpos[link])
        print("Velocity: ", self.data.qvel[link])
        print("COM Inertia: ", self.data.cinert[link])
        print("COM Velocity: ", self.data.cvel[link])
        print("Actuator Forces: ", self.data.qfrc_actuator[link].shape)


if __name__ == "__main__":
    env = CathSimEnv(scene=1,
                     obs_type="internal")
    print("Observation: ", env.observation_space["observation"])
    print("Achived Goal: ", env.observation_space["achieved_goal"])
    print("Desired Goal: ", env.observation_space["desired_goal"])
    obs = env.reset()
    check_env(env)

    algorithm = ALGOS["ddpg"]

    model = algorithm(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=1,
            goal_selection_strategy="future",
            # IMPORTANT: because the env is not wrapped with a TimeLimit wrapper
            # we have to manually specify the max number of steps per episode
            max_episode_length=100,
            online_sampling=True,
        ),
        verbose=1,
        buffer_size=int(1e6),
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
    )

    model.learn(int(2e5))
    exit()

    obs = env.reset()

    # Evaluate the agent
    episode_reward = 0
    for _ in range(100):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
        episode_reward += reward
        if done or info.get("is_success", False):
            print("Reward:", episode_reward, "Success?",
                  info.get("is_success", False))
            episode_reward = 0.0
            obs = env.reset()
