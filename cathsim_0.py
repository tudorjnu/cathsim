import cv2
import time
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from gym import utils
from stable_baselines3.common.env_checker import check_env
from utils import ALGOS
from tqdm import trange

import mujoco_env

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
               'lcca': [0.019936, -0.048568, 1.0315]},
           3: {"bca": [-0.029918, 0.055143, 1.0431],
               "lcca": [0.003474, 0.055143, 1.0357]}}
SCENE = 1
TARGET = "bca"

DEFAULT_CAMERA_CONFIG = {
    "pos": [0.007738, - 0.029034, 1.550]
}


def test_env(env):
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

    def __init__(self,
                 scene: int = SCENE,
                 target: str = TARGET,
                 obs_type: str = OBS_TYPE,
                 ep_length: int = EP_LENGTH,
                 ctrl_cost_weight=0.1,
                 contact_cost_weight=5e-7,
                 contact_cost_range=(-np.inf, 10.0),
                 distance_cost_weight=1,
                 distance_cost_range=(0, 1),
                 exclude_current_positions_from_observation=True):

        utils.EzPickle.__init__(self)

        xml_file = f'scene_{scene}.xml'
        self.target = TARGETS[scene][target]

        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1

        self._ctrl_cost_weight = ctrl_cost_weight
        self._contact_cost_weight = contact_cost_weight
        self._contact_cost_range = contact_cost_range
        self._distance_cost_weight = distance_cost_weight
        self._distance_cost_range = distance_cost_range
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        self.obs_type = obs_type

        if self.obs_type == "image_time":
            self.obs = np.zeros(
                shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 3))

        """ Inherits from MujocoEnv """

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

        self.top_camera_matrix = self.get_camera_matrix("top_view")

    @property
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

    @property
    def contact_cost(self):
        contact_forces = self.force_image.flatten()
        contact_cost = self._contact_cost_weight * \
            np.sum(np.square(contact_forces))
        min_cost, max_cost = self._contact_cost_range
        contact_cost = np.clip(contact_cost, min_cost, max_cost)
        return contact_cost

    @property
    def control_cost(self):
        control_cost = self._ctrl_cost_weight * \
            np.sum(np.square(self.sim.data.ctrl))
        return control_cost

    @property
    def head_pos(self):
        head_pos = self.sim.data.get_body_xpos("B99")
        return head_pos

    @property
    def target_distance(self):
        """Calculates the distance between the head and the target"""
        target = np.array(self.target)
        distance = np.linalg.norm(self.head_pos - target)
        return distance

    @property
    def distance_cost(self):
        target = np.array(self.target)
        distance = np.linalg.norm(self.head_pos - target)
        distance_cost = self._distance_cost_weight * distance
        return distance_cost

    @property
    def done(self):
        done = bool(self.target_distance <=
                    0.008 or self.current_step >= self.ep_length)
        return done

    def step(self, a):
        """ Performs the simulation step
        """

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        # control_cost = self.control_cost
        distance_cost = self.distance_cost
        reward = - distance_cost

        if self.target_distance <= 0.008:
            reward += 300

        done = self.done

        self.current_step += 1

        info = dict(distance_cost=distance_cost)

        return obs, reward, done, info

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
            com_velocity = data.cvel.flat.copy()

            actuator_forces = data.qfrc_actuator.flat.copy()
            external_contact_forces = self.force_image.flat.copy()

            if self._exclude_current_positions_from_observation:
                position = position[2:]

            obs = np.concatenate(
                (
                    position,
                    velocity,
                    com_inertia,
                    com_velocity,
                    actuator_forces,
                    external_contact_forces,
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
        self.current_step = 0
        self.num_resets += 1

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
                     obs_type="internal",
                     ep_length=3072,
                     target="lcca")
    obs = env.reset()

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
            env.print_info()
            # time.sleep(2)
            env.render()
            action = env.action_space.sample()
            print("Action:", action)
            obs, reward, done, info = env.step(action)

    env.close()
