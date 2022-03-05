import os

import cv2
import gym
import matplotlib.pyplot as plt
import mujoco_py
import numpy as np
from gym import utils
from stable_baselines3.common.env_checker import check_env
from tqdm import trange

import mujoco_env

IMAGES_SAVING_INTERVAL = 500
EPISODES = 10
STEPS = 5000
DEFAULT_IMAGE_SIZE = 256
TIMESTEPS = 300000
EP_LENGTH = 1000
OBS_TYPE = "internal"
SCENE = "scene_viz"
TARGET = "lca"


# xml_path = os.path.join(os.getcwd(), "assets/scene_2.xml")


class CatheterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """CatheterEnv."""

    def __init__(self, scene=SCENE, obs_type=OBS_TYPE,
                 target=TARGET, ep_length=EP_LENGTH):

        self.obs_type = obs_type
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1
        self.top_camera_matrix = np.array([[-309.01933598, 0., 127.5, -195.23380838],
                                           [0., 309.01933598, 127.5, -188.6529326],
                                           [0., 0., 1., -1.55]])
        # print(self.top_camera_matrix)

        if scene == "scene_1" or scene == "scene_viz":
            if target == "bca":
                self.target = [-0.029918, 0.055143, 1.0431]
            elif target == "lca":
                self.target = [0.003474, 0.055143, 1.0357]

        elif scene == "scene_2":
            if target == "bca":
                self.target = [-0.013049, -0.077002, 1.0384]
            elif target == "lca":
                self.target = [0.019936, -0.048568, 1.0315]

        xml_path = f"{scene}.xml"

        """ Inherits from MujocoEnv """
        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

        # self.top_camera_matrix = self.get_camera_matrix("top_view")

        if self.obs_type == "image_1":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(256, 256, 1),
                                                    dtype=np.uint8)
        elif self.obs_type == "image_2":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(256, 256, 2),
                                                    dtype=np.uint8)
        elif self.obs_type == "image_time":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(256, 256, 3),
                                                    dtype=np.uint8)

        self.reset()

    def step(self, a):
        """ Performs the simulation step

        :param a:
        """

        # coefficients
        ctrl_cost_coeff = -1 * (10 ** -3)
        force_coeff = - 0.001
        distance_coeff = -1

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        # calculate the distance between head and target
        distance, head_pos = self.get_target_distance()

        # calculate the force and force image
        force_image, forces = self.get_collision_force()

        # control reward
        reward_ctrl = ctrl_cost_coeff * np.square(a).sum()
        # distance reward
        reward_distance = distance_coeff * distance
        # force reward
        max_force = np.max(force_image)
        mean_force = 0
        if force_image.any():
            mean_force = np.true_divide(
                force_image.sum(), (force_image != 0).sum())
        reward_force = force_coeff * mean_force
        # total reward
        reward = reward_distance
        # reward = -1 + reward_distance

        self.current_step += 1

        done = bool(distance <= 0.008 or self.current_step >= self.ep_length)

        self.info = dict(reward_distance=reward_distance,
                         reward_force=reward_force,
                         mean_force=mean_force,
                         reward_ctrl=reward_ctrl,
                         max_force=max_force,
                         force_image=force_image,
                         forces=forces,
                         done=done)

        if self.obs_type == 'image':
            self.info['original_image'] = ob

        return ob, reward, done, self.info

    def _get_obs(self):

        if self.obs_type == "image_1":
            obs = self.get_image('top_view', mode='gray')

        elif self.obs_type == "image_2":
            top = self.get_image('top_view', mode='gray')
            side = self.get_image('side_view', mode='gray')

            obs = np.concatenate([top, side], axis=-1)

        elif self.obs_type == "image_time":
            if self.current_step == 0:
                self.obs = np.zeros(shape=(256, 256, 3))
            self.obs[:, :, 0] = self.obs[:, :, 1]
            self.obs[:, :, 1] = self.obs[:, :, 2]
            self.obs[:, :, 2] = self.get_image(
                'top_view', mode='gray')[:, :, 0]
            obs = self.obs

        else:
            obs = np.append(
                self.state_vector(), self.get_target_distance()[0])
        return obs

    def reset(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0.001,
                                     high=0.001, size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0.001,
                                     high=0.001, size=self.model.nv),
        )
        self.current_step = 0
        self.num_resets += 1

        return self._get_obs()

    def get_collision_force(self):
        """ computes the force and maps it to an image """
        image_range = 2
        forces = []

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
                forces.append(collision_force)
                for i in range(collision_pos[0]-image_range,
                               collision_pos[0]+image_range):
                    for j in range(collision_pos[1]-image_range,
                                   collision_pos[1]+image_range):
                        if (0 <= i <= 255 and
                                0 <= j <= 255):
                            force_image[j, i] = collision_force

        return force_image, forces

    def get_target_distance(self, target=None):
        """Calculates the distance between the head and the target"""
        if target is None:
            target = self.target
        head_pos = self.sim.data.get_body_xpos("B59")  # change this for head
        target = np.array(target)
        distance = np.linalg.norm(head_pos - target)
        return distance, head_pos

    def get_camera_matrix(self, camera_name, height=DEFAULT_IMAGE_SIZE,
                          width=DEFAULT_IMAGE_SIZE):
        """ calculates the camera matrix for a camera """
        camera_id = self.sim.model.camera_name2id(camera_name)
        # camera parameters
        fov = self.sim.model.cam_fovy[camera_id]
        pos = self.sim.data.get_camera_xpos(camera_name)
        rot = self.sim.data.get_camera_xmat(camera_name)

        # Translation matrix (4x4).
        translation = np.eye(4)
        translation[0:3, 3] = -pos

        # Rotation matrix (4x4).
        rotation = np.eye(4)
        rotation[0:3, 0:3] = rot

        # Focal transformation matrix (3x4).
        focal_scaling = (1./np.tan(np.deg2rad(fov)/2)) * height / 2.0
        focal = np.diag([-focal_scaling, focal_scaling, 1.0, 0])[0:3, :]

        # Image matrix (3x3).
        image = np.eye(3)
        image[0, 2] = (width - 1) / 2.0
        image[1, 2] = (height - 1) / 2.0

        camera_matrix = image @ focal @ rotation @ translation

        return camera_matrix

    def point2pixel(self, point, camera_matrix):
        """Transforms from world coordinates to pixel coordinates."""
        # Point
        x, y, z = point

        xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))
        return round(xs/s), round(ys/s)

    def get_image(self, camera_name, mode="rgb"):
        image = self.render("rgb_array", camera_name=camera_name)
        if mode == "gray":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[:, :, np.newaxis]

        return image


def save_image_force(original_image, force_image, episode,
                     step, interval=50):
    if step % interval == 0 and step != 0:
        filename = f"{IMAGES_PATH}/{episode}_{step}"
        force_image = force_image
        x = np.concatenate([original_image, force_image], axis=-1)

        np.save(filename, x)


def test_env(env):
    check_env(env, warn=True)
    print("Observation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    print("Action space:", env.action_space)
    action = env.action_space.sample()
    # print("Sampled action:", action)
    obs, reward, done, info = env.step(action)
    print("obs_shape:", obs.shape, "\nreward:", reward, "\ndone:", done)
    for key, value in info.items():
        if key != "force_image":
            print(f'{key}: {value}')
        else:
            print(f'{key}: {value.shape}')
    print("Degrees of Freedom:", env.sim.model.nv)
    obs = env.reset()


if __name__ == "__main__":

    IMAGES_PATH = "./data/images"
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    env = CatheterEnv()
    test_env(env)
    for episode in trange(EPISODES):
        obs = env.reset()
        done = False
        for i in range(EP_LENGTH):
            env.render()
            # image = env.get_image("top_cath", mode="gray")
            # plt.imsave("./data/Figures/a_2.png",
            # np.squeeze(image, axis=-1), cmap="gray")
            # exit()
            # print(image.shape)
            # if i % 40 == 0:
            # plt.imshow(image, cmap="gray")
            # plt.axis(False)
            # plt.show()
            # plt.imsave(
            # f"./data/Figures/Catheter/close-up/{i}.png", np.squeeze(image, axis=-1), cmap="gray")
            # exit()
            action = env.action_space.sample()  # this takes random actions
            action = np.zeros_like(action)
            # qpos = np.zeros_like(env.model.nq)
            # qvel = np.zeros_like(env.model.nv)

            # x = 1
            # for i in range(10, 21, 2):
            # if x % 2 == 0:
            # action[i] = -1
            # else:
            # action[i] = 1
            # x += 1
            observation, reward, done, info = env.step(action)
        # exit()

    env.close()
