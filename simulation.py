import numpy as np
from tqdm import trange
from stable_baselines3.common.env_checker import check_env
import cv2
import mujoco_env
import gym
from gym import utils
import os
import mujoco_py

DEFAULT_IMAGE_SIZE = 256


xml_path = os.path.join(os.getcwd(), "assets/scene.xml")


class CatheterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """CatheterEnv."""

    def __init__(self, obs_type="image", ep_length=20000):
        self.obs_type = obs_type
        self.ep_length = ep_length
        self.current_step = 0
        self.num_resets = -1

        """ Inherits from MujocoEnv """
        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

        self.top_camera_matrix = self.get_camera_matrix("top_view")

        if self.obs_type == "image":
            self.observation_space = gym.spaces.Box(low=0, high=255,
                                                    shape=(256, 256, 1),
                                                    dtype=np.uint8)

        self.reset()

    def step(self, a):
        """ Performs the simulation step

        :param a:
        """

        # coefficients
        ctrl_cost_coeff = -0.000001
        force_coeff = -0.01
        distance_coeff = -0.1

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()

        # calculate the distance between head and target
        distance = self.get_target_distance()

        # calculate the force and force image
        force_image = self.get_collision_force()

        # control reward
        reward_ctrl = ctrl_cost_coeff * np.square(a).sum()
        # distance reward
        reward_distance = distance_coeff * distance
        # force reward
        reward_force = force_coeff * force_image.sum()
        # total reward
        reward = reward_distance + reward_ctrl

        self.current_step += 1
        done = bool(distance <= 0.01 or self.current_step >= self.ep_length)

        self.info = dict(reward_distance=reward_distance,
                         reward_force=reward_force,
                         reward_ctrl=reward_ctrl,
                         force_image=force_image)

        if self.obs_type == 'image':
            self.info['original_image'] = ob

        return ob, reward, done, self.info

    def _get_obs(self):
        if self.obs_type == "image":
            obs = self.get_image('top_view', mode='gray')
        else:
            qpos = self.sim.data.qpos
            # print("Qpos:\n", f"\t - {len(qpos)}\n", f"\t - {qpos[4]}")
            qvel = self.sim.data.qvel
            # print("Qvel:\n", f"\t - {len(qvel)}\n",f"\t - {qvel}")
            obs = np.concatenate([qpos.flat, qvel.flat])
        return obs

    def reset(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0., high=0., size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0., high=0., size=self.model.nv),
        )
        self.current_step = 0
        self.num_resets += 1

        return self._get_obs()

    def get_collision_force(self):
        """ computes the force and maps it to an image """
        image_range = 2

        data = self.sim.data
        # print('number of contacts', data.ncon)
        force_image = np.zeros(
            shape=(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE, 1))

        # for all available contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            # contact_distance = contact.dist
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

    def get_target_distance(self, target=[-0.029918, 0.055143, 1.0431]):
        "Calculates the distance between the head and the target"
        head_pos = self.sim.data.get_body_xpos("B99")
        target = np.array(target)
        distance = np.linalg.norm(head_pos - target)
        return distance

    def get_camera_matrix(self, camera_name, height=DEFAULT_IMAGE_SIZE,
                          width=DEFAULT_IMAGE_SIZE):
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
        force_image = force_image  # / 0.244461324375  # normalise
        x = np.concatenate([original_image, force_image], axis=-1)

        np.save(filename, x)


def test_env(env):
    check_env(env, warn=True)
    print("Observation space:", env.observation_space)
    print("Shape:", env.observation_space.shape)
    print("Action space:", env.action_space)
    action = env.action_space.sample()
    print("Sampled action:", action)
    obs, reward, done, info = env.step(action)
    print(obs.shape, reward, done, info)
    print("Degrees of Freedom:", env.sim.model.nv)
    obs = env.reset()


if __name__ == "__main__":

    IMAGES_PATH = "./data/images"
    if not os.path.exists(IMAGES_PATH):
        os.makedirs(IMAGES_PATH)

    IMAGES_SAVING_INTERVAL = 500
    EPISODES = 10
    STEPS = 50

    env = CatheterEnv(obs_type="dsa", ep_length=STEPS)
    test_env(env)
    # exit()
    for episode in trange(EPISODES):
        obs = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # this takes random actions
            action[0] = 0.2
            observation, reward, done, info = env.step(action)
            # original_image = info['original_image']
            force_image = info['force_image']
            total_force = info['reward_force']
            if total_force != 0:
                pass
                # save_image_force(original_image, force_image, episode, step)

    env.close()
