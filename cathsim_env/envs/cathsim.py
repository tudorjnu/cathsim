import os
import mujoco_py
import numpy as np
from collections import OrderedDict
from gym import utils
from gym import spaces
from gym.envs.mujoco.mujoco_env import MujocoEnv

DEFAULT_CAMERA_CONFIG = {}

TARGETS = {1: {"bca": np.array([-0.029918, 0.035143, 1.0431]),
               "lcca": np.array([0.003474, 0.035143, 1.0357])},
           2: {'bca': np.array([-0.013049, -0.077002, 1.0384]),
               'lcca': np.array([0.019936, -0.048568, 1.0315])}}


def convert_observation_to_space(observation):
    if isinstance(observation, dict):
        space = spaces.Dict(
            OrderedDict(
                [
                    (key, convert_observation_to_space(value))
                    for key, value in observation.items()
                ]
            )
        )
    elif isinstance(observation, np.ndarray):
        if len(observation.shape) == 1:
            low = np.full(observation.shape, -float("inf"), dtype=np.float32)
            high = np.full(observation.shape, float("inf"), dtype=np.float32)
            space = spaces.Box(low, high, dtype=observation.dtype)
        else:
            space = spaces.Box(low=0, high=255,
                               shape=observation.shape,
                               dtype=np.uint8)

    else:
        raise NotImplementedError(type(observation), observation)

    return space


class CathSimEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
            "single_rgb_array",
            "single_depth_array",
        ],
        "render_fps": 25,
    }

    def __init__(self,
                 scene: int = 1,
                 target: str = "bca",
                 obs_type: str = "internal",
                 image_size: int = 128,
                 delta: float = 0.008,
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 compute_force: bool = False,
                 **kwargs):

        utils.EzPickle.__init__(self)

        self.scene = scene
        self.target = TARGETS[scene][target]
        self.obs_type = obs_type
        self.image_size = image_size
        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.compute_force = compute_force
        self.image_size = image_size

        """ Inherits from MujocoEnv """

        xml_file = f'scene_{scene}.xml'
        path = os.getcwd()
        xml_path = os.path.join(path, "cathsim_env", "envs", "assets", xml_file)
        MujocoEnv.__init__(self, xml_path, 5)

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

        if self.obs_type == "image":
            obs = self.render("rgb_array", camera_name="top_camera",
                              width=self.image_size, height=self.image_size)
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

        return round(xs / s), round(ys / s)

    def get_camera_matrix(self, camera_name):
        """ calculates the camera matrix for a camera """
        width, height = (self.image_size, self.image_size)

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


if __name__ == "__main__":

    env = CathSimEnv(scene=1,
                     obs_type="image",
                     target="lcca",
                     image_size=1028)

    obs = env.reset()
    print(obs.shape)
    print(obs)
    for _ in range(2000):
        action = env.action_space.sample()
        obs, reward, done, _ = env.step(action)
