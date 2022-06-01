import os
from collections import OrderedDict
from os import path

import gym
import numpy as np
from gym import error, spaces
from gym.utils import seeding

try:
    import mujoco_py
except ImportError as e:
    raise error.DependencyNotInstalled(
        "{}. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)".format(
            e
        )
    )

DEFAULT_SIZE = 128


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


class MujocoEnv(gym.GoalEnv):
    """Superclass for all MuJoCo environments."""

    def __init__(self, model_path, frame_skip, image_size=DEFAULT_SIZE):
        if model_path.startswith("/"):
            fullpath = model_path
        else:
            fullpath = os.path.join(os.path.dirname(
                __file__), "assets", model_path)
        if not path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.frame_skip = frame_skip
        self.model = mujoco_py.load_model_from_path(fullpath)
        self.sim = mujoco_py.MjSim(self.model)
        self.data = self.sim.data
        self.viewer = None
        self._viewers = {}

        self.metadata = {
            "render.modes": ["human", "rgb_array", "depth_array"],
            "video.frames_per_second": int(np.round(0.01 / self.dt)),
        }

        self.init_qpos = self.sim.data.qpos.ravel().copy()
        self.init_qvel = self.sim.data.qvel.ravel().copy()

        self._set_action_space()

        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done

        self._set_observation_space(observation)
        self.top_camera_matrix = self.get_camera_matrix("top_view")

        self.seed()

    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # methods to override:
    # ----------------------------

    def reset_model(self):
        """
        Reset the robot degrees of freedom (qpos and qvel).
        Implement this in each subclass.
        """
        raise NotImplementedError

    def viewer_setup(self):
        """
        This method is called when the viewer is initialized.
        Optionally implement this method, if you need to tinker with camera position
        and so forth.
        """
        pass

    # -----------------------------

    def reset(self):
        self.sim.reset()
        ob = self.reset_model()
        return ob

    def set_state(self, qpos, qvel):
        assert qpos.shape == (
            self.model.nq,) and qvel.shape == (self.model.nv,)
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(
            old_state.time, qpos, qvel, old_state.act, old_state.udd_state
        )
        self.sim.set_state(new_state)
        self.sim.forward()

    @property
    def dt(self):
        return self.model.opt.timestep * self.frame_skip

    def do_simulation(self, ctrl, n_frames):
        self.sim.data.ctrl[:] = ctrl
        for _ in range(n_frames):
            self.sim.step()

    def render(
        self,
        mode="human",
        camera_id=None,
        camera_name=None,
    ):
        width, height = (self.image_size, self.image_size)
        if mode == "rgb_array" or mode == "depth_array":
            if camera_id is not None and camera_name is not None:
                raise ValueError(
                    "Both `camera_id` and `camera_name` cannot be"
                    " specified at the same time."
                )

            no_camera_specified = camera_name is None and camera_id is None
            if no_camera_specified:
                camera_name = "track"

            if camera_id is None and camera_name in self.model._camera_name2id:
                camera_id = self.model.camera_name2id(camera_name)

            self._get_viewer(mode).render(width, height, camera_id=camera_id)

        if mode == "rgb_array":
            # window size used for old mujoco-py:
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=False)
            # original image is upside-down, so flip it
            return data[::-1, :, :]
        elif mode == "depth_array":
            self._get_viewer(mode).render(width, height)
            # window size used for old mujoco-py:
            # Extract depth part of the read_pixels() tuple
            data = self._get_viewer(mode).read_pixels(
                width, height, depth=True)[1]
            # original image is upside-down, so flip it
            return data[::-1, :]
        elif mode == "human":
            self._get_viewer(mode).render()

    def close(self):
        if self.viewer is not None:
            # self.viewer.finish()
            self.viewer = None
            self._viewers = {}

    def _get_viewer(self, mode):
        self.viewer = self._viewers.get(mode)
        if self.viewer is None:
            if mode == "human":
                self.viewer = mujoco_py.MjViewer(self.sim)
            elif mode == "rgb_array" or mode == "depth_array":
                self.viewer = mujoco_py.MjRenderContextOffscreen(
                    self.sim, -1)

            self.viewer_setup()
            self._viewers[mode] = self.viewer
        return self.viewer

    def get_body_com(self, body_name):
        return self.data.get_body_xpos(body_name)

    def state_vector(self):
        return np.concatenate([self.sim.data.qpos.flat, self.sim.data.qvel.flat])

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
