import yaml
import numpy as np
from pathlib import Path

from dm_control.composer.observation.observable import MujocoCamera
from dm_control.mujoco import wrapper
from dm_env import specs

config_path = Path(__file__).parent / 'env_config.yaml'
with open(config_path.as_posix()) as f:
    env_config = yaml.safe_load(f)


SCALE = 1


def make_scene(geom_groups: list):
    scene_option = wrapper.MjvOption()
    for geom_group in geom_groups:
        scene_option.geomgroup[geom_group] = True
    return scene_option


def normalize_rgba(rgba: list):
    new_rgba = [c / 255. for c in rgba]
    new_rgba[-1] = rgba[-1]
    return new_rgba


def point2pixel(point, camera_matrix: np.ndarray = None):
    """Transforms from world coordinates to pixel coordinates."""
    if camera_matrix is None:
        camera_matrix = np.array([[-96.56854249, 0., 39.5, - 8.82205627],
                                  [0., 96.56854249, 39.5, - 17.99606781],
                                  [0., 0., 1., - 0.15]])

        # camera_matrix = np.array([
        #     [-5.79411255e+02, 0.00000000e+00, 2.39500000e+02, - 5.33073376e+01],
        #     [0.00000000e+00, 5.79411255e+02, 2.39500000e+02, - 1.08351407e+02],
        #     [0.00000000e+00, 0.00000000e+00, 1.00000000e+00, - 1.50000000e-01]
        # ])
    x, y, z = point
    xs, ys, s = camera_matrix.dot(np.array([x, y, z, 1.0]))

    return np.array([round(xs / s), round(ys / s)]).asttype(np.int8)


class CameraObservable(MujocoCamera):
    def __init__(self, camera_name, height=128, width=128, corruptor=None,
                 depth=False, preprocess=False, grayscale=False,
                 segmentation=False, scene_option=None):
        super().__init__(camera_name, height, width)
        self._dtype = np.uint8
        self._n_channels = 1 if segmentation else 3
        self._preprocess = preprocess
        self.scene_option = scene_option
        self.segmentation = segmentation

    def _callable(self, physics):
        def get_image():
            image = physics.render(  # pylint: disable=g-long-lambda
                self._height, self._width, self._camera_name, depth=self._depth,
                scene_option=self.scene_option, segmentation=self.segmentation)
            if self.segmentation:
                geom_ids = image[:, :, 0]
                if np.all(geom_ids == -1):
                    return np.zeros((self._height, self._width, 1), dtype=self._dtype)
                geom_ids = geom_ids.astype(np.float64) + 1
                geom_ids = geom_ids / geom_ids.max()
                image = 255 * geom_ids
                image = np.expand_dims(image, axis=-1)
            image = image.astype(self._dtype)
            return image
        return get_image

    @property
    def array_spec(self):
        return specs.BoundedArray(
            shape=(self._height, self._width, self._n_channels),
            dtype=self._dtype,
            minimum=0,
            maximum=255,
        )
