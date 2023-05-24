"""Tests for the Cathsim class."""
from absl.testing import absltest
from absl.testing import parameterized
from dm_control import composer
from dm_control import mjcf
from dm_control.mujoco.wrapper import mjbindings
import numpy as np
from cathsim.cathsim import Scene, Guidewire, Tip, Navigate
from cathsim.wrappers import DMEnvironmentWrapper

mjlib = mjbindings.mjlib


class GuidewireTest(parameterized.TestCase):

    def test_can_compile_and_step_model(self):
        scene = Scene()
        guidewire = Guidewire()
        scene.attach(guidewire)
        physics = mjcf.Physics.from_mjcf_model(scene._mjcf_root)
        physics.step()

    def test_can_attach_tip(self):
        scene = Scene()
        guidewire = Guidewire()
        tip = Tip()
        guidewire.attach(tip)
        scene.attach(guidewire)
        physics = mjcf.Physics.from_mjcf_model(scene._mjcf_root)
        physics.step()


class NavigateTest(parameterized.TestCase):

    def test_can_compile_env(self):
        tip = Tip()
        guidewire = Guidewire()

        task = Navigate(
            guidewire=guidewire,
            tip=tip,
        )

        env = composer.Environment(
            task=task,
            time_limit=2000,
            random_state=np.random.RandomState(42),
            strip_singleton_obs_buffer_dim=True,
        )
        action_spec = env.action_spec()
        env.reset()
        action = np.random.uniform(action_spec.minimum,
                                   action_spec.maximum,
                                   size=action_spec.shape)
        env.step(action)

    def test_action_space(self):
        tip = Tip()
        guidewire = Guidewire()

        task = Navigate(
            guidewire=guidewire,
            tip=tip,
        )

        env = composer.Environment(
            task=task,
            time_limit=2000,
            random_state=np.random.RandomState(42),
            strip_singleton_obs_buffer_dim=True,
        )
        action_spec = env.action_spec()
        self.assertTrue(action_spec.shape == (2,))

    def test_observation_space(self):
        pass

    def test_image_observation_space(self):

        guidewire_n_bodies = 10
        tip_n_bodies = 4
        tip = Tip(n_bodies=tip_n_bodies)
        guidewire = Guidewire(n_bodies=guidewire_n_bodies)

        task = Navigate(
            guidewire=guidewire,
            tip=tip,
            image_size=64,
            use_pixels=True,
        )

        env = composer.Environment(
            task=task,
            time_limit=2000,
            random_state=np.random.RandomState(42),
            strip_singleton_obs_buffer_dim=True,
        )

        env = DMEnvironmentWrapper(env)

        self.assertTrue(env.observation_space['pixels'].shape == (64, 64, 3))


if __name__ == "__main__":
    absltest.main()
