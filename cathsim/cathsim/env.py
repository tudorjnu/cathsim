import math
import yaml
import numpy as np
from pathlib import Path

from dm_control import mjcf
from dm_control.mujoco import wrapper
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable
from cathsim.cathsim.phantom import Phantom
from cathsim.cathsim.common import CameraObservable
from cathsim.cathsim.env_utils import distance

with open(Path(__file__).parent / 'env_config.yaml', 'r') as f:
    env_config = yaml.safe_load(f)

option = env_config['option']
option_flag = option.pop('flag')
compiler = env_config['compiler']
visual = env_config['visual']
visual_global = visual.pop('global')
guidewire_config = env_config['guidewire']

BODY_DIAMETER = guidewire_config['diameter'] * guidewire_config['scale']
SPHERE_RADIUS = (BODY_DIAMETER / 2) * guidewire_config['scale']
CYLINDER_HEIGHT = SPHERE_RADIUS * guidewire_config['sphere_to_cylinder_ratio']
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2

TARGET_POS = np.array([-0.043272, 0.136586, 0.034102])
TARGET_POS = np.array([-0.02809, 0.13286, 0.033681])

random_state = np.random.RandomState(42)


class Scene(composer.Arena):

    def _build(self,
               name: str = 'arena',
               render_site: bool = False,
               ):
        super()._build(name=name)

        self._mjcf_root.compiler.set_attributes(**compiler)
        self._mjcf_root.option.set_attributes(**option)
        self._mjcf_root.option.flag.set_attributes(**option_flag)
        self._mjcf_root.visual.set_attributes(**visual)

        self._top_camera = self.add_camera('top_camera',
                                           [-0.03, 0.125, 0.15], [0, 0, 0])
        self._top_camera_close = self.add_camera('top_camera_close',
                                                 [-0.03, 0.125, 0.065], [0, 0, 0])
        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[0.004],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self._mjcf_root.asset.add(
            'texture', type="skybox", builtin="gradient", rgb1=[1, 1, 1],
            rgb2=[1, 1, 1], width=256, height=256)

        self.add_light(pos=[0, 0, 10], dir=[20, 20, -20], castshadow=False)
        site = self.add_site('target', TARGET_POS)

        if render_site:
            site.rgba = self._mjcf_root.default.site.rgba
            site.rgba[-1] = 1

    def regenerate(self, random_state):
        pass

    def add_light(self, pos: list = [0, 0, 0], dir: list = [0, 0, 0], castshadow: bool = False):
        light = self._mjcf_root.worldbody.add('light', pos=pos, dir=dir, castshadow=castshadow)
        return light

    def add_camera(self, name: str, pos: list = [0, 0, 0], euler: list = [0, 0, 0]):
        camera = self._mjcf_root.worldbody.add('camera', name=name, pos=pos, euler=euler)
        return camera

    def add_site(self, name: str, pos: list = [0, 0, 0]):
        site = self._mjcf_root.worldbody.add('site', name=name, pos=pos)
        return site


def add_body(
        n: int = 0,
        parent: mjcf.Element = None,  # the parent body
        stiffness: float = None,  # the stiffness of the joint
        name: str = None,
):
    child = parent.add('body', name=f"{name}_body_{n}", pos=[0, 0, OFFSET])
    child.add('geom', name=f'geom_{n}')
    j0 = child.add('joint', name=f'{name}_J0_{n}', axis=[1, 0, 0])
    j1 = child.add('joint', name=f'{name}_J1_{n}', axis=[0, 1, 0])
    if stiffness is not None:
        j0.stiffness = stiffness
        j1.stiffness = stiffness

    return child


class Guidewire(composer.Entity):

    def _build(self, n_bodies: int = 80):

        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            group=1,
            rgba=guidewire_config['rgba'],
            type='capsule',
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            density=guidewire_config['density'],
            condim=guidewire_config['condim'],
            friction=guidewire_config['friction'],
            fluidshape='ellipsoid',
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            ref=0,
            stiffness=guidewire_config['stiffness'],
            springref=0,
            armature=0.05,
            axis=[0, 0, 1],
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[SPHERE_RADIUS],
            rgba=[0.3, 0, 0, 0.0],
        )

        self._mjcf_root.default.velocity.set_attributes(
            ctrlrange=[-1, 1],
            forcerange=[-guidewire_config['force'], guidewire_config['force']],
            kv=5,
        )

        parent = self._mjcf_root.worldbody.add(
            'body',
            name='guidewire_body_0',
            euler=[-math.pi / 2, 0, math.pi],
            pos=[0, -(self._length - 0.015), 0]
        )
        parent.add('geom', name='guidewire_geom_0')
        parent.add('joint', type='slide', name='slider', range=[-0, 0.2], stiffness=0, damping=2)
        parent.add('joint', type='hinge', name='rotator', stiffness=0, damping=2)
        self._mjcf_root.actuator.add('velocity', joint='slider', name='slider_actuator')
        kp = 40
        self._mjcf_root.actuator.add(
            'general', joint='rotator', name='rotator_actuator',
            dyntype=None, gaintype='fixed', biastype='None',
            dynprm=[1, 0, 0], gainprm=[kp, 0, 0], biasprm=[0, kp, 0])

        # make the main body
        stiffness = self._mjcf_root.default.joint.stiffness
        for n in range(1, n_bodies):
            parent = add_body(n, parent, stiffness=stiffness, name='guidewire')
            stiffness *= 0.995
        self._tip_site = parent.add(
            'site', name='tip_site', pos=[0, 0, OFFSET])

    @ property
    def attachment_site(self):
        return self._tip_site

    @ property
    def mjcf_model(self):
        return self._mjcf_root

    def _build_observables(self):
        return GuidewireObservables(self)

    @ property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @ property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))


class GuidewireObservables(composer.Observables):

    @ composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @ composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Tip(composer.Entity):

    def _build(self, name=None, n_bodies=3):

        if name is None:
            name = 'tip'
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.default.geom.set_attributes(

            group=2,
            rgba=guidewire_config['rgba'],
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            type="capsule",
            condim=guidewire_config['condim'],
            friction=guidewire_config['friction'],
            fluidshape='ellipsoid',
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            springref=math.pi / 3 / n_bodies,
            # ref=math.pi / 5 / n_bodies ,
            damping=0.5,
            stiffness=1,
            armature=0.05,
        )

        parent = self._mjcf_root.worldbody.add(
            'body',
            name='tip_body_0',
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add('geom', name='tip_geom_0',)
        parent.add('joint', name='tip_J0_0', axis=[0, 0, 1])
        parent.add('joint', name='tip_J1_0', axis=[0, 1, 0])

        for n in range(1, n_bodies):
            parent = add_body(n, parent, name='tip')

        self.head_geom.name = 'head'

    @ property
    def mjcf_model(self):
        return self._mjcf_root

    @ property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))

    def _build_observables(self):
        return TipObservables(self)

    @ property
    def head_geom(self):
        return self._mjcf_root.find_all('geom')[-1]


class TipObservables(composer.Observables):

    @ composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @ composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Navigate(composer.Task):

    def __init__(self,
                 phantom: composer.Entity = None,
                 guidewire: composer.Entity = None,
                 tip: composer.Entity = None,
                 delta: float = 0.004,  # distance threshold for success
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 use_pixels: bool = False,
                 use_segment: bool = False,
                 image_size: int = 480,
                 target=None,
                 ):

        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_pixels = use_pixels
        self.use_segment = use_segment
        self.image_size = image_size

        self._arena = Scene("arena")
        if phantom is not None:
            self._phantom = phantom
            self._arena.attach(self._phantom)
        if guidewire is not None:
            self._guidewire = guidewire
            if tip is not None:
                self._tip = tip
                self._guidewire.attach(self._tip)
            self._arena.attach(self._guidewire)

        # Configure initial poses
        self._guidewire_initial_pose = [0, 0, 0]

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0001))
        vel_corruptor = noises.Multiplicative(
            distributions.LogNormal(sigma=0.0001))

        self._task_observables = {}

        if self.use_pixels:
            self._task_observables['pixels'] = CameraObservable(
                camera_name='top_camera',
                width=image_size,
                height=image_size,
            )

        if self.use_segment:
            guidewire_option = wrapper.MjvOption()
            guidewire_option.geomgroup = np.zeros_like(
                guidewire_option.geomgroup)
            guidewire_option.geomgroup[1] = 1  # show the guidewire
            guidewire_option.geomgroup[2] = 1  # show the tip

            self._task_observables['guidewire'] = CameraObservable(
                camera_name='top_camera',
                height=image_size,
                width=image_size,
                scene_option=guidewire_option,
                segmentation=True
            )
            #
            # guidewire_option = wrapper.MjvOption()
            # guidewire_option.geomgroup = np.zeros_like(
            #     guidewire_option.geomgroup)
            # guidewire_option.geomgroup[0] = 1  # show the phantom
            #
            # self._task_observables['phantom'] = CameraObservable(
            #     camera_name='top_camera',
            #     height=image_size,
            #     width=image_size,
            #     scene_option=guidewire_option,
            #     segmentation=True
            # )

        self._task_observables['joint_pos'] = observable.Generic(
            self.get_joint_positions)
        self._task_observables['joint_vel'] = observable.Generic(
            self.get_joint_velocities)

        self._task_observables['joint_pos'].corruptor = pos_corrptor
        self._task_observables['joint_vel'].corruptor = vel_corruptor

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = env_config['num_substeps'] * self.physics_timestep

        self.success = False

        self.guidewire_joints = [
            joint.name for joint in self._guidewire.joints]
        self.tip_joints = [joint.name for joint in self._tip.joints]

        self.set_target(target)

    @ property
    def root_entity(self):
        return self._arena

    @ property
    def task_observables(self):
        return self._task_observables

    @property
    def target_pos(self):
        """The target_pos property."""
        return self._target_pos

    def set_target(self, target) -> None:
        """ target is one of:
            - str: name of the site
            - np.ndarray: target position"""

        if type(target) is str:
            sites = self._phantom.sites
            assert target in sites, f"Target site not found. Valid sites are: {sites.keys()}"
            target = sites[target]
        self._target_pos = target

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        guidewire_pose = variation.evaluate(self._guidewire_initial_pose,
                                            random_state=random_state)
        self._guidewire.set_pose(physics, position=guidewire_pose)
        self.success = False

    def get_reward(self, physics):
        self.head_pos = self.get_head_pos(physics)
        reward = self.compute_reward(self.head_pos, self._target_pos)
        return reward

    def should_terminate_episode(self, physics):
        return self.success

    def get_head_pos(self, physics):
        return physics.named.data.geom_xpos[-1]

    def compute_reward(self, achieved_goal, desired_goal):
        d = distance(achieved_goal, desired_goal)
        success = np.array(d < self.delta, dtype=bool)

        if self.dense_reward:
            reward = np.where(success, self.success_reward, -d)
        else:
            reward = np.where(success, self.success_reward, -1.0)
        self.success = success
        return reward

    def get_joint_positions(self, physics):
        positions = physics.named.data.qpos
        return positions

    def get_joint_velocities(self, physics):
        velocities = physics.named.data.qvel
        return velocities

    def get_contact_forces(self, physics):
        forces = physics.data.qfrc_constraint[0:3]
        forces = np.linalg.norm(forces)
        return forces


def run_env(args=None):
    from argparse import ArgumentParser
    from dm_control.viewer import launch

    parser = ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=80)
    parser.add_argument('--tip_n_bodies', type=int, default=4)
    parser.add_argument('--interact', type=bool, default=True)
    target = 'bca'

    parsed_args = parser.parse_args(args)

    phantom = Phantom()
    tip = Tip(n_bodies=parsed_args.tip_n_bodies)
    guidewire = Guidewire(n_bodies=parsed_args.n_bodies)

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target=target,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    physics = env.physics
    print(phantom.sites[target])
    print(env._task._target_pos)
    # print(env._task.get_head_pos(physics))
    assert (phantom.sites[target] == env._task._target_pos).all(), \
        f"{phantom.sites[target]} != {env._task._target_pos}"
    assert (env._task.get_reward(physics) == env._task.compute_reward(
        env._task.get_head_pos(physics), env._task._target_pos)), \
        f"{env._task.get_reward(physics)} != {env._task.compute_reward(env._task.get_head_pos(physics), env._task._target_pos)}"
    exit()

    def random_policy(time_step):
        del time_step  # Unused.
        return [0, 0]

    # Launch the viewer application.
    # if parsed_args.interact:
    #     from cathsim.cathsim.env_utils import launch
    #     launch(env)
    # else:
    #     launch(env, policy=random_policy)

    # camera = mujoco.Camera(env.physics, 480, 480, 0)
    # print(camera.matrix)
    # exit()

    env.reset()

    for k, v in env.observation_spec().items():
        print(k, v.dtype, v.shape)

    def plot_obs(obs):
        import matplotlib.pyplot as plt
        import cv2
        top_camera = obs['pixels']
        guidewire = obs['guidewire']
        # phantom = obs['phantom']
        # top_camera = cv2.cvtColor(top_camera, cv2.COLOR_RGB2GRAY)

        # plot the phantom and guidewire in subplot
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(top_camera)
        ax[0].axis('off')
        ax[0].set_title('top_camera')
        ax[1].imshow(guidewire)
        ax[1].set_title('guidewire segmentation')
        ax[1].axis('off')
        # ax[2].imshow(phantom)
        # ax[2].set_title('phantom segmentation')
        # ax[2].axis('off')
        plt.show()
        # exit()

        # plt.imsave('./figures/phantom_mask.png', np.squeeze(phantom))
        plt.imsave('./figures/phantom_2.png', top_camera)
        # cv2.imwrite('./figures/phantom.png', top_camera)
        exit()

    for i in range(100):
        action = np.zeros(env.action_spec().shape)
        action[0] = 1
        timestep = env.step(action)
        forces = env.task.get_contact_forces(env.physics)
        plot_obs(timestep.observation)
        exit()
        print('forces', np.round(np.linalg.norm(forces), 2))


if __name__ == "__main__":
    run_env()
