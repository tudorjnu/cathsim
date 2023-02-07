import math
import cv2
import numpy as np
from pathlib import Path

from dm_control import mjcf
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable
from dm_control.composer.observation.observable import MujocoCamera


_DEFAULT_TIME_LIMIT = 2
_CONTROL_TIMESTEP = .004
_NUM_SUBSTEPS = 3


SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2
TWIST = False
STRETCH = False
FORCE = 300 * SCALE
STIFFNESS = 20
TARGET_POS = (-0.043272, 0.136586, 0.034102)
CONDIM = 1
FRICTION = [0.1]  # default: [1, 0.005, 0.0001]
SPRING = 0.05


TIP_N_BODIES = 2

# OPTIONS
_GRAVITY = [0, 0, -9.81]
_DENSITY = 1000
_VISCOSITY = 0.0009 * 4
_MARGIN = 0.004
_INTEGRATOR = 'implicit'  # euler, implicit, rk4
_CONE = 'pyramidal'  # pyramidal, elliptic
_JACOBIAN = 'sparse'  # dense, sparse
_SOLVER = 'newton'  # cg, newton, pgs
# FLAGS
_MULTICCD = 'disable'
_FRICTIONLOSS = "enable"
_GRAVITY = "enable"

random_state = np.random.RandomState(42)


class Scene(composer.Arena):

    def _build(self,
               name: str = 'arena',
               render_site: bool = False,
               ):
        super()._build(name=name)

        self._mjcf_root.compiler.set_attributes(
            angle='radian',
            meshdir='./meshes',
            autolimits=True,
        )
        self._mjcf_root.option.set_attributes(
            timestep=_CONTROL_TIMESTEP,
            viscosity=_VISCOSITY,  # 0.0009 * 4,
            density=_DENSITY,
            solver=_SOLVER,         # pgs, cg, newton
            integrator=_INTEGRATOR,  # euler, implicit, rk4
            cone=_CONE,          # pyramidal, elliptic
            jacobian=_JACOBIAN,    # dense, sparse
        )

        self._mjcf_root.option.flag.set_attributes(
            multiccd=_MULTICCD,
            frictionloss=_FRICTIONLOSS,
            gravity=_GRAVITY,
        )

        self._mjcf_root.size.set_attributes(
            nconmax=6000,
            njmax=6000,
            nstack=50000000,
        )

        self._mjcf_root.default.site.set_attributes(
            type='sphere',
            size=[0.004],
            rgba=[0.8, 0.8, 0.8, 0],
        )

        self._top_camera = self._mjcf_root.worldbody.add(
            'camera',
            name='top_camera',
            pos=[-0.03, 0.125, 0.15],
            euler=[0, 0, 0],
        )

        self._mjcf_root.asset.add(
            'texture', type="skybox", builtin="gradient", rgb1=[1, 1, 1],
            rgb2=[1, 1, 1], width=256, height=256)
        self._mjcf_root.worldbody.add(
            'light', pos=[0, 0, 10], dir=[20, 20, -20], castshadow=False)

        site = self._mjcf_root.worldbody.add('site', pos=TARGET_POS)

        if render_site:
            site.rgba = self._mjcf_root.default.site.rgba
            site.rgba[-1] = 1

    def regenerate(self, random_state):
        pass


class CameraObservable(MujocoCamera):
    def __init__(self, camera_name, height=128, width=128, corruptor=None,
                 depth=False, preprocess=True, grayscale=True):
        super().__init__(camera_name, height, width, corruptor, depth)
        self._preprocess = preprocess
        self._grayscale = grayscale
        self._dtype = np.float32 if depth or grayscale or preprocess else np.int8
        self._n_channels = 1 if depth or grayscale else 3

    def _callable(self, physics):
        def get_image():
            image = physics.render(  # pylint: disable=g-long-lambda
                self._height, self._width, self._camera_name, depth=self._depth)
            if self._grayscale:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            if self._preprocess and not self._depth:
                image = image / 255.0 - 0.5
            return image

        return get_image


def add_body(
        n: int = 0,
        parent: mjcf.Element = None,  # the parent body
        ref: float = None,  # the reference angle of the joint
        stiffness: float = None,  # the stiffness of the joint
        stretch: bool = False,
        twist: bool = False,
):
    child = parent.add('body', name=f"body_{n}", pos=[0, 0, OFFSET])
    child.add('geom', name=f'geom_{n}')
    j0 = child.add('joint', name=f'J0_{n}', axis=[1, 0, 0])
    j1 = child.add('joint', name=f'J1_{n}', axis=[0, 1, 0])
    if stiffness is not None:
        j0.stiffness = stiffness
        j1.stiffness = stiffness

    return child


class Phantom(composer.Entity):
    def _build(self, xml_path: str = "phantom3.xml", **kwargs):
        cwd = Path(__file__).parent
        model_dir = cwd / 'assets'
        xml_path = model_dir / xml_path
        self._rgba = [111 / 255, 18 / 255, 0 / 255, 0]
        self._mjcf_root = mjcf.from_file(xml_path.as_posix(), False,
                                         model_dir.as_posix(), **kwargs)
        self._mjcf_root.default.geom.set_attributes(
            margin=_MARGIN,
            group=0,
            rgba=self._rgba,
            condim=CONDIM,
            friction=FRICTION,
        )
        self._mjcf_root.default.mesh.set_attributes(
            scale=[SCALE for i in range(3)])
        self._rgba[-1] = 0.3
        self._mjcf_root.find('geom', 'visual').rgba = self._rgba

    @property
    def mjcf_model(self):
        return self._mjcf_root


class Guidewire(composer.Entity):

    def _build(self, n_bodies: int = 80):

        self._length = CYLINDER_HEIGHT * 2 + SPHERE_RADIUS * 2 + OFFSET * n_bodies

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            margin=_MARGIN,
            group=1,
            rgba=[0.1, 0.1, 0.1, 1],
            type='capsule',
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            density=7980,
            condim=CONDIM,
            friction=FRICTION,
            fluidshape='ellipsoid',
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -OFFSET / 2],
            ref=0,
            # damping=0.005,
            stiffness=STIFFNESS,
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
            forcerange=[-FORCE, FORCE],
            kv=10,
        )

        parent = self._mjcf_root.worldbody.add(
            'body',
            name='body_0',
            euler=[-math.pi / 2, 0, math.pi],
            pos=[
                0, -(self._length - 0.015), 0
            ]
        )
        parent.add('geom', name='geom_0')
        parent.add('joint', type='slide', name='slider', range=[-0, 0.2],
                   stiffness=0, damping=2)
        parent.add('joint', type='hinge', name='rotator',
                   stiffness=0, damping=2)
        self._mjcf_root.actuator.add(
            'velocity', joint='slider', name='slider_actuator')
        self._mjcf_root.actuator.add(
            'general', joint='rotator', name='rotator_actuator',
            dyntype=None, gaintype='fixed', biastype='None',
            dynprm=[1, 0, 0], gainprm=[40, 0, 0], biasprm=[2])

        # make the main body
        stiffness = self._mjcf_root.default.joint.stiffness
        for n in range(1, n_bodies):
            parent = add_body(n, parent, stiffness=stiffness)
            stiffness *= 0.995
        # print('stiffness', stiffness)
        self._tip_site = parent.add(
            'site', name='tip_site', pos=[0, 0, OFFSET])

    @property
    def attachment_site(self):
        return self._tip_site

    @property
    def mjcf_model(self):
        return self._mjcf_root

    def _build_observables(self):
        return GuidewireObservables(self)

    @property
    def actuators(self):
        return tuple(self._mjcf_root.find_all('actuator'))

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))


class GuidewireObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Tip(composer.Entity):

    def _build(self, name=None, n_bodies=3):

        if name is None:
            name = 'tip'
        self._mjcf_root = mjcf.RootElement(model=name)

        self._mjcf_root.default.geom.set_attributes(
            margin=_MARGIN,
            group=2,
            rgba=[0.1, 0.1, 0.1, 1],
            size=[SPHERE_RADIUS, CYLINDER_HEIGHT],
            type="capsule",
            condim=CONDIM,
            friction=FRICTION,
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
            name='body_0',
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add('geom', name='geom_0',)
        parent.add('joint', name='T0_0', axis=[0, 0, 1])
        parent.add('joint', name='T1_0', axis=[0, 1, 0])

        for n in range(1, n_bodies):
            parent = add_body(n, parent)

        self.head_geom.name = 'head'

    @property
    def mjcf_model(self):
        return self._mjcf_root

    @property
    def joints(self):
        return tuple(self._mjcf_root.find_all('joint'))

    def _build_observables(self):
        return TipObservables(self)

    @property
    def head_geom(self):
        return self._mjcf_root.find_all('geom')[-1]


class TipObservables(composer.Observables):

    @composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Navigate(composer.Task):

    def __init__(self,
                 phantom: composer.Entity = None,
                 guidewire: composer.Entity = None,
                 tip: composer.Entity = None,
                 obs_type: str = 'full',
                 delta: float = 0.004,  # distance threshold for success
                 dense_reward: bool = True,
                 success_reward: float = 10.0,
                 use_image: bool = False,
                 image_size: int = 480,
                 ):

        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_image = use_image

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
        self._target_pos = TARGET_POS

        # Configure variators
        self._mjcf_variator = variation.MJCFVariator()
        self._physics_variator = variation.PhysicsVariator()

        # Configure and enable observables
        pos_corrptor = noises.Additive(distributions.Normal(scale=0.0001))
        self._guidewire.observables.joint_positions.corruptor = pos_corrptor
        vel_corruptor = noises.Multiplicative(
            distributions.LogNormal(sigma=0.0001))
        self._guidewire.observables.joint_velocities.corruptor = vel_corruptor

        self._guidewire.observables.joint_positions.enabled = True
        self._guidewire.observables.joint_velocities.enabled = True

        self._tip.observables.joint_positions.enabled = True
        self._tip.observables.joint_velocities.enabled = True

        self._task_observables = {}

        if self.use_image:
            self._task_observables['top_camera'] = CameraObservable(
                camera_name='top_camera',
                width=image_size,
                height=image_size,
            )

        for obs in self._task_observables.values():
            obs.enabled = True

        self.control_timestep = _NUM_SUBSTEPS * self.physics_timestep

        self.success = False

    @property
    def root_entity(self):
        return self._arena

    @property
    def task_observables(self):
        return self._task_observables

    def initialize_episode_mjcf(self, random_state):
        self._mjcf_variator.apply_variations(random_state)

    def initialize_episode(self, physics, random_state):
        self._physics_variator.apply_variations(physics, random_state)
        guidewire_pose = variation.evaluate(self._guidewire_initial_pose,
                                            random_state=random_state)
        self._guidewire.set_pose(physics, position=guidewire_pose)
        self.success = False

    def get_reward(self, physics):
        head_pos = physics.named.data.geom_xpos[-1]
        reward = self.compute_reward(head_pos, self._target_pos)
        return reward

    def should_terminate_episode(self, physics):
        return self.success

    @property
    def head_pos(self, physics):
        return physics.named.data.geom_xpos[-1]

    def compute_reward(self, achieved_goal, desired_goal):
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = bool(distance <= self.delta)

        if self.dense_reward:
            reward = self.success_reward if success else -distance
        else:
            reward = self.success_reward if success else -1.0
        self.success = success
        return reward


def run_env(args=None):
    from argparse import ArgumentParser
    from dm_control.viewer import launch

    parser = ArgumentParser()
    parser.add_argument('--n_bodies', type=int, default=80)
    parser.add_argument('--tip_n_bodies', type=int, default=4)

    parsed_args = parser.parse_args(args)

    phantom = Phantom()
    tip = Tip(n_bodies=parsed_args.tip_n_bodies)
    guidewire = Guidewire(n_bodies=parsed_args.n_bodies)

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    launch(env)


if __name__ == "__main__":
    run_env()
