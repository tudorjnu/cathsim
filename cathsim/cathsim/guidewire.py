import math

from dm_control import mjcf
from dm_control import composer
from dm_control.composer.observation import observable
from cathsim.cathsim.common import _MARGIN
from cathsim.cathsim.common import *


SCALE = 1
RGBA = [0.2, 0.2, 0.2, 1]
BODY_DIAMETER = 0.001
SPHERE_RADIUS = (BODY_DIAMETER / 2) * SCALE
CYLINDER_HEIGHT = SPHERE_RADIUS * 1.5
OFFSET = SPHERE_RADIUS + CYLINDER_HEIGHT * 2
FORCE = 300 * SCALE
FRICTION = [0.1]  # default: [1, 0.005, 0.0001]
DENSITY = 6.72e+3


class ContinuousStructure(composer.Entity):

    def add_body(self,
                 n: int = 0,
                 parent: mjcf.Element = None,  # the parent body
                 stiffness: float = None,  # the stiffness of the joint
                 ):
        child = parent.add('body', name=f"body_{n}", pos=[0, 0, OFFSET])
        child.add('geom', name=f'geom_{n}')
        j0 = child.add('joint', name=f'J0_{n}', axis=[1, 0, 0])
        j1 = child.add('joint', name=f'J1_{n}', axis=[0, 1, 0])
        if stiffness is not None:
            j0.stiffness = stiffness
            j1.stiffness = stiffness

        return child

    def get_body(self, n: int = 0):
        return self.mjcf_model.find('body', f'body_{n}')

    def get_joint(self, n: int = 0):
        return self.mjcf_model.find('joint', f'J0_{n}')

    def get_sphere_radius(self):
        return self.thickness / 2

    def get_half_cylinder_height(self):
        return self.thickness * 1.5

    def get_length(self):
        return self.length

    @ property
    def mjcf_model(self):
        return self._mjcf_root


class GuidewireBody(ContinuousStructure):

    def _build(self, n_bodies: int = 80, length: float = 0.1625,
               thickness: float = 0.001, stiffness: float = 20, spring: float = 0.05,
               twist: bool = False, stretch: bool = False):
        """
        Procedurally generate guidewire.

        :param n_bodies: number of n_bodies
        :param length: length of guidewire in meters
        :param thickness: diameter of guidewire in meters
        :param stiffness: stiffness of guidewire
        :param spring: spring constant of guidewire
        :param twist: twisting joints
        :param stretch: stretching joints
        """
        self.scale = SCALE
        self.n_bodies = n_bodies
        self.thickness = thickness
        self.offset = self.get_sphere_radius() + 2 * self.get_half_cylinder_height()
        self.length = self.offset * (self.n_bodies + 1)

        self._mjcf_root = mjcf.RootElement(model="guidewire")

        self._mjcf_root.default.geom.set_attributes(
            margin=_MARGIN,
            group=1,
            rgba=[0.1, 0.1, 0.1, 1],
            type='capsule',
            size=[self.get_sphere_radius(), self.get_half_cylinder_height()],
            density=7980,
            condim=CONDIM,
            friction=FRICTION,
            fluidshape='ellipsoid',
        )

        self._mjcf_root.default.joint.set_attributes(
            type='hinge',
            pos=[0, 0, -self.offset / 2],
            ref=0,
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
            kv=5,
        )

        parent = self._mjcf_root.worldbody.add(
            'body',
            name='body_0',
            euler=[-math.pi / 2, 0, math.pi],
            pos=[0, -(self.length - 0.015), 0]
        )

        parent.add('geom', name='geom_0')
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
            parent = self.add_body(n, parent, stiffness=stiffness)
            stiffness *= 0.995
        self._tip_site = parent.add(
            'site', name='tip_site', pos=[0, 0, offset])

    def _build_observables(self):
        return GuidewireObservables(self)

    @ property
    def attachment_site(self):
        return self._tip_site


class GuidewireObservables(composer.Observables):

    @ composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @ composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class GuidewireTip(ContinuousStructure):

    def _build(self, name='tip', n_bodies=3, thickness=0.001):
        self.scale = SCALE
        self.n_bodies = n_bodies
        self.thickness = thickness
        self.offset = self.get_sphere_radius() + 2 * self.get_half_cylinder_height()
        self.length = self.offset * (self.n_bodies + 1)

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
            name='tip_body_0',
            euler=[0, 0, 0],
            pos=[0, 0, 0],
        )

        parent.add('geom', name='tip_geom_0',)
        parent.add('joint', name='tip_J0_0', axis=[0, 0, 1])
        parent.add('joint', name='tip_J1_0', axis=[0, 1, 0])

        for n in range(1, n_bodies):
            parent = self.add_body(n, parent, name='tip')

        self.head_geom.name = 'head'

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


class GuidewireCreator(composer.Entity):

    def _build(self):
        raise NotImplementedError

    @ composer.observable
    def joint_positions(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qpos', all_joints)

    @ composer.observable
    def joint_velocities(self):
        all_joints = self._entity.mjcf_model.find_all('joint')
        return observable.MJCFFeature('qvel', all_joints)


class Guidewire(GuidewireCreator):

    def _build(self, guidewire_body: GuidewireBody, guidewire_tip: GuidewireTip):
        pass
