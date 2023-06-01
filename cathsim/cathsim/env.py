import math
import cv2
import yaml
import numpy as np
from pathlib import Path


from dm_control import mjcf
from dm_control import mujoco
from dm_control.mujoco import wrapper
from dm_control import composer
from dm_control.composer import variation
from dm_control.composer.variation import distributions, noises
from dm_control.composer.observation import observable
from cathsim.cathsim.phantom import Phantom
from cathsim.cathsim.common import CameraObservable, point2pixel
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

random_state = np.random.RandomState(42)


def make_scene(geom_groups: list):
    scene_option = wrapper.MjvOption()
    scene_option.geomgroup = np.zeros_like(
        scene_option.geomgroup)
    for geom_group in geom_groups:
        scene_option.geomgroup[geom_group] = True
    return scene_option


def filter_mask(segment_image: np.ndarray):
    geom_ids = segment_image[:, :, 0]
    geom_ids = geom_ids.astype(np.float64) + 1
    geom_ids = geom_ids / geom_ids.max()
    segment_image = 255 * geom_ids
    return segment_image


def generate_random_point(mask: np.array, x_min: int, x_max: int,
                          y_min: int, y_max: int) -> tuple:
    """ Generate a random point within the given rectangle within the mask."""

    # get all valid coordinates within the specified rectangle
    coords = np.argwhere(mask[y_min:y_max, x_min:x_max] == 1)

    # return None if there are no valid pixels
    if len(coords) == 0:
        print("No valid pixels found in the given range.")
        return None

    # randomly select one of the coordinates
    random_index = np.random.randint(len(coords))
    y_coord, x_coord = coords[random_index]

    # adjust for the offset of the rectangle
    x_coord += x_min
    y_coord += y_min

    return (x_coord, y_coord)


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
                 use_phantom_segment: bool = False,
                 image_size: int = 80,
                 sample_target: bool = False,
                 visualize_sites: bool = False,
                 target=None,
                 ):

        self.delta = delta
        self.dense_reward = dense_reward
        self.success_reward = success_reward
        self.use_pixels = use_pixels
        self.use_segment = use_segment
        self.use_phantom_segment = use_phantom_segment
        self.image_size = image_size
        self.sample_target = sample_target
        self.visualize_sites = visualize_sites

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
            guidewire_option = make_scene([1, 2])

            self._task_observables['guidewire'] = CameraObservable(
                camera_name='top_camera',
                height=image_size,
                width=image_size,
                scene_option=guidewire_option,
                segmentation=True
            )

        if self.use_phantom_segment:
            phantom_option = make_scene([0])
            self._task_observables['phantom'] = CameraObservable(
                camera_name='top_camera',
                height=image_size,
                width=image_size,
                scene_option=phantom_option,
                segmentation=True
            )

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
        self.camera_matrix = None

        if self.visualize_sites:
            sites = self._arena.mjcf_model.find_all('site')
            for site in sites:
                print(site.name)
                site.rgba = [1, 0, 0, 1]

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
        if self.sample_target:
            self.set_target(self.get_random_target())

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

    def get_force(self, physics):
        forces = physics.data.qfrc_constraint[0:3]
        forces = np.linalg.norm(forces)
        return forces

    def get_contact_forces(self, physics, threshold=0.01, to_pixels=True, image_size=64):
        if self.camera_matrix is None:
            self.camera_matrix = self.get_camera_matrix(physics, image_size)
        data = physics.data
        forces = {'pos': [], 'force': []}
        for i in range(data.ncon):
            if data.contact[i].dist < 0.002:
                force = data.contact_force(i)[0][0]
                if abs(force) > threshold:
                    pass
                else:
                    forces['force'].append(force)
                    pos = data.contact[i].pos
                    if to_pixels is not None:
                        pos = point2pixel(pos, self.camera_matrix)
                    forces['pos'].append(pos)
        return forces

    def get_camera_matrix(self, physics, image_size: int = None, camera_id=0):
        from dm_control.mujoco.engine import Camera
        if image_size is None:
            image_size = self.image_size
        camera = Camera(physics, height=image_size, width=image_size, camera_id=camera_id)
        return camera.matrix

    def get_phantom_mask(self, physics, image_size: int = None, camera_id=0):
        scene_option = make_scene([0])
        if image_size is None:
            image_size = self.image_size
        image = physics.render(height=image_size,
                               width=image_size,
                               camera_id=camera_id,
                               scene_option=scene_option)
        mask = filter_mask(image)
        return mask

    def get_guidewire_mask(self, physics, image_size: int = None, camera_id=0):
        scene_option = make_scene([1, 2])
        if image_size is None:
            image_size = self.image_size
        image = physics.render(height=image_size,
                               width=image_size,
                               camera_id=camera_id,
                               scene_option=scene_option)
        mask = filter_mask(image)
        return mask

    def get_random_target(self):
        sites = self._phantom.sites
        site = np.random.choice(list(sites.keys()))
        target = sites[site]
        return target

    # def sample_target(self):
    #     goal = generate_random_point(self.phantom_mask, 10, 70, 30, 60)
    #     return goal


def run_env(args=None):
    from argparse import ArgumentParser
    from dm_control.viewer import launch

    parser = ArgumentParser()
    parser.add_argument('--interact', type=bool, default=True)
    parser.add_argument('--phantom', default='phantom3', type=str)
    parser.add_argument('--target', default='bca', type=str)

    parsed_args = parser.parse_args(args)

    phantom = Phantom(parsed_args.phantom + '.xml')

    tip = Tip()
    guidewire = Guidewire()

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target=parsed_args.target,
        visualize_sites=True,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    def random_policy(time_step):
        del time_step  # Unused
        return [0, 0]

    # Launch the viewer application.
    if parsed_args.interact:
        from cathsim.cathsim.env_utils import launch
        launch(env)
    else:
        launch(env, policy=random_policy)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    phantom_name = 'phantom4'
    phantom = Phantom(phantom_name + '.xml')
    tip = Tip()
    guidewire = Guidewire()

    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_pixels=True,
        use_segment=True,
        target='bca',
        sample_target=True,
        image_size=480,
        visualize_sites=True,
    )

    env = composer.Environment(
        task=task,
        time_limit=2000,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    def random_policy(time_step):
        del time_step  # Unused
        return [0, 0]

    # loop 2 episodes of 2 steps
    for episode in range(2):
        time_step = env.reset()
        print(env._task.get_camera_matrix(env.physics), 480)
        for step in range(2):
            action = random_policy(time_step)
            img = env.physics.render(height=480, width=480, camera_id=0)
            plt.imshow(img)
            plt.imsave(f'../figures/{phantom_name}.png', img)
            exit()
            plt.show()
            time_step = env.step(action)
            print(env._task._target_pos)
