import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from dm_control.viewer.application import Application


def process_trajectory(data):
    from imitation.data.types import Trajectory
    acts = data.pop('action')[:-1]
    obs = []
    for key, value in data.items():
        obs.append(value)
    obs = np.concatenate(obs, axis=1)
    return Trajectory(obs, acts, None, terminal=True)


def process_transitions(trial_path: str, images: bool = False):
    from imitation.data.rollout import flatten_trajectories
    trial_path = Path(trial_path)
    trajectories = []
    for episode_path in trial_path.iterdir():
        print('Processing: ', episode_path)
        data = np.load(episode_path / "trajectory.npz", allow_pickle=True)
        data = dict(data)
        if images:
            data.setdefault('pixels', [])
            images_path = episode_path / "images"
            for image_path in images_path.iterdir():
                data['pixels'].append(plt.imread(image_path)[0].flatten())
        trajectories.append(process_trajectory(data))
    transitions = flatten_trajectories(trajectories)
    print(
        f'Processed {len(trajectories)} trajectories ({len(transitions)} transitions)')
    trajectory_lengths = [len(traj) for traj in trajectories]
    print('mean trajectory length:', np.mean(trajectory_lengths))

    return transitions


def make_env(flatten_obs: bool = True, time_limit: int = 200,
             normalize_obs: bool = True, frame_stack: int = 1,
             render_kwargs: dict = None, env_kwargs: dict = None,
             gym_version: str = 'gym', wrap_monitor: bool = False,
             task_kwargs: dict = {}):
    """
    Create a gym environment from cathsim, dm_control environment.

    :param flatten_obs: flattens the observation space
    :param time_limit: sets a time limit to the environment
    :param normalize_obs: normalizes the observation space
    :param frame_stack: stacks n frames of the environment
    :param render_kwargs: dict of kwargs for the render function. Valid keys are:
        from_pixels: bool, if True, render from pixels
        width: int, width of the rendered image
        height: int, height of the rendered image
        camera_id: int, camera id to use
    :param env_kwargs: dict of kwargs for the environment. Valid keys are:
        None for now
    :param gym_version: gyn or gymnasium
    """
    from cathsim.env import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer
    if gym_version == 'gym':
        from gym import wrappers
        from cathsim.wrappers.wrapper_gym import DMEnv
    elif gym_version == 'gymnasium':
        from gymnasium import wrappers
        from wrapper_gymnasium import DMEnv
    phantom = Phantom()
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        **task_kwargs,
    )
    env = composer.Environment(
        task=task,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )
    env = DMEnv(
        env=env,
        render_kwargs=render_kwargs,
        env_kwargs=env_kwargs,
    )
    if gym_version == 'gymnasium':
        env = wrappers.EnvCompatibility(env, render_mode='rgb_array')
    if flatten_obs:
        env = wrappers.FlattenObservation(env)
    if time_limit is not None:
        env = wrappers.TimeLimit(env, max_episode_steps=time_limit)
    if normalize_obs:
        env = wrappers.NormalizeObservation(env)
    if frame_stack > 1:
        env = wrappers.FrameStack(env, frame_stack)
    if wrap_monitor:
        from stable_baselines3.common.monitor import Monitor
        env = Monitor(env)
    return env


class Application(Application):

    def __init__(self, title, width, height, trial_path=None):
        super().__init__(title, width, height)
        from dm_control.viewer import user_input

        self._input_map.bind(self._move_forward, user_input.KEY_UP)
        self._input_map.bind(self._move_back, user_input.KEY_DOWN)
        self._input_map.bind(self._move_left, user_input.KEY_LEFT)
        self._input_map.bind(self._move_right, user_input.KEY_RIGHT)
        self.null_action = np.zeros(2)
        self._trial_path = trial_path
        if trial_path is not None:
            self._step = 0
            self._episode = 0
            self._policy = None
            self._trajectory = {}
            self._episode_path = self._trial_path / 'episode_0'
            self._images_path = self._episode_path / 'images'
            self._images_path.mkdir(parents=True, exist_ok=True)

    def _save_transition(self, observation, action):
        for key, value in observation.items():
            if key != 'top_camera':
                self._trajectory.setdefault(key, []).append(value)
            else:
                image_path = self._images_path / f'{self._step:03}.png'
                plt.imsave(image_path.as_posix(), value)
        self._trajectory.setdefault('action', []).append(action)

    def _initialize_episode(self):
        self._restart_runtime()
        if self._trial_path is not None:
            trajectory_path = self._episode_path / 'trajectory'
            np.savez_compressed(trajectory_path.as_posix(), **self._trajectory)
            print(f'Episode {self._episode:02} finished')
            self._trajectory = {}
            self._step = 0
            self._episode += 1
            # change the episode path to the new episode
            self._episode_path = self._trial_path / f'episode_{self._episode}'
            self._images_path = self._episode_path / 'images'
            self._images_path.mkdir(parents=True, exist_ok=True)

    def perform_action(self):
        time_step = self._runtime._time_step
        if not time_step.last():
            self._advance_simulation()
            action = self._runtime._last_action
            if self._trial_path is not None:
                print(f'step {self._step:03}')
                self._save_transition(time_step.observation, action)
                self._step += 1
        else:
            self._initialize_episode()

    def _move_forward(self):
        self._runtime._default_action = [1, 0]
        self.perform_action()

    def _move_back(self):
        self._runtime._default_action = [-1, 0]
        self.perform_action()

    def _move_left(self):
        self._runtime._default_action = [0, -1]
        self.perform_action()

    def _move_right(self):
        self._runtime._default_action = [0, 1]
        self.perform_action()


def launch(environment_loader, policy=None, title='Explorer', width=1024,
           height=768, trial_path=None):
    """Launches an environment viewer.

    Args:
      environment_loader: An environment loader (a callable that returns an
        instance of dm_control.rl.control.Environment), an instance of
        dm_control.rl.control.Environment.
      policy: An optional callable corresponding to a policy to execute within the
        environment. It should accept a `TimeStep` and return a numpy array of
        actions conforming to the output of `environment.action_spec()`.
      title: Application title to be displayed in the title bar.
      width: Window width, in pixels.
      height: Window height, in pixels.
    Raises:
        ValueError: When 'environment_loader' argument is set to None.
    """
    app = Application(title=title, width=width,
                      height=height, trial_path=trial_path)
    app.launch(environment_loader=environment_loader, policy=policy)


def record_expert_trajectories(trial_path: Path = None):
    from cathsim import Phantom, Tip, Guidewire, Navigate
    from dm_control import composer

    trial_path.mkdir(parents=True, exist_ok=True)

    phantom = Phantom()
    tip = Tip(n_bodies=4)
    guidewire = Guidewire(n_bodies=80)
    task = Navigate(
        phantom=phantom,
        guidewire=guidewire,
        tip=tip,
        use_image=True,
    )
    env = composer.Environment(
        task=task,
        random_state=np.random.RandomState(42),
        strip_singleton_obs_buffer_dim=True,
    )

    action_spec_name = '\t' + env.action_spec().name.replace('\t', '\n\t')
    print('\nAction Spec:\n', action_spec_name)
    time_step = env.reset()
    print('\nObservation Spec:')
    for key, value in time_step.observation.items():
        print('\t', key, value.shape)

    launch(env, trial_path=trial_path)
