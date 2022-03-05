import os
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common import base_class
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.logger import Image
from stable_baselines3.common.vec_env import (DummyVecEnv, VecEnv, VecMonitor,
                                              is_vecenv_wrapped,
                                              sync_envs_normalization)
from tqdm import trange


def evaluate_policy(
    model: "base_class.BaseAlgorithm",
    env: Union[gym.Env, VecEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    callback: Optional[Callable[[
        Dict[str, Any], Dict[str, Any]], None]] = None,
    reward_threshold: Optional[float] = None,
    return_episode_rewards: bool = False,
    warn: bool = True,
) -> Union[Tuple[float, float], Tuple[List[float], List[int]]]:
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.
    .. note::
        If environment has not been wrapped with ``Monitor`` wrapper, reward and
        episode lengths are counted as it appears with ``env.step`` calls. If
        the environment contains wrappers that modify rewards or episode lengths
        (e.g. reward scaling, early episode reset), these will affect the evaluation
        results as well. You can avoid this by wrapping environment with ``Monitor``
        wrapper before anything else.
    :param model: The RL agent you want to evaluate.
    :param env: The gym environment or ``VecEnv`` environment.
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param callback: callback function to do additional checks,
        called after each step. Gets locals() and globals() passed as parameters.
    :param reward_threshold: Minimum expected reward per episode,
        this will raise an error if the performance is not met
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param warn: If True (default), warns user about lack of a Monitor wrapper in the
        evaluation environment.
    :return: Mean reward per episode, std of reward per episode.
        Returns ([float], [int]) when ``return_episode_rewards`` is True, first
        list containing per-episode rewards and second containing per-episode lengths
        (in number of steps).
    """
    is_monitor_wrapped = False
    # Avoid circular import
    from stable_baselines3.common.monitor import Monitor

    if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])

    is_monitor_wrapped = is_vecenv_wrapped(
        env, VecMonitor) or env.env_is_wrapped(Monitor)[0]

    if not is_monitor_wrapped and warn:
        warnings.warn(
            "Evaluation environment is not wrapped with a ``Monitor`` wrapper. "
            "This may result in reporting modified episode lengths and rewards, if other wrappers happen to modify these. "
            "Consider wrapping environment first with ``Monitor`` wrapper.",
            UserWarning,
        )

    n_envs = env.num_envs
    episode_rewards = []
    episode_lengths = []
    episode_forces = []

    episode_counts = np.zeros(n_envs, dtype="int")
    # Divides episodes among different sub environments in the vector as evenly as possible
    episode_count_targets = np.array(
        [(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype="int")

    current_rewards = np.zeros(n_envs)
    current_lengths = np.zeros(n_envs, dtype="int")
    current_forces = np.zeros(n_envs)

    observations = env.reset()
    states = None
    episode_starts = np.ones((env.num_envs,), dtype=bool)
    while (episode_counts < episode_count_targets).any():
        actions, states = model.predict(
            observations, state=states, episode_start=episode_starts, deterministic=deterministic)
        observations, rewards, dones, infos = env.step(actions)
        current_rewards += rewards
        current_lengths += 1
        current_forces += infos[0]['reward_force']
        for i in range(n_envs):
            if episode_counts[i] < episode_count_targets[i]:

                # unpack values so that the callback can access the local variables
                reward = rewards[i]
                done = dones[i]
                info = infos[i]
                episode_starts[i] = done

                if callback is not None:
                    callback(locals(), globals())

                if dones[i]:
                    if is_monitor_wrapped:
                        # Atari wrapper can send a "done" signal when
                        # the agent loses a life, but it does not correspond
                        # to the true end of episode
                        if "episode" in info.keys():
                            # Do not trust "done" with episode endings.
                            # Monitor wrapper includes "episode" key in info if environment
                            # has been wrapped with it. Use those rewards instead.
                            episode_rewards.append(info["episode"]["r"])
                            episode_lengths.append(info["episode"]["l"])
                            episode_forces.append(
                                info["episode"]["reward_force"])
                            # Only increment at the real end of an episode
                            episode_counts[i] += 1
                    else:
                        episode_rewards.append(current_rewards[i])
                        episode_lengths.append(current_lengths[i])
                        episode_forces.append(current_forces[i])
                        episode_counts[i] += 1
                    current_rewards[i] = 0
                    current_lengths[i] = 0
                    current_forces[i] = 0

        if render:
            env.render()

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    mean_force = np.mean(episode_forces)
    std_force = np.std(episode_forces)

    if reward_threshold is not None:
        assert mean_reward > reward_threshold, "Mean reward below threshold: " f"{mean_reward:.2f} < {reward_threshold:.2f}"
    if return_episode_rewards:
        return episode_rewards, episode_lengths, episode_forces
    return mean_reward, std_reward, mean_force, std_force


class EvalCallback(EventCallback):
    """
    Callback for evaluating an agent.
    .. warning::
      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    :param warn: Passed to ``evaluate_policy`` (warns if ``eval_env`` has not been
        wrapped with a Monitor wrapper)
    """

    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        callback_on_new_best: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
    ):
        super(EvalCallback, self).__init__(
            callback_on_new_best, verbose=verbose)
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.warn = warn

        # Convert to VecEnv for consistency
        if not isinstance(eval_env, VecEnv):
            eval_env = DummyVecEnv([lambda: eval_env])

        self.eval_env = eval_env
        self.best_model_save_path = best_model_save_path
        # Logs will be written in ``evaluations.npz``
        if log_path is not None:
            log_path = os.path.join(log_path, "evaluations")
        self.log_path = log_path
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.evaluations_length = []
        # For computing success rate
        self._is_success_buffer = []
        self.evaluations_successes = []

        self.evaluations_force = []

    def _init_callback(self) -> None:
        # Does not work in some corner cases, where the wrapper is not the same
        if not isinstance(self.training_env, type(self.eval_env)):
            warnings.warn(
                "Training and eval env are not of the same type" f"{self.training_env} != {self.eval_env}")

        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)

    def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
        """
        Callback passed to the  ``evaluate_policy`` function
        in order to log the success rate (when applicable),
        for instance when using HER.
        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            maybe_is_success = info.get("is_success")
            if maybe_is_success is not None:
                self._is_success_buffer.append(maybe_is_success)

    def _on_step(self) -> bool:

        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Sync training and eval env if there is VecNormalize
            if self.model.get_vec_normalize_env() is not None:
                try:
                    sync_envs_normalization(self.training_env, self.eval_env)
                except AttributeError:
                    raise AssertionError(
                        "Training and eval env are not wrapped the same way, "
                        "see https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback "
                        "and warning above."
                    )

            # Reset success rate buffer
            self._is_success_buffer = []

            episode_rewards, episode_lengths, episode_forces = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                deterministic=self.deterministic,
                return_episode_rewards=True,
                warn=self.warn,
                callback=self._log_success_callback,
            )

            if self.log_path is not None:
                self.evaluations_timesteps.append(self.num_timesteps)
                self.evaluations_results.append(episode_rewards)
                self.evaluations_length.append(episode_lengths)
                self.evaluations_force.append(episode_forces)

                kwargs = {}
                # Save success log if present
                if len(self._is_success_buffer) > 0:
                    self.evaluations_successes.append(self._is_success_buffer)
                    kwargs = dict(successes=self.evaluations_successes)

                np.savez(
                    self.log_path,
                    timesteps=self.evaluations_timesteps,
                    results=self.evaluations_results,
                    ep_lengths=self.evaluations_length,
                    forces=self.evaluations_force,
                    **kwargs,
                )

            mean_reward, std_reward = np.mean(
                episode_rewards), np.std(episode_rewards)
            mean_force, std_force = np.mean(
                episode_forces), np.std(episode_forces)
            mean_ep_length, std_ep_length = np.mean(
                episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = mean_reward

            if self.verbose > 0:
                print(
                    f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
                print(
                    f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
                print(
                    f"Episode force: {mean_force:.2f} +/- {std_force:.2f}")
            # Add to current Logger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)
            self.logger.record("eval/mean_force", float(mean_force))

            if len(self._is_success_buffer) > 0:
                success_rate = np.mean(self._is_success_buffer)
                if self.verbose > 0:
                    print(f"Success rate: {100 * success_rate:.2f}%")
                self.logger.record("eval/success_rate", success_rate)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps",
                               self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps)

            if mean_reward > self.best_mean_reward:
                if self.verbose > 0:
                    print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.save(os.path.join(
                        self.best_model_save_path, "best_model"))
                self.best_mean_reward = mean_reward
                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()

        return True


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0, ep_length=20000, heat_path=str, fname=str):
        super(TensorboardCallback, self).__init__(verbose)

        self.path = os.path.join(heat_path, f"{fname}.png")

        self.force_image = np.zeros(shape=(256, 256, 1))
        self.force_image_mean = np.zeros(shape=(256, 256, 1))
        self.n = 1
        self.step_counter = 0
        self.ep_length = ep_length

    def _on_step(self) -> bool:
        info = self.training_env.get_attr("info", 0)[0]
        # reward_force = -1 * info['reward_force']
        mean_force_step = info['mean_force']

        done = info['done']
        self.force_image += info['force_image']
        self.logger.record('step/force_mean', mean_force_step)
        self.logger.record('step/max_force', np.max(info['force_image']))
        if done:
            self.force_image = self.force_image / self.step_counter
            self.force_image = (self.force_image - self.force_image.min()) / \
                (self.force_image.max()-self.force_image.min())
            self.logger.record("force_image", Image(
                self.force_image, "HWC"), exclude=("stdout", "log", "json", "csv"))

            self.force_image_mean = self.force_image_mean + \
                (self.force_image - self.force_image_mean)/self.n
            self.n += 1

            plt.imsave(
                self.path, np.squeeze(
                    self.force_image_mean, -1))

            mean_force = np.true_divide(
                self.force_image.sum(), (self.force_image != 0).sum())

            self.logger.record("episode/force_mean", mean_force)
            self.logger.record("episode/length", self.step_counter)

            self.force_image = np.zeros(shape=(256, 256, 1))
            self.step_counter = 0

        else:
            self.step_counter += 1
        return True

    def _on_rollout_end(self):
        pass


def evaluate(model, env, n_episodes=3, deterministic=False, render=False,
             saving_path=None, fname=None):

    rewards = []
    lengths = []
    forces = []
    all_forces = []
    max_forces = []
    dones = 0
    step_cntr = 1

    for i in range(n_episodes):
        episode_rewards = 0
        episode_forces = []
        episode_max_forces = []
        done = False
        obs = env.reset()
        for i in trange(env.ep_length):
            if render:
                env.render()
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            episode_forces.append(info['mean_force'])
            episode_max_forces.append(info['max_force'])
            forces = info['forces']
            if len(forces) > 0:
                all_forces.extend(forces)
            if done:
                if step_cntr < env.ep_length:
                    dones += 1
                step_cntr = 1
                break
            step_cntr += 1
        lengths.append(env.current_step)
        rewards.append(episode_rewards)
        forces.append(np.mean(episode_forces))
        max_forces.append(np.max(episode_max_forces))

    mean_reward = 0
    std_reward = 0

    mean_force = 0
    std_force = 0

    mean_max_force = 0
    std_max_force = 0

    if len(rewards) > 0:
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards)

    if len(forces) > 0:
        mean_force = np.mean(forces)
        std_force = np.std(forces)

    if len(max_forces) > 0:
        mean_max_force = np.mean(max_forces)
        std_max_force = np.std(max_forces)

    all_forces = np.squeeze(np.array(all_forces))

    successes = round(dones/n_episodes, 2)

    if saving_path is not None:
        np.savez(os.path.join(saving_path, fname),
                 forces=forces,
                 rewards=rewards,
                 max_forces=max_forces,
                 all_forces=all_forces,
                 successes=successes)

    return (mean_reward, std_reward, mean_force, std_force,
            mean_max_force, std_max_force, successes,
            all_forces)
