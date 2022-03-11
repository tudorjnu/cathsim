import os
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import Image
from tqdm import trange


class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self,
                 heat_path: str,
                 fname: str,
                 verbose: int = 0):
        super(TensorboardCallback, self).__init__(verbose)

        self.path = os.path.join(heat_path, f"{fname}.png")

        self.force_images_mean = np.zeros(shape=(256, 256, 1))
        self.force_image_episode = np.zeros(shape=(256, 256, 1))
        self.n = 1

    def _on_step(self) -> bool:
        # info = self.training_env.get_attr("info", 0)[0]
        done = self.training_env.get_attr("done", 0)[0]
        force_image = self.training_env.get_attr("force_image", 0)[0]
        current_step = self.training_env.get_attr("current_step", 0)[0]

        self.force_images_mean = self.force_images_mean + \
            (force_image - self.force_images_mean)/self.n
        self.n += 1

        self.force_image_episode = self.force_image_episode + \
            (force_image - self.force_image_episode)/(current_step+1)

        if done:
            print(self.force_image_episode.min(),
                  self.force_image_episode.max(),
                  np.count_nonzero(self.force_image_episode))

            self.force_image_episode = (self.force_image_episode -
                                        self.force_image_episode.min())/(
                self.force_image_episode.max() -
                self.force_image_episode.min())

            self.logger.record("force_image", Image(
                self.force_image_episode, "HWC"),
                exclude=("stdout", "log", "json", "csv"))

            self.force_image_episode = np.zeros(shape=(256, 256, 1))

            force_images_mean = (self.force_images_mean -
                                 self.force_images_mean.min())/(
                self.force_images_mean.max() -
                self.force_images_mean.min())

            plt.imsave(self.path, np.squeeze(
                force_images_mean, -1), cmap="gray")

            self.logger.record("episode/length",  current_step)

        return True

    def _on_rollout_end(self):
        pass


def evaluate_env(model, env,
                 fname: str,
                 saving_path: str,
                 n_episodes: int = 30,
                 deterministic: bool = False,
                 render: bool = False,
                 verbose: bool = 0
                 ) -> tuple:

    rewards = []
    lengths = []
    forces = []
    max_forces = []
    dones = 0

    for i in range(n_episodes):
        episode_rewards = 0
        episode_forces = []

        done = False
        obs = env.reset()
        for i in trange(env.ep_length):
            if render:
                env.render()
            action, _state = model.predict(obs, deterministic=deterministic)
            obs, reward, done, info = env.step(action)
            episode_rewards += reward
            step_forces = env.force_image[env.force_image.nonzero()]
            if len(step_forces) > 0:
                episode_forces.extend(step_forces)
                forces.extend(step_forces)

            if done:
                if env.current_step <= env.ep_length:
                    dones += 1
                break

        lengths.append(env.current_step)
        rewards.append(episode_rewards)
        forces.extend(episode_forces)
        max_forces.append(np.max(episode_forces))

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

    successes = round(dones/n_episodes, 2)

    if saving_path is not None:
        np.savez(os.path.join(saving_path, fname),
                 forces=forces,
                 rewards=rewards,
                 max_forces=max_forces,
                 successes=successes)
    if verbose:
        print(
            f"Mean Reward:    {mean_reward:.2f}+-{std_reward:.4f}")
        print(
            f"Mean Force:     {mean_force:.4f}+-{std_force:.4f}")
        print(
            f"Mean Max Force: {mean_max_force:.4f}+-{std_max_force:.4f}")
        print(
            f"Success:        {successes}")

    return (mean_reward, std_reward,
            mean_force, std_force,
            mean_max_force, std_max_force,
            successes)
