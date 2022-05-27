from stable_baselines3 import HerReplayBuffer, DDPG, DQN, SAC, TD3
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from collections import OrderedDict
import numpy as np
from gym import GoalEnv, spaces
from gym.envs.registration import EnvSpec
from stable_baselines3.common.env_checker import check_env


class BitFlippingEnv(GoalEnv):

    def __init__(self, n_bits: int = 10):
        super().__init__()

        self.n_bits = n_bits
        self.spec = EnvSpec("BitFlipping-v0")

        self.observation_space = spaces.Dict(
            {
                "observation": spaces.MultiBinary(n_bits),
                "achieved_goal": spaces.MultiBinary(n_bits),
                "desired_goal": spaces.MultiBinary(n_bits)
            }
        )

        self.action_space = spaces.Discrete(n_bits)
        self.state = np.random.randint(2, size=n_bits)
        self.desired_goal = np.random.randint(2, size=n_bits)
        self.max_steps = n_bits
        self.current_step = 0

    def _get_obs(self):
        """
        Helper to create the observation.

        :return: The current observation.
        """
        return OrderedDict([("observation", self.state.copy()),
                            ("achieved_goal", self.state.copy()),
                            ("desired_goal", self.desired_goal.copy())])

    def reset_goal(self):
        self.desired_goal = np.random.randint(2, size=self.n_bits)
        return self.desired_goal.copy()

    def reset(self):
        self.current_step = 0
        self.state = np.random.randint(2, size=self.n_bits)
        self.desired_goal = np.random.randint(2, size=self.n_bits)
        return self._get_obs()

    def step(self, action: int):
        self.state[action] = 1 - self.state[action]
        obs = self._get_obs()
        reward = self.compute_reward(
            obs["achieved_goal"], obs["desired_goal"])
        done = reward == 0.0
        self.current_step += 1
        info = {"is_success": done}
        done = done or self.current_step >= self.max_steps
        return obs, reward, done, info

    @ staticmethod
    def compute_reward(achieved_goal: np.ndarray,
                       desired_goal: np.ndarray,
                       info: dict = None) -> float:
        distance = np.linalg.norm(achieved_goal - desired_goal)
        success = distance == 0

        reward = 1.0 if success else -1.0
        reward = 1.0 if success else -distance

        return reward

    def render(self, mode: str = "human"):
        if mode == "rgb_array":
            return self.state.copy()
        print(self.state)

    def close(self) -> None:
        pass


if __name__ == "__main__":
    N_BITS = 30
    env = BitFlippingEnv(n_bits=N_BITS)
    obs = env.reset()
    print("State: ", obs["observation"])
    print("Desired goal: ", obs["desired_goal"])
    print("Achieved goal: ", obs["achieved_goal"])
    obs, reward, done, info = env.step(0)
    print("State: ", obs["observation"])
    print("Desired goal: ", obs["desired_goal"])
    print("Achieved goal: ", obs["achieved_goal"])
    print("Reward: ", reward)
    print("Done: ", done)
    action = env.action_space.sample()
    print("Action: ", action)

    check_env(env, warn=True)
    # exit()

    model = DQN(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            goal_selection_strategy="future",
            max_episode_length=N_BITS,
            online_sampling=True,
            n_sampled_goal=1,
        ),
        verbose=1,
    )

    model.learn(100_000)
