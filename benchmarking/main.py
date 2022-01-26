import gym
import os

from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env

algorithm_name = "ppo"
env_name = "MountainCar-v0"

# Parallel environments
# env = make_vec_env(env_name, n_envs=4)
env = gym.make(env_name)
if os.path.exists(f"./benchmarking/{algorithm_name}/{env_name}.zip"):
    print("yes")
    model = PPO.load(f"benchmarking/{algorithm_name}/{env_name}", env=env)
else:
    model = PPO("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=1_000_000)
# model.save(f"benchmarking/{algorithm_name}/{env_name}")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    if dones:
        obs = env.reset()
    env.render()
