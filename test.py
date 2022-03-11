from stable_baselines3.common.env_util import make_vec_env
import gym
from stable_baselines3 import PPO

model_path = "./benchmarking/test/PPO"


# Parallel environments
train_env = make_vec_env("CartPole-v1", n_envs=4)
env = gym.make("CartPole-v1")

model = PPO("MlpPolicy", train_env, verbose=1)
model.learn(total_timesteps=25000)
model.save(model_path)
