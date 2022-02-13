from stable_baselines3 import PPO, A2C, DDPG, HER, SAC, TD3
from sb3_contrib import ARS, TQC, TRPO
from stable_baselines3.common.monitor import Monitor
import os
from simulation import CatheterEnv
from utils import TensorboardCallback

SAVING_PATH = "./benchmarking"
TIMESTEPS = 500_000
EP_LENGTH = 20_000

env = CatheterEnv(obs_type="internal")
env = Monitor(env)


# , "A2C": A2C, "DDPG": DDPG, "HER": HER, "SAC": SAC, "TD3": TD3, "ARS": ARS, "TQC": TQC, "TRPO": TRPO}
algorithms = {"PPO": PPO, "SAC": SAC, "TD3": TD3, "TQC": TQC}

policies = ["MlpPolicy"]

# env = make_vec_env(env, n_envs=1)


def train_algorithms(algorithms={}, policies=[], timesteps=int,
                     path=SAVING_PATH, env_name=str):
    for policy in policies:
        for algorithm_name, algorithm in algorithms.items():
            tensorboard_log = f"{SAVING_PATH}/{env_name}/logs"
            model_path = f"{SAVING_PATH}/{env_name}/models/{algorithm_name}"

            if os.path.exists(f"{model_path}.zip"):
                print("...loading_env...")
                model = algorithm.load(model_path, env=env)
            else:
                model = algorithm(policy, env, verbose=1,
                                  tensorboard_log=tensorboard_log)
            model.learn(total_timesteps=timesteps, reset_num_timesteps=False,
                        callback=TensorboardCallback(ep_length=EP_LENGTH))
            model.save(model_path)


train_algorithms(algorithms, policies, timesteps=TIMESTEPS,
                 env_name="mlp_internal_obs")
