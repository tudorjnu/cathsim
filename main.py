import os

import numpy as np
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.monitor import Monitor
from simulation import CatheterEnv
from utils import TensorboardCallback, evaluate

TIMESTEPS = 150_000
EP_LENGTH = 1000
N_EVAL = 30

ENV_NAME = "distance_1"
OBS_TYPE = "internal"
TARGET = ["bca", "lca"]
SCENE = ["scene_1", "scene_2"]
policies = ["MlpPolicy", "CnnPolicy"]
algorithms = {"PPO": PPO, "A2C": A2C}

SAVING_PATH = f"./benchmarking/{ENV_NAME}"
MODELS_PATH = os.path.join(SAVING_PATH, "models", OBS_TYPE)
LOGS_PATH = os.path.join(SAVING_PATH, "logs", OBS_TYPE)
RESULTS_PATH = os.path.join(SAVING_PATH, "results", OBS_TYPE)
HEATMAPS_PATH = os.path.join(SAVING_PATH, "heatmaps", OBS_TYPE)
for path in [MODELS_PATH, LOGS_PATH, RESULTS_PATH, HEATMAPS_PATH]:
    os.makedirs(path, exist_ok=True)


# , "A2C": A2C, "DDPG": DDPG,  "SAC": SAC, "TD3": TD3, "ARS": ARS, "TQC": TQC, "TRPO": TRPO}
# algorithms = {"PPO": PPO, "A2C": A2C}  # "SAC": SAC, "TD3": TD3, "TQC": TQC,
# algorithms = {"SAC": SAC, "TD3": TD3, "TQC": TQC}


# env = Monitor(env)
# env = make_vec_env(env, n_envs=4)


def train_algorithms(algorithms={}, policies=[], timesteps=int):

    for algorithm_name, algorithm in algorithms.items():
        for policy in policies:
            for scene in SCENE:
                for target in TARGET:
                    env = CatheterEnv(scene=scene, obs_type=OBS_TYPE,
                                      target=target, ep_length=EP_LENGTH)
                    env = Monitor(env)

                    fname = f"{algorithm_name}_{scene}_{target}_{policy}"
                    model_path = os.path.join(MODELS_PATH, fname)

                    tb_cb = TensorboardCallback(ep_length=EP_LENGTH,
                                                heat_path=HEATMAPS_PATH,
                                                fname=fname)

                    if os.path.exists(f"{model_path}.zip"):
                        print("...loading_env...")
                        model = algorithm.load(model_path, env=env)
                    else:
                        model = algorithm(policy, env, verbose=1,
                                          tensorboard_log=LOGS_PATH)
                    model.learn(total_timesteps=timesteps,
                                reset_num_timesteps=False,
                                tb_log_name=fname,
                                callback=tb_cb)
                    model.save(model_path)


def test_algorithms(algorithms={}, policies=[], n_eval=int, render=False):

    for algorithm_name, algorithm in algorithms.items():
        for policy in policies:
            for scene in SCENE:
                for target in TARGET:
                    env = CatheterEnv(scene=scene, obs_type=OBS_TYPE,
                                      target=target, ep_length=EP_LENGTH)

                    fname = f"{algorithm_name}_{scene}_{target}_{policy}"
                    model_path = os.path.join(MODELS_PATH, fname)
                    model_path = os.path.join(MODELS_PATH, fname)
                    print(model_path)
                    print(os.path.exists(f"{model_path}.zip"))
                    if os.path.exists(f"{model_path}.zip"):
                        print(f"...loading {algorithm_name}...")
                        model = algorithm.load(model_path, env=env)

                        results = evaluate(model=model, env=env,
                                           n_episodes=n_eval,
                                           render=render, deterministic=False,
                                           saving_path=RESULTS_PATH,
                                           fname=fname)
                        print(
                            f"Mean Reward:    {results[0]:.2f}+-{results[1]:.4f}")
                        print(
                            f"Mean Force:     {results[2]:.4f}+-{results[3]:.4f}")
                        print(
                            f"Mean Max Force: {results[4]:.4f}+-{results[5]:.4f}")
                        print(
                            f"Success:        {results[6]}")


if __name__ == "__main__":

    # train_algorithms(algorithms, policies, timesteps=TIMESTEPS)

    test_algorithms(algorithms, policies=policies, n_eval=N_EVAL, render=False)
