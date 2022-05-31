import os
from cathsim import CathSimEnv
from utils import evaluate_env
from utils import ALGOS
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, VecTransposeImage
from gym.wrappers import TimeLimit

EP_LENGTH = 2000
TIMESTEPS = EP_LENGTH * 300
N_EVAL = 30

ENV_NAME = "2"
OBS_TYPE = ["image_time", "image"]
TARGET = ["bca", "lcca"]
SCENE = [1, 2]
algo = "sac"
ALGORITHMS = {f"{algo}": ALGOS[f"{algo}"]}


SAVING_PATH = f"./benchmarking/{ENV_NAME}"


def train_algorithms(algorithms: dict = ALGORITHMS):
    for obs_type in OBS_TYPE:
        MODELS_PATH = os.path.join(SAVING_PATH, "models", obs_type)
        CKPT_PATH = os.path.join(SAVING_PATH, "ckpt", obs_type)
        LOGS_PATH = os.path.join(SAVING_PATH, "logs", obs_type)
        RESULTS_PATH = os.path.join(SAVING_PATH, "results", obs_type)
        HEATMAPS_PATH = os.path.join(SAVING_PATH, "heatmaps", obs_type)

        for path in [MODELS_PATH, LOGS_PATH, CKPT_PATH, RESULTS_PATH, HEATMAPS_PATH]:
            os.makedirs(path, exist_ok=True)

        for algorithm_name, algorithm in algorithms.items():
            for scene in SCENE:
                for target in TARGET:
                    if obs_type == "internal":
                        policy = "MlpPolicy"
                    else:
                        policy = "CnnPolicy"

                    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

                    env = CathSimEnv(scene=scene,
                                     obs_type=obs_type,
                                     target=target)

                    env = TimeLimit(env, max_episode_steps=EP_LENGTH)

                    N_ENVS = 4
                    if algorithm_name == "sac" and obs_type != "internal":
                        N_ENVS = 1
                    env = make_vec_env(
                        lambda: env, n_envs=N_ENVS, vec_env_cls=SubprocVecEnv)

                    model_path = os.path.join(MODELS_PATH, fname)

                    if os.path.exists(f"{model_path}.zip"):
                        print("...loading_env...")
                        model = algorithm.load(model_path, env=env)
                    else:
                        if algorithm_name == "sac":
                            buffer_size = int(1e6)
                            if obs_type == "image_time":
                                buffer_size = int(5e5)
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH,
                                learning_starts=10000,
                                buffer_size=buffer_size,
                                seed=42)
                        elif algorithm_name == "ppo":
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH,
                                learning_rate=5.05041e-05,
                                n_steps=512,
                                clip_range=0.1,
                                ent_coef=0.000585045,
                                n_epochs=20,
                                max_grad_norm=1,
                                vf_coef=0.871923,
                                batch_size=32,
                                seed=42)
                        else:
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH)

                    model.learn(total_timesteps=TIMESTEPS,
                                reset_num_timesteps=False,
                                tb_log_name=fname)

                    model.save(model_path)


def test_algorithms(algorithms: dict = ALGORITHMS,
                    n_eval: int = 30,
                    render: bool = False,
                    verbose: bool = True):

    for obs_type in OBS_TYPE:
        MODELS_PATH = os.path.join(SAVING_PATH, "models", obs_type)
        RESULTS_PATH = os.path.join(SAVING_PATH, "results", obs_type)

        for algorithm_name, algorithm in algorithms.items():
            for scene in SCENE:
                for target in TARGET:
                    if obs_type == "internal":
                        policy = "MlpPolicy"
                    else:
                        policy = "CnnPolicy"

                    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

                    env = CathSimEnv(scene=scene,
                                     obs_type=obs_type,
                                     target=target,
                                     ep_length=EP_LENGTH)

                    model_path = os.path.join(MODELS_PATH, fname)
                    if os.path.exists(f"{model_path}.zip"):
                        print(f"...loading {algorithm_name}...")

                        model = algorithm.load(model_path, env=env)

                        evaluate_env(model=model, env=env,
                                     n_episodes=n_eval,
                                     render=render,
                                     deterministic=False,
                                     saving_path=RESULTS_PATH,
                                     fname=fname)


if __name__ == "__main__":

    train_algorithms()
