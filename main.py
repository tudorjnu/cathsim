import os
from cathsim import CathSimEnv
from utils import ALGOS
# from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.monitor import Monitor
from gym.wrappers import TimeLimit

EP_LENGTH = 2000
TIMESTEPS = EP_LENGTH * 300
N_EVAL = 30

ENV_NAME = "test_vectorised"
OBS_TYPE = ["image_time"]
TARGET = ["lcca", "bca"]
SCENE = [1, 2]
algo = "sac"
ALGORITHMS = {f"{algo}": ALGOS[f"{algo}"]}
IMAGE_SIZE = 64
n_env = 2

SAVING_PATH = f"./benchmarking/{ENV_NAME}"


def make_env(rank, scene, target, obs_type, image_size, n_frames, seed):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = CathSimEnv(scene=scene, target=target, obs_type=obs_type,
                         image_size=image_size, n_frames=n_frames)
        env = TimeLimit(env, max_episode_steps=2000)
        env = Monitor(env)
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init


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
                    policy = "MlpPolicy"
                    if obs_type != "internal":
                        policy = "CnnPolicy"

                    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

                    n_frames = 1
                    if obs_type == "image_time":
                        obs_type = "image"
                        n_frames = 4

                    env = SubprocVecEnv([make_env(i, scene=scene,
                                                  target=target, obs_type=obs_type,
                                                  image_size=IMAGE_SIZE, n_frames=n_frames, seed=42)
                                         for i in range(n_env)])

                    model_path = os.path.join(MODELS_PATH, fname)

                    if os.path.exists(f"{model_path}.zip"):
                        print("...loading_env...")
                        model = algorithm.load(model_path, env=env)
                    else:
                        buffer_size = int(1e6)
                        if obs_type == "image":
                            buffer_size = int(3e5)
                        if algorithm_name == "sac":
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH,
                                learning_starts=10000,
                                gradient_steps=-1,
                                buffer_size=buffer_size,
                                seed=42)
                        elif algorithm_name == "ppo":
                            if policy == "MlpPolicy":
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
                                model = algorithm(policy, env,
                                                  verbose=1,
                                                  device="cuda",
                                                  tensorboard_log=LOGS_PATH,
                                                  learning_rate=2.5e-4,
                                                  n_steps=128,
                                                  clip_range=0.1,
                                                  ent_coef=0.01,
                                                  vf_coef=0.5,
                                                  batch_size=256,
                                                  seed=42)

                        elif algorithm_name == "td3":
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH,
                                gradient_steps=1,
                                learning_rate=3e-4,
                                buffer_size=buffer_size,
                                batch_size=256,
                                seed=42)
                        else:
                            model = algorithm(
                                policy, env,
                                verbose=1,
                                device="cuda",
                                tensorboard_log=LOGS_PATH)

                    model.learn(total_timesteps=TIMESTEPS,
                                reset_num_timesteps=True,
                                tb_log_name=fname)

                    model.save(model_path)


if __name__ == "__main__":

    train_algorithms()
