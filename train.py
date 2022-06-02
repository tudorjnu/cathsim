from gym.wrappers import TimeLimit
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

from utils import ALGOS
from cathsim import CathSimEnv
import os
import argparse


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=False, default="ppo",
                    help="RL Algorithm", type=str, choices=list(ALGOS.keys()))

    ap.add_argument("-n", "--n-timesteps", required=False, default=6e5,
                    help="total timesteps", type=int)

    ap.add_argument("-e", "--ep-len", required=False, default=2000,
                    help="episode length", type=int)

    ap.add_argument("-E", "--env-name", required=False, default="test",
                    help="Environment Name", type=str)

    ap.add_argument("-s", "--scene", required=False, default=1,
                    help="scene number", type=int, choices=[1, 2, 3])

    ap.add_argument("-t", "--target", required=False, default="bca",
                    help="cannulation target", type=str, choices=["bca", "lcca"])

    ap.add_argument("-o", "--observation", required=False, default="interal",
                    help="Observation Type", type=str, choices=["internal", "image", "image_time"])

    ap.add_argument("-p", "--policy", required=False, default="MlpPolicy",
                    help="Policy", type=str, choices=["MlpPolicy", "CnnPolicy"])

    ap.add_argument("-S", "--saving-path", required=False, default="./benchmarking/",
                    help="saving path", type=str)

    ap.add_argument("-d", "--device", required=False, default="cpu",
                    type=str, choices=["cpu", "cuda"])

    ap.add_argument("--n-env", required=False, default=1,
                    help="Number of Environments", type=int)

    ap.add_argument("--n-frames", required=False, default=1,
                    help="Number of Frames", type=int)
    args = vars(ap.parse_args())

    ep_len = args["ep_len"]
    timesteps = args["n_timesteps"]
    n_eval = 30
    env_name = args["env_name"]
    obs_type = args["observation"]
    n_frames = args["n_frames"]
    target = args["target"]
    scene = args["scene"]
    policy = args["policy"]
    algo_name = args["algo"]
    algo = ALGOS[algo_name]

    SAVING_PATH = f"./benchmarking/{env_name}"
    MODELS_PATH = os.path.join(
        SAVING_PATH, "models", obs_type + f"_{args['n_frames']}")
    LOGS_PATH = os.path.join(
        SAVING_PATH, "logs", obs_type + f"_{args['n_frames']}")

    for path in [MODELS_PATH, LOGS_PATH]:
        os.makedirs(path, exist_ok=True)

    fname = f"{algo_name}-{scene}-{target}-{policy}"

    env = CathSimEnv(scene=scene,
                     obs_type=obs_type,
                     target=target,
                     n_frames=n_frames)

    env = TimeLimit(env, max_episode_steps=ep_len)

    if obs_type == "image":
        env = make_vec_env(
            lambda: env, n_envs=args["n_env"], vec_env_cls=SubprocVecEnv)

    model_path = os.path.join(MODELS_PATH, fname)

    if os.path.exists(f"{model_path}.zip"):
        print("...loading_env...")
        model = algo.load(model_path, env=env)
    else:
        buffer_size = int(1e6)
        if obs_type == "image_time":
            buffer_size = int(3e5)
        if algo_name == "sac":
            model = algo(
                policy, env,
                verbose=1,
                device="cuda",
                tensorboard_log=LOGS_PATH,
                learning_starts=10000,
                buffer_size=buffer_size,
                seed=42)
        elif algo_name == "ppo":
            model = algo(
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
            model = algo(
                policy, env,
                verbose=1,
                device="cuda",
                tensorboard_log=LOGS_PATH)

    model.learn(total_timesteps=timesteps,
                reset_num_timesteps=False,
                tb_log_name=fname)
