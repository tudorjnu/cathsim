from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from utils import TensorboardCallback, ALGOS
from cathsim_0 import CathSimEnv
import os
import argparse


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=False, default="ppo",
                    help="RL Algorithm", type=str, choices=list(ALGOS.keys()))

    ap.add_argument("-n", "--n-timesteps", required=False, default=3072*100,
                    help="total timesteps", type=int)

    ap.add_argument("-e", "--ep-len", required=False, default=3072,
                    help="episode length", type=int)

    ap.add_argument("-E", "--env-name", required=False, default="1",
                    help="Environment Name", type=str)

    ap.add_argument("-s", "--scene", required=False, default=1,
                    help="scene number", type=int, choices=[1, 2])

    ap.add_argument("-t", "--target", required=False, default="bca",
                    help="cannulation target", type=str, choices=["bca", "lcca"])

    ap.add_argument("-o", "--observation", required=False, default="interal",
                    help="Observation Type", type=str, choices=["internal", "image"])

    ap.add_argument("-p", "--policy", required=False, default="MlpPolicy",
                    help="Policy", type=str, choices=["MlpPolicy", "CnnPolicy"])

    ap.add_argument("-S", "--saving-path", required=False, default="./benchmarking/",
                    help="saving path", type=str)

    ap.add_argument("-d", "--device", required=False, default="cpu",
                    type=str, choices=["cpu", "cuda"])

    args = vars(ap.parse_args())

    ep_len = args["ep_len"]
    timesteps = args["n_timesteps"]
    save_freq = round(timesteps/4)
    n_eval = 30

    algo_name = args["algo"]
    algo = ALGOS[algo_name]
    env_name = args["env_name"]
    obs_type = args["observation"]
    target = args["target"]
    scene = args["scene"]
    policy = args["policy"]

    SAVING_PATH = f"./benchmarking/{env_name}"
    MODELS_PATH = os.path.join(SAVING_PATH, "models", obs_type)
    CKPT_PATH = os.path.join(SAVING_PATH, "ckpt", obs_type)
    LOGS_PATH = os.path.join(SAVING_PATH, "logs", obs_type)
    RESULTS_PATH = os.path.join(SAVING_PATH, "results", obs_type)
    HEATMAPS_PATH = os.path.join(SAVING_PATH, "heatmaps", obs_type)

    for path in [MODELS_PATH, LOGS_PATH, CKPT_PATH, RESULTS_PATH, HEATMAPS_PATH]:
        os.makedirs(path, exist_ok=True)

    # MODEL
    # policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])

    fname = f"{algo_name}-{scene}-{target}-{policy}"

    tb_cb = TensorboardCallback(heat_path=HEATMAPS_PATH,
                                fname=fname)

    ckpt_cb = CheckpointCallback(save_freq=save_freq,
                                 save_path=CKPT_PATH,
                                 name_prefix=fname)

    env = CathSimEnv(scene=scene,
                     obs_type=obs_type,
                     target=target,
                     ep_length=ep_len)

    env = Monitor(env)

    model_path = os.path.join(MODELS_PATH, fname)

    if os.path.exists(f"{model_path}.zip"):
        print("...loading_env...")
        model = algo.load(model_path, env=env)
    else:
        model = algo(policy, env,
                     # policy_kwargs=policy_kwargs,
                     verbose=1,
                     device=args["device"],
                     tensorboard_log=LOGS_PATH)

        model.learn(total_timesteps=timesteps,
                    reset_num_timesteps=False,
                    tb_log_name=fname,
                    callback=[tb_cb, ckpt_cb])

        model.save(model_path)
