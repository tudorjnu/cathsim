from stable_baselines3.common.callbacks import CheckpointCallback
from utils import TensorboardCallback,  ALGOS
from cathsim_her import CathSimEnv
import os
import argparse
from stable_baselines3 import HerReplayBuffer


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
                    help="scene number", type=int, choices=[1, 2, 3])

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

    ap.add_argument("--n-env", required=False, default=1,
                    help="Number of Environments", type=int)

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
                     obs_type=obs_type)

    model = algo(
        "MultiInputPolicy",
        env,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            max_episode_length=ep_len,
            online_sampling=True,
        ),
        verbose=1,
        buffer_size=5e+5,
        learning_rate=1e-3,
        gamma=0.95,
        batch_size=256,
        policy_kwargs=dict(net_arch=[256, 256, 256]),
        tensorboard_log=LOGS_PATH,
    )

    model.learn(timesteps, callback=tb_cb)
    model.save("{}/{}".format(MODELS_PATH, fname))

    # Load saved model
    # Because it needs access to `env.compute_reward()`
    # HER must be loaded with the env
    # model = SAC.load("her_sac_highway", env=env)

    obs = env.reset()
