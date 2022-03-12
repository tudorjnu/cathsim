import os
from cathsim_0 import CathSimEnv
from utils import TensorboardCallback, evaluate_env
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

EP_LENGTH = 3072
TIMESTEPS = EP_LENGTH * 100
SAVE_FREQ = round(TIMESTEPS/10)
N_EVAL = 30

ENV_NAME = "1_gpu"
OBS_TYPE = "internal"
TARGET = ["bca", "lcca"]
SCENE = [1, 2]
# "DDPG": DDPG, "SAC": SAC, "TD3": TD3, "ARS": ARS, "TQC": TQC, "TRPO": TRPO}
POLICIES = ["MlpPolicy"]
ALGORITHMS = {"PPO":PPO,"A2C":A2C}

SAVING_PATH = f"./benchmarking/{ENV_NAME}"
MODELS_PATH = os.path.join(SAVING_PATH, "models", OBS_TYPE)
CKPT_PATH = os.path.join(SAVING_PATH, "ckpt", OBS_TYPE)
LOGS_PATH = os.path.join(SAVING_PATH, "logs", OBS_TYPE)
RESULTS_PATH = os.path.join(SAVING_PATH, "results", OBS_TYPE)
HEATMAPS_PATH = os.path.join(SAVING_PATH, "heatmaps", OBS_TYPE)

for path in [MODELS_PATH, LOGS_PATH, CKPT_PATH, RESULTS_PATH, HEATMAPS_PATH]:
    os.makedirs(path, exist_ok=True)

# MODEL
policy_kwargs = dict(net_arch=[dict(pi=[128, 128], vf=[128, 128])])


def train_algorithms(algorithms: dict = ALGORITHMS,
                     policies: list = POLICIES,
                     timesteps: int = TIMESTEPS):

    for algorithm_name, algorithm in algorithms.items():
        for policy in policies:
            for scene in SCENE:
                for target in TARGET:

                    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

                    tb_cb = TensorboardCallback(heat_path=HEATMAPS_PATH,
                                                fname=fname)

                    ckpt_cb = CheckpointCallback(save_freq=SAVE_FREQ,
                                                 save_path=CKPT_PATH,
                                                 name_prefix=fname)
                    env = CathSimEnv(scene=scene,
                                     obs_type=OBS_TYPE,
                                     target=target,
                                     ep_length=EP_LENGTH)

                    env = Monitor(env)

                    model_path = os.path.join(MODELS_PATH, fname)

                    if os.path.exists(f"{model_path}.zip"):
                        print("...loading_env...")
                        model = algorithm.load(model_path, env=env)
                    else:
                        model = algorithm(policy, env,
					  policy_kwargs=policy_kwargs,
                                          verbose=1,
                                          tensorboard_log=LOGS_PATH)

                    model.learn(total_timesteps=timesteps,
                                reset_num_timesteps=False,
                                tb_log_name=fname,
                                callback=[tb_cb, ckpt_cb])

                    model.save(model_path)


def test_algorithms(algorithms: dict = ALGORITHMS,
                    policies: list = POLICIES,
                    n_eval: int = 30,
                    render: bool = False,
                    verbose: bool = True):

    for algorithm_name, algorithm in algorithms.items():
        for policy in policies:
            for scene in SCENE:
                for target in TARGET:

                    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

                    env = CathSimEnv(scene=scene,
                                     obs_type=OBS_TYPE,
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
    test_algorithms()
