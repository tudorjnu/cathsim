import os
from cathsim import CathSimEnv
from utils import ALGOS
from gym.wrappers import TimeLimit
from stable_baselines3.common.evaluation import evaluate_policy
EP_LENGTH = 2000

ENV_NAME = "1"
OBS_TYPE = "internal"
SAVING_PATH = f"./benchmarking/{ENV_NAME}"
MODELS_PATH = os.path.join(SAVING_PATH, "models", OBS_TYPE)


if __name__ == "__main__":
    algorithm_name = "ppo"
    algorithm = ALGOS[algorithm_name]
    scene = 1
    target = "bca"
    policy = "MlpPolicy"

    fname = f"{algorithm_name}-{scene}-{target}-{policy}"

    env = CathSimEnv(scene=scene,
                     obs_type=OBS_TYPE,
                     target="bca",
                     delta=0.008)
    # env = gym.make('cathsim-gym/CathSim-v0')

    env = TimeLimit(env, max_episode_steps=EP_LENGTH)

    model_path = os.path.join(MODELS_PATH, fname)
    if os.path.exists(f"{model_path}.zip"):
        print(f"...loading {algorithm_name}...")

    model = algorithm.load(model_path, env=env)

    # mean_reward, std_reward = evaluate_policy(
    # model, model.get_env(), n_eval_episodes=10)

    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)
        env.render()
    env.close()
