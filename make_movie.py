import os
from cathsim import CathSimEnv
from utils import ALGOS, save_clip
import numpy as np
from gym.wrappers import TimeLimit
from stable_baselines3.common.evaluation import evaluate_policy
import cv2
EP_LENGTH = 2000

ENV_NAME = "1"
OBS_TYPE = "internal"
SAVING_PATH = f"./benchmarking/{ENV_NAME}"
MODELS_PATH = os.path.join(SAVING_PATH, "models", OBS_TYPE)


def write_number(number=0.9999999):

    # Create a black image
    number = str(round(number, 6))
    img = np.zeros((250, 1100, 3), np.uint8)
    img = 255 - img

    # Write some Text

    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (0, 180)
    fontScale = 6
    fontColor = (0, 0, 0)
    thickness = 3

    cv2.putText(img, number,
                position,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType=cv2.LINE_AA)
    return img


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
                     delta=0.008,
                     image_size=1080)

    # env = TimeLimit(env, max_episode_steps=EP_LENGTH)

    model_path = os.path.join(MODELS_PATH, fname)
    if os.path.exists(f"{model_path}.zip"):
        print(f"...loading {algorithm_name}...")

    model = algorithm.load(model_path, env=env)

    # mean_reward, std_reward = evaluate_policy(
    # model, model.get_env(), n_eval_episodes=10)

    # print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    frames = []
    number_frames = []
    obs = env.reset()
    done = False
    for i in range(2000):
        action, _states = model.predict(obs, deterministic=True)
        # action = env.action_space.sample()
        # action[-1] = 0.01
        obs, rewards, done, info = env.step(action)
        image = env.render(mode="rgb_array", width=1080,
                           height=1080, camera_name="top_camera")
        number_frame = write_number(number=rewards)
        frames.append(image)
        number_frames.append(number_frame)
        if done:
            print(len(frames))
            print(len(number_frames))
            break

    save_clip("observation.mp4", frames, fps=15)
    save_clip("reward.mp4", number_frames, fps=15)

    env.close()
