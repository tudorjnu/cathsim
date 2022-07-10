import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from moviepy.editor import *
from stable_baselines3.common.evaluation import evaluate_policy
from gym.wrappers import TimeLimit
from utils import ALGOS, save_clip
from cathsim import CathSimEnv
import cv2
import numpy as np

EP_LENGTH = 2000

ENV_NAME = "2"
OBS_TYPE = "image"
SAVING_PATH = f"./benchmarking/{ENV_NAME}"
MODELS_PATH = os.path.join(SAVING_PATH, "models", OBS_TYPE)
VIDEOS_PATH = os.path.join(SAVING_PATH, "videos", OBS_TYPE)


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
    policy = "CnnPolicy"

    fname = f"{algorithm_name}-{scene}-{target}-{policy}"
    n_frames = 1
    if OBS_TYPE == "image_time":
        OBS_TYPE = "image"
        n_frames = 4

    env = CathSimEnv(scene=scene,
                     obs_type=OBS_TYPE,
                     n_frames=n_frames,
                     target=target,
                     delta=0.008)
    print(env.observation_space.shape)

    env = TimeLimit(env, max_episode_steps=EP_LENGTH)

    model_path = os.path.join(MODELS_PATH, fname)
    if os.path.exists(f"{model_path}.zip"):
        print(f"...loading {algorithm_name}...")

    model = algorithm.load(model_path, env=env)

    for episode in range(10):
        obs = env.reset()
        done = False

        clip_name = f'scene-{scene}_target-{target}_{episode}.mp4'
        clip_name = os.path.join(VIDEOS_PATH, clip_name)

        number_frame_name = f'scene-{scene}_target-{target}_{episode}_number.mp4'
        number_frame_name = os.path.join(VIDEOS_PATH, number_frame_name)

        frames = []
        number_frames = []
        for i in range(EP_LENGTH):
            action, _states = model.predict(obs)
            obs, rewards, done, info = env.step(action)

            frame = env.render(mode="rgb_array", camera="top_view",
                               height=128, width=128)
            frames.append(frame)

            number_frame = write_number(rewards)
            number_frames.append(number_frame)
            if done:
                break
                exit()

        save_clip(clip_name, frames)
        save_clip(number_frame_name, number_frames)

    env.close()
