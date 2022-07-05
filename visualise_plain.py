import numpy as np
import cv2
from cathsim import CathSimEnv
from gym.wrappers import TimeLimit

if __name__ == "__main__":

    env = CathSimEnv(image_size=128,
                     scene=1,
                     obs_type="image",
                     n_frames=4,
                     target="bca",
                     delta=0.008)

    env = TimeLimit(env, max_episode_steps=2000)

    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()
        action = np.zeros_like(action)
        obs, reward, done, info = env.step(action)
        cv2.imshow("image", obs[-1])
        cv2.imwrite("aorta_1_obs.png", obs[-1])
        cv2.waitKey(1)
        exit()
        if done:
            break

    env.close()
