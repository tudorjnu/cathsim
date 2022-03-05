import os
from stable_baselines3 import A2C
import gym
import numpy as np
import matplotlib.pyplot as plt
from simulation import CatheterEnv
from stable_baselines3 import PPO

cwd = os.getcwd()
images_path = os.path.join(cwd, "data", "images")

# for filename in os.listdir(images_path)[::-1]:
    # image_pair = np.load(os.path.join(images_path, filename))
    # image_A = image_pair[:, :, 0]
    # image_B = image_pair[:, :, 1]
    # plt.imshow(image_A)
    # f, axarr = plt.subplots(1, 2)
    # axarr[0].imshow(image_A)  # , cmap="gray")
    # axarr[1].imshow(image_B)  # , cmap="gray")
    # print(image_A.any())
    # # print(np.true_divide(image_B.sum(), (image_B != 0).sum()))
    # # plt.imsave("aorta_image.kpng", image_A)
    # # exit()
    # plt.show()

TIMESTEPS = 300000
EP_LENGTH = 5000

env = CatheterEnv(obs_type="internal")
model_path = "./benchmarking/mlp_internal_6/models/PPO"

model = PPO.load(model_path, env=env)


env = gym.make('CartPole-v1')

model = A2C('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

obs = env.reset()
for i in range(1000):
    env.render()
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
