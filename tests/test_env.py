import cathsim_env
import gym

scene = 1  # 1 or 2 for Type-I Aortic Arch and Type-II Aortic Arch
target = "bca"  # "bca" or "lcca"
obs_type = "internal"  # image or internal
image_size = 128
delta = 0.008  # the distance threshold between catheter head and target
success_reward = 10.0  # the reward for reaching the target
compute_force = False  # whether to compute the force
dense_reward = True  # whether to use a dense reward or a sparse reward

env = gym.make('cathsim_env/CathSim-v0', scene=scene, target=target,
               obs_type=obs_type, image_size=image_size, delta=delta,
               success_reward=success_reward, compute_force=compute_force,
               dense_reward=dense_reward)

obs = env.reset()
for _ in range(2):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    print("obs: ", obs.shape)
    print("reward: ", reward)
    print("done: ", done)
    for k, v in info.items():
        print(k, v)
