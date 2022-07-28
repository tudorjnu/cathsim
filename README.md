# CathSim

## Installation Procedure

1. Download [MuJoCo](https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz)

```bash
mkdir .mujoco
cd .mujoco
wget https://github.com/deepmind/mujoco/releases/download/2.1.0/mujoco210-linux-x86_64.tar.gz
tar xvzf mujoco210-linux-x86_64.tar.gz
rm -rf mujoco210-linux-x86_64.tar.gz
```

3. Install Dependencies

```bash
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```

3. Add the following to the `.bashrc` file:

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/user/.mujoco/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
```

4. Install the environment

```bash
git clone git@github.com:tudorjnu/cathsim.git
cd cathsim
pip install -e .
```

## Quick start

```python
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
for _ in range(2000):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
```


