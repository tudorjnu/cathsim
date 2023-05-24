# CathSim

![CathSim](./cathsim.png)

## Installation Procedure

1. If using a `conda environment`:

```bash
conda create -n cathsim python=3.9
conda activate cathsim
pip install setuptools==58.2.0
```

2. Install the environment:

```bash
git clone -b git@github.com:tudorjnu/cathsim.git
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the enviromnent run with gym is to make use of the `make_dm_env` function and then wrap the resulting environment into a `DMEnvToGymWrapper` resulting in a `gym.Env`.

```python
from cathsim.cathsim.env_utils import make_dm_env
from cathsim.wrappers.wrappers import DMEnvToGymWrapper

env = make_dm_env(
    dense_reward=True,
    success_reward=10.0,
    delta=0.004,
    use_pixels=False,
    use_segment=False,
    image_size=64,
    phantom='phantom3',
    target='bca',
)

env = DMEnvToGymWrapper(env)

obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    for obs_key in obs:
        print(obs_key, obs[obs_key].shape)
    print(reward)
    print(done)
    for info_key in info:
        print(info_key, info[info_key])
```


## Citation
```
@article{jianu2022cathsim,
  title={CathSim: An Open-source Simulator for Autonomous Cannulation},
  author={Jianu, Tudor and Huang, Baoru and Abdelaziz, Mohamed EMK and Vu, Minh Nhat and Fichera, Sebastiano and Lee, Chun-Yi and Berthet-Rayne, Pierre and Nguyen, Anh and others},
  journal={arXiv preprint arXiv:2208.01455},
  year={2022}
}
```



