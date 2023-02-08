# CathSim

![CathSim](./cathsim.png)

## Installation Procedure

1. If using a `conda environment`:

```bash
conda create -n cathsim python=3.9
conda activate cathsim
```

2. Install the environment:

```bash
git clone -b git@github.com:tudorjnu/cathsim.git
cd cathsim
pip install -e .
```

## Quickstart

A quick way to have the enviromnent run with gym is to make use of the `make_env` function. The function automatically configures the environment and wraps it in a gym compatible environment. 

```python
from cathim.utils import make_env

env = make_env(
    flatten_obs=True,
    time_limit=300,
    normalize_obs=False,
    frame_stack=1,
)

obs = env.reset()
for _ in range(1):
    action = env.action_space.sample()
    obs, rewards, dones, info = env.step(action)
    print(obs, rewards, dones, info)
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



