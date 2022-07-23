from gym.envs.registration import register

register(
    id='cathsim_env/CathSim-v0',
    entry_point='cathsim_env.envs:CathSimEnv',
    max_episode_steps=2000,
    kwargs={'scene': 1,
            'target': 'bca',
            'image_size': 128,
            'delta': 0.008,
            'dense_reward': True,
            'success_reward': 10,
            'compute_force': False,
            }
)
