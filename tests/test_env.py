import cathsim_env
import gym

env = gym.make('cathsim_env/CathSim-v0', obs_type='image')

obs = env.reset()
