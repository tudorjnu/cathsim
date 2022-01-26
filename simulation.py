import mujoco_env
import gym
from gym import utils
import os
import numpy as np
import mujoco_py

xml_path = os.path.join(os.getcwd(), "assets/scene.xml")


class CatheterEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    collisions = []  # [frame, collision_pos, collision_force]

    def __init__(self):
        """ Inherits from MujocoEnv """
        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        # ctrl_force_coeff = 0.0001
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        # calculate the reward
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward_distance = self.get_target_distance()
        # reward_force = - ctrl_force_coeff * self.get_collision_force()
        reward = reward_ctrl - reward_distance - 1
        done = bool(reward_distance <= 0.0001)
        return ob, reward, done, dict(reward_distance=reward_distance)

    def _get_obs(self):
        # qpos = self.sim.data.qpos
        # qvel = self.sim.data.qvel
        top_view = self.render(mode="rgb_array", height=256, width=256,
                               camera_id=0)
        # side_view = self.render(mode="rgb_array", height=256, width=256,
                                # camera_id=2)
        # return np.concatenate([top_view, side_view], axis=-1)
        return top_view

    def reset_model(self):
        self.set_state(
            self.init_qpos
            + self.np_random.uniform(low=-0., high=0., size=self.model.nq),
            self.init_qvel
            + self.np_random.uniform(low=-0., high=0., size=self.model.nv),
        )

        return self._get_obs()

    def get_collision_force(self):
        frame = 1
        data = self.sim.data
        print('number of contacts', data.ncon)
        collision_pairs = []  # [position, force]
        # for all available contacts
        for i in range(data.ncon):
            contact = data.contact[i]
            # contact_distance = contact.dist
            geom_1 = [contact.geom1,
                      self.sim.model.geom_id2name(contact.geom1)]
            geom_2 = [contact.geom2,
                      self.sim.model.geom_id2name(contact.geom2)]
            if geom_1[1] is None or geom_2[1] is None:
                collision_pos = contact.pos
                c_array = np.zeros(6, dtype=np.float64)
                mujoco_py.functions.mj_contactForce(
                    self.sim.model, data, i, c_array)
                collision_force = np.linalg.norm(c_array[:3])
                collision_pairs.append([collision_pos, collision_force])
        if len(collision_pairs) == 0:
            collision_pairs.append([None, 0])
        collision_pairs = np.array(collision_pairs)
        self.collisions.append([frame, collision_pairs])
        frame += 1
        return np.mean(collision_pairs[:, 1], axis=0)

    def get_target_distance(self):
        "Calculates the distance between the head and the target"
        data = self.sim.data
        head_pos = data.get_body_xpos("B99")
        target_lcca = np.array([0.044562, 0.10332, 1.041])
        distance = np.linalg.norm(head_pos - target_lcca)
        return distance


env = CatheterEnv()
# env = gym.make("MountainCarContinuous-v0")
# Box(4,) means that it is a Vector with 4 components
print("Observation space:", env.observation_space)
print("Shape:", env.observation_space.shape)
# Discrete(2) means that there is two discrete actions
print("Action space:", env.action_space)

# The reset method is called at the beginning of an episode
obs = env.reset()
# Sample a random action
action = env.action_space.sample()
print("Sampled action:", action)
obs, reward, done, info = env.step(action)
# Note the obs is a numpy array
# info is an empty dict for now but can contain any debugging info
# reward is a scalar
print(obs.shape, reward, done, info)


# observation = env.reset()
# start = 1
# print("Degrees of Freedom:", env.sim.model.nv)
# for frame in range(start + 1000):
# env.render()
# # env._get_viewer('rgb_array').read_pixels(
# # 256, 256, depth=False)
# # original image is upside-down, so flip it
# # return data[::-1, :, :]

# # top_camera = env.render(
# # "rgb_array", width=256, height=256, camera_name='top_view')
# # env.get_collision_force(frame)
# action = env.action_space.sample()  # your agent here (this takes random actions)
# # print(action.shape)
# action[0] = .01
# # action[1] = .9
# observation, reward, done, info = env.step(action)
# print(reward)

# env.close()
