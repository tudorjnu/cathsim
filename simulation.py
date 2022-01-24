import mujoco_env
from gym import utils
import os
import numpy as np
import mujoco_py

xml_path = os.path.join(os.getcwd(), "assets/scene.xml")


class CatheterEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    collisions = []  # [frame, collision_pos, collision_force]

    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, xml_path, 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        ctrl_cost_coeff = 0.0001
        ctrl_force_coeff = 0.0001
        self.do_simulation(a, self.frame_skip)
        reward_ctrl = -ctrl_cost_coeff * np.square(a).sum()
        reward_distance = - self.get_target_distance()
        reward_force = - ctrl_force_coeff * self.get_collision_force()
        reward = reward_ctrl + reward_distance + reward_force
        ob = self._get_obs()
        return ob, reward, False, dict(reward_distance=reward_distance,
                                       reward_force=reward_force,
                                       reward_ctrl=reward_ctrl)

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos.flat, qvel.flat])

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
observation = env.reset()
start = 1
print("Degrees of Freedom:", env.sim.model.nv)
for frame in range(start + 1000):
    env.render()
    # env.get_collision_force(frame)
    action = env.action_space.sample()  # your agent here (this takes random actions)
    # print(action.shape)
    action[0] = .01
    # action[1] = .9
    observation, reward, done, info = env.step(action)
    print(reward)

env.close()
