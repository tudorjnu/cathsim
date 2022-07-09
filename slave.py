import numpy as np
import time
import cv2
import os
from gym import utils
from stable_baselines3.common.env_checker import check_env
from gym.envs.mujoco import mujoco_env
from utils import save_clip

path = os.getcwd()


class CathSimEnv(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self):

        xml_file = 'scene_slave.xml'
        xml_file = os.path.join(path, 'assets', xml_file)
        self.clamps = {"front": {"indexes": [1, 2], "position": "open"},
                       "back": {"indexes": [7, 8], "position": "open"}}
        self.slider = {"front": 0, "middle": 3, "upper": 6, "rear": 9}
        self.catheter_wheel_indexes = [4, 5]
        self.guidewire_wheel_indexes = [10, 11]

        utils.EzPickle.__init__(self)

        """ Inherits from MujocoEnv """

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def step(self, a):
        """ Performs the simulation step
        """

        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()

        return obs, 0, False, {}

    def _get_obs(self):
        """_get_obs."""
        data = self.sim.data

        position = data.qpos.flat.copy()
        velocity = data.qvel.flat.copy()

        obs = np.concatenate(
            (
                position,
                velocity,
            )
        )

        return obs

    def reset(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-0.00,
                                                    high=0.00,
                                                    size=self.model.nv),
        )

        return self._get_obs()

    def move_slider(self, slider="front", distance=0.0001,):
        """Moves the sliders forward by distance"""
        q_pos = self.data.qpos.flat.copy()
        q_vel = self.data.qvel.flat.copy()
        i = self.slider[slider]
        q_pos[i] += distance
        q_vel[i] = 0
        self.set_state(q_pos, q_vel)

    def rotate_catheter(self, angle=0.0001):
        """Rotates the catheter by angle"""
        q_pos = self.data.qpos.flat.copy()
        q_vel = self.data.qvel.flat.copy()
        q_pos[self.catheter_wheel_indexes[0]] -= angle
        q_pos[self.catheter_wheel_indexes[1]] += angle
        q_vel[self.catheter_wheel_indexes[0]] = 0
        q_vel[self.catheter_wheel_indexes[1]] = 0
        self.set_state(q_pos, self.init_qvel)
        self.sim.forward()

    def rotate_guidewire(self, angle=0.0001):
        """Rotates the guidewire by angle"""
        q_pos = self.data.qpos.flat.copy()
        q_vel = self.data.qvel.flat.copy()
        q_pos[self.guidewire_wheel_indexes[0]] -= angle
        q_pos[self.guidewire_wheel_indexes[1]] += angle
        q_vel[self.guidewire_wheel_indexes[0]] = 0
        q_vel[self.guidewire_wheel_indexes[1]] = 0
        self.set_state(q_pos, self.init_qvel)
        self.sim.forward()

    def set_clamp(self, clamp="front", position="open"):
        """Sets the clamp to open or closed"""
        q_pos = self.data.qpos.flat.copy()
        q_vel = self.data.qvel.flat.copy()
        if position == "open":
            if not self.clamps[clamp]["position"] == "open":
                q_pos[self.clamps[clamp]["indexes"][0]
                      ] = self.model.jnt_range[self.clamps[clamp]["indexes"][0]][0]
                q_pos[self.clamps[clamp]["indexes"][1]
                      ] = self.model.jnt_range[self.clamps[clamp]["indexes"][1]][0]
                self.clamps[clamp]["position"] = "open"
        elif position == "closed":
            if not self.clamps[clamp]["position"] == "closed":
                q_pos[self.clamps[clamp]["indexes"][0]
                      ] = self.model.jnt_range[self.clamps[clamp]["indexes"][0]][1]
                q_pos[self.clamps[clamp]["indexes"][1]
                      ] = self.model.jnt_range[self.clamps[clamp]["indexes"][1]][1]
                self.clamps[clamp]["position"] = "closed"

        q_vel[self.clamps[clamp]["indexes"][0]] = 0
        q_vel[self.clamps[clamp]["indexes"][1]] = 0

        self.set_state(q_pos, q_vel)
        self.sim.forward()

    def push_catheter_forward(self, distance=0.00003, i=0):
        """Pushes the catheter forward by distance"""
        frames = []
        if self.clamps["front"]["position"] == "open":
            self.set_clamp("front", "closed")
            image = self.render(mode="rgb_array", width=1920,
                                height=1080, camera_name="profile_camera")
            frames.append(image)

        for slider in self.slider.keys():
            self.move_slider(slider, distance)
            if i % 50 == 0:
                image = self.render(mode="rgb_array", width=1920,
                                    height=1080, camera_name="profile_camera")
                cv2.imshow("image", image)
                cv2.waitKey(1)
                frames.append(image)

        # if the front slider reached the joint limit,
        # the front slider is pushed back to the mean distance between the joint
        # limits
        if self.data.qpos[self.slider["front"]] > self.model.jnt_range[self.slider["front"]][1]:
            mean_distance = (self.model.jnt_range[self.slider["front"]][1] -
                             self.model.jnt_range[self.slider["front"]][0]) / 2
            j = 0
            self.set_clamp("front", "open")
            while self.data.qpos[self.slider["front"]] > mean_distance:
                self.move_slider("front", -distance)
                if j % 50 == 0:
                    image = self.render(mode="rgb_array", width=1920,
                                        height=1080, camera_name="profile_camera")
                    frames.append(image)
                j += 1
                # self.render()
        return frames


if __name__ == "__main__":

    env = CathSimEnv()
    check_env(env)
    print(env.observation_space.shape)
    obs = env.reset()
    done = False
    clamp_pose = "open"
    frames = []
    for i in range(4000):
        # env.render()
        action = [0.000023]
        obs, reward, done, info = env.step(action)
        frames.extend(env.push_catheter_forward(i=i))
        print(i)

    save_clip("catheter_push_test.mp4", frames, fps=30)

    # if i % 400 == 0:
    # clamp_pose = "closed" if clamp_pose == "open" else "open"
    # env.set_clamp("front", clamp_pose)
