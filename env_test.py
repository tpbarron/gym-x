import os
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, AntBulletEnv
from pybullet_envs.gym_pendulum_envs import InvertedPendulumSwingupBulletEnv
from gym import spaces
import gym
import gym_x

class InvertedPendulumSwingupBulletEnvX(InvertedPendulumSwingupBulletEnv):
    def __init__(self):
        InvertedPendulumSwingupBulletEnv.__init__(self)

    def _step(self, a):
        """
        Duplicate of super class so that can modify costs, etc
        """
        obs, rew, done, info = super()._step(a)
        return obs, rew, done, info

    def _reset(self):
        state = super()._reset()
        return state

class AntBulletEnvX(AntBulletEnv):

    def __init__(self):
        AntBulletEnv.__init__(self)

    def _step(self, a):
        """
        Duplicate of super class so that can modify costs, etc
        """
        obs, rew, done, info = super()._step(a)
        return obs, rew, done, info

    def _reset(self):
        state = super()._reset()
        return state


if __name__ == '__main__':
    env = InvertedPendulumSwingupBulletEnv()
    done = False
    env.reset()
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        print (done)

    # env = AntBulletEnvX() #gym.make('AntBulletX-v0')
    # done = False
    # env.reset()
    # while not done:
    #     obs, rew, done, info = env.step(env.action_space.sample())
    #     print (done)
