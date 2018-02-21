import cv2
import gym
import numpy as np
from gym import spaces

class TimeHorizonEnv(gym.Wrapper):

    def __init__(self,
                 env,
                 horizon=1000):

        super(TimeHorizonEnv, self).__init__(env)
        self.horizon = horizon
        self.t = 0

    def reset(self):
        self.t = 0
        ret = self.env.reset()
        return ret

    def step(self, action):
        self.t += 1
        next_obs, rew, done, info = self.env.step(action)
        if self.t >= self.horizon:
            done = True
        return next_obs, rew, done, info

    def __str__(self):
        return "TimeHorizonEnv: %s" % self.env
