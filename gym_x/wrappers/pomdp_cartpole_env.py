
import gym
import numpy as np
from gym import spaces

class PomdpCartPoleEnv(gym.Wrapper):

    def __init__(self, env):
        super(PomdpCartPoleEnv, self).__init__(env)
        # self.state = (x,x_dot,theta,theta_dot)
        orig_high = env.observation_space.high
        new_high = np.array([orig_high[0], orig_high[2]])
        self.observation_space = spaces.Box(-new_high, new_high)

    def _get_po_state(self, state):
        return np.array([state[0], state[2]])

    def _reset(self):
        ret = self.env.reset()
        return self._get_po_state(ret)

    def _step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        return self._get_po_state(next_obs), rew, done, info

    def __str__(self):
        return "PomdpCartPole: %s" % self.env
