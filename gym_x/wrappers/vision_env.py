import cv2
import gym
import numpy as np
from gym import spaces

class VisionEnv(gym.Wrapper):

    def __init__(self,
                 env,
                 downsample=(32, 32),
                 framestack=4):

        super(VisionEnv, self).__init__(env)
        self._downsample = downsample
        self._framestack = framestack
        self._last_obs = []

        self.observation_space = spaces.Box(0.0, 1.0, (self._framestack,)+self._downsample)

    def _get_obs(self, obs):
        # ignore true observation and take render
        # downsample observation
        obs = self.env.render(mode='rgb_array')
        obs = cv2.resize(obs, self._downsample, interpolation=cv2.INTER_AREA)
        obs = obs.mean(2)
        obs = obs.astype(np.float32)
        # y = input("y/n: ")
        # if y == 'y':
        #     print (obs)
        #     from PIL import Image
        #     I = Image.fromarray(obs)
        #     I = I.resize((256, 256))
        #     I.show()
        #     input("ready?")
        # obs *= (1.0 / 255.0)
        obs = np.reshape(obs, (1,)+self._downsample)

        # store last frames
        if len(self._last_obs) == 0:
            # if we are empty, start of episode, make full
            self._last_obs = [np.copy(obs) for _ in range(self._framestack)]
        else:
            self._last_obs.append(obs)
            if len(self._last_obs) > self._framestack:
                self._last_obs.pop(0)
        full_obs = np.stack(self._last_obs, axis=1)
        return full_obs

    def _reset(self):
        self._last_obs = []
        ret = self.env.reset()
        ret = self._get_obs(ret)
        return ret

    def _step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        next_obs = self._get_obs(next_obs)
        return next_obs, rew, done, info

    def __str__(self):
        return "VisionEnv: %s" % self.env
