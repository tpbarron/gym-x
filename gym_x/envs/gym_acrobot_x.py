"""classic Acrobot task"""
import gym
from gym import spaces
from gym.envs.classic_control import acrobot
import numpy as np
from numpy import pi

class AcrobotEnvX(acrobot.AcrobotEnv):
    """
    Note: The differences between the existing Gym Acrobot and this implementation are:
        * Continuous torque input is an option
        * Sparse reward given only at terminal state
    """

    continuous = False

    def __init__(self):
        super(AcrobotEnvX, self).__init__()
        if self.continuous:
            # torque from -1, 1
            self.action_space = spaces.Box(-1., 1., shape = (1,))
        else:
            self.action_space = spaces.Discrete(3)

    def _step(self, a):
        """ Exactly the same as original with modification for discrete or continuous actions """
        s = self.state
        if self.continuous:
            torque = a
        else:
            a = np.squeeze(a)
            torque = self.AVAIL_TORQUE[a]

        # Add noise to the force action
        if self.torque_noise_max > 0:
            torque += self.np_random.uniform(-self.torque_noise_max, self.torque_noise_max)

        # Now, augment the state with our force action so it can be passed to
        # _dsdt
        s_augmented = np.append(s, torque)

        ns = acrobot.rk4(self._dsdt, s_augmented, [0, self.dt])
        # only care about final timestep of integration returned by integrator
        ns = ns[-1]
        ns = ns[:4]  # omit action
        # ODEINT IS TOO SLOW!
        # ns_continuous = integrate.odeint(self._dsdt, self.s_continuous, [0, self.dt])
        # self.s_continuous = ns_continuous[-1] # We only care about the state
        # at the ''final timestep'', self.dt

        ns[0] = acrobot.wrap(ns[0], -pi, pi)
        ns[1] = acrobot.wrap(ns[1], -pi, pi)
        ns[2] = acrobot.bound(ns[2], -self.MAX_VEL_1, self.MAX_VEL_1)
        ns[3] = acrobot.bound(ns[3], -self.MAX_VEL_2, self.MAX_VEL_2)
        self.state = ns
        terminal = self._terminal()
        reward = 1. if terminal else 0.
        # reward = -1. if not terminal else 0.
        return (self._get_ob(), reward, terminal, {})


class AcrobotContinuousEnvX(AcrobotEnvX):
    continuous = True

from gym_x.wrappers import VisionEnv

def make_acrobot_vision_env_x():
    env = AcrobotEnvX()
    env = VisionEnv(env, downsample=(84, 84), framestack=1)
    return env

def make_acrobot_vision_continuous_env_x():
    env = AcrobotContinuousEnvX()
    env = VisionEnv(env, downsample=(84, 84), framestack=1)
    return env

AcrobotVisionEnvX = make_acrobot_vision_env_x
AcrobotContinuousVisionEnvX = make_acrobot_vision_continuous_env_x

if __name__ == '__main__':
    # env = AcrobotEnvX()
    # env = AcrobotContinuousEnvX()
    # env = AcrobotVisionEnvX()
    env = AcrobotContinuousVisionEnvX()
    env.reset()
    while True:
        env.render()
        obs, rew, done, info = env.step(env.action_space.sample())
        if done:
            break
