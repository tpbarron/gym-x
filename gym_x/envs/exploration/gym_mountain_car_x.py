import gym
from gym.envs.classic_control import continuous_mountain_car
# import numpy as np

class MountainCarContinuousEnvX(continuous_mountain_car.Continuous_MountainCarEnv):
    """
    Note: The differences between the existing Gym Continuous_MountainCarEnv and this implementation are:
        * Sparse reward given only at terminal state
    """

    def _step(self, a):
        obs, rew, done, info = super(MountainCarContinuousEnvX, self)._step(a)
        if done:
            rew = 1.0
        else:
            rew = 0.0
        return obs, rew, done, info

from gym_x.wrappers import VisionEnv

def make_mountaincar_continuous_vision_env_x():
    env = MountainCarContinuousEnvX()
    env = VisionEnv(env, downsample=(32, 32), framestack=1)
    return env

MountainCarContinuousVisionEnvX = make_mountaincar_continuous_vision_env_x

# if __name__ == '__main__':
#     # env = AcrobotEnvX()
#     # env = AcrobotContinuousEnvX()
#     # env = AcrobotVisionEnvX()
#     env = AcrobotVisionContinuousEnvX()
#     env.reset()
#     while True:
#         env.render()
#         obs, rew, done, info = env.step(env.action_space.sample())
#         if done:
#             break
