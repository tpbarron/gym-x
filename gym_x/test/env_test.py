import numpy as np

import gym
import gym_x


if __name__ == '__main__':
    # 1. enumerate all envs
    # 2. create env and get obs, act space
    # 3. while not done:
    # 4. run episode
    env = InvertedPendulumSwingupBulletEnv()
    done = False
    env.reset()
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        print (done)
