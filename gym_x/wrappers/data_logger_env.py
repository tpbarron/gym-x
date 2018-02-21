import gym
import numpy as np

class DataLoggerEnv(gym.Wrapper):

    def __init__(self,
                 env):
        super(DataLoggerEnv, self).__init__(env)
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminals = []

    def to_numpy(self):
        np_states = np.array(self.states)
        np_actions = np.array(self.actions)
        np_rewards = np.array(self.rewards)
        np_terminals = np.array(self.terminals)
        return np_states, np_actions, np_rewards, np_terminals

    def reset(self):
        obs = self.env.reset()
        self.states.append(obs)
        return obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        self.rewards.append(rew)
        self.actions.append(action)
        self.terminals.append(done)
        if not done:
            self.states.append(next_obs)
        return next_obs, rew, done, info

    def __str__(self):
        return "DataLoggerEnv: %s" % self.env
