import numpy as np
import gym
from gym import spaces
import pdb

class TMazeEnv(gym.Env):
    """
    Define T maze

    XOY
     O
     O
     O
     O
     S

    If start signal is 0, then set goal to be X, if start signal is 1,
    set goal to be Y

    Discrete actions of forward / backward / turn left / turn right

    State is normalized x, y location
        x = [-1, 1], -1 is left goal, 1 is right goal, 0 is center
        y = [0, 1], 0 is start, 1 is goal height
        dir = []

    """

    def __init__(self, length=5):
        self.length = length
        self.x = 0
        self.y = 0
        self.theta = 0

        # 1 means right goal +x, -1 means left goal -1
        self.switch = 1
        self.furthest = 0

        high = np.array([1, 1, 1, 1, 1])
        self.observation_space = spaces.Box(-high, high)
        self.action_space = spaces.Discrete(4)

    def _initialize(self):
        self.x = 0
        self.y = 0
        self.theta = 0
        self.furthest = 0
        self.switch = 1 if np.random.random() < 0.5 else -1

    def _theta_map(self, t):
        if t == 0:
            return 0.
        if t == 1:
            return np.pi / 2.
        if t == 2:
            return np.pi
        if t == 3:
            return 3 * np.pi / 2
        raise Exception

    def _get_state(self):
        xs = self.x
        ys = self.y / (self.length-1)
        # print (self.y, self.length)
        xtheta = np.cos(self._theta_map(self.theta))
        ytheta = np.sin(self._theta_map(self.theta))
        signal = 0
        if self.y == 0:
            signal = self.switch
        return np.array([xs, ys, xtheta, ytheta, signal])

    def _reset(self):
        self._initialize()
        return self._get_state()

    def _step(self, a):
        assert self.action_space.contains(a)
        # if a == 2 or a == 3:
        #     print ("turning")
        if a == 0:
            # move forward, need to check what forward is
            if self.theta == 0:
                self.y += 1
            elif self.theta == 1:
                self.x -= 1
            elif self.theta == 2:
                self.y -= 1
            elif self.theta == 3:
                self.x += 1
        elif a == 1:
            if self.theta == 0:
                self.y -= 1
            elif self.theta == 1:
                self.x += 1
            elif self.theta == 2:
                self.y += 1
            elif self.theta == 3:
                self.x -= 1
        elif a == 2:
            self.theta += 1
        elif a == 3:
            self.theta -= 1

        self.theta = self.theta % 4

        if self.y != self.length - 1:
            self.x = 0
        if self.y < 0:
            self.y = 0
        if self.y >= self.length:
            self.y = self.length - 1

        state = self._get_state()
        rew = 0.
        done = False

        if self.y > self.furthest:
            self.furthest = self.y
            rew += 0.1
        elif self.y != self.length - 1:
            rew -= 0.1
        if self.y == self.length-1 and self.x == self.switch:
            rew += 0.1
            done = True
            print ("Success")
            # input("")
        elif self.y == self.length-1 and self.x == -self.switch:
            rew -= 0.1
            done = True
            # print ("Failure")
            # input("")
        # print (a, state, rew, done)
        # input("")

        return state, rew, done, {}


class TMazeSimpleEnv(gym.Env):
    """
    Define T maze

    XOY
     O
     O
     O
     O
     S

    If start signal is 0, then set goal to be X, if start signal is 1,
    set goal to be Y

    Discrete actions of forward / backward / turn left / turn right

    State is normalized x, y location
        x = [-1, 1], -1 is left goal, 1 is right goal, 0 is center
        y = [0, 1], 0 is start, 1 is goal height

    """

    def __init__(self, length=10):
        self.length = length
        self.x = 0
        self.y = 0

        # 1 means right goal +x, -1 means left goal -1
        self.switch = 1
        self.furthest = 0

        low = np.zeros((3,))
        high = np.ones((3,))
        self.observation_space = spaces.Box(low, high, dtype=np.float32)
        self.action_space = spaces.Discrete(4)

    def _initialize(self):
        self.x = 0
        self.y = 0
        self.furthest = 0
        self.switch = 1 if np.random.random() < 0.5 else -1

    def _get_state(self):
        if self.y == 0:
            state = [0, 1, 0]
            state[self.switch + 1] = 1
        elif self.y != self.length - 1:
            # y > 0
            state = [1, 0, 1]
        else: # y = self.length - 1
            state = [0, 1, 0]
        return np.array(state)

    def _reset(self):
        self._initialize()
        # print ("--- new episode ---")
        return self._get_state()

    def _step(self, a):
        assert self.action_space.contains(a)
        # if a == 2 or a == 3:
        #     print ("turning")
        prev_y = self.y
        if a == 0:
            self.y += 1
        elif a == 1:
            self.y -= 1
        elif a == 2:
            self.x += 1
        elif a == 3:
            self.x -= 1

        state = self._get_state()
        rew = 0.
        done = False

        # check bounds
        if self.y != self.length - 1 and abs(self.x) > 0:
            # x not in center and not at end point
            self.x = 0
            rew -= 0.1
        elif self.y < 0:
            # under flow
            self.y = 0
            rew -= 0.1
        elif self.y >= self.length:
            # overflow
            self.y = self.length - 1
            rew -= 0.1

        # control for forward / backward movement
        if self.y < self.furthest:
            rew -= 0.1
        elif self.y == self.furthest:
            pass # rew += 0
        else: # y > furthest
            # past the furthest point you've gotten + 0.1
            self.furthest = self.y
            rew += 0.1
            #pdb.set_trace()

        # check for terminal
        if self.y == self.length-1 and self.x == self.switch:
            rew += 0.5
            done = True
            # print ("Success")
            # input("")
        elif self.y == self.length-1 and self.x == -self.switch:
            rew -= 0.1
            done = True

        # print (a,self.x, self.y, rew)
        #pdb.set_trace()
        # print (a, state, rew, done)
        # input("")
        return state, rew, done, {}


class StochasticTMazeSimpleEnv(TMazeSimpleEnv):
    """
    Same structure as TMaze but location of signal is stochastic

    Define T maze

    XOY
     O
     O
     O
     O
     S

    If start signal is 0, then set goal to be X, if start signal is 1,
    set goal to be Y

    Discrete actions of forward / backward / left / right
    """

    def __init__(self, length=10):
        super(StochasticTMazeSimpleEnv, self).__init__(length=length)
        # 1 means right goal +x, -1 means left goal -1
        self.switch_position = 0

    def _initialize(self):
        super()._initialize()
        self.switch_position = np.random.choice(np.arange(self.length))

    def _get_state(self):
        if self.y == self.switch_position:
            state = [0, 1, 0]
            state[self.switch + 1] = 1
        elif self.y != self.length - 1:
            # y > 0
            state = [1, 0, 1]
        else: # y = self.length - 1
            state = [0, 1, 0]
        return np.array(state)
