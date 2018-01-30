"""
Adapted from: https://raw.githubusercontent.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/bullet/racecarGymEnv.py
"""

import os,  inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0,parentdir)

import math
import time
import random

import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

import pybullet
import pybullet_data
import pybullet_envs.bullet.bullet_client as bullet_client
# import pybullet_envs.bullet.racecar as racecar
from gym_pomdp.envs import racecar as racecar

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class TMazeRacecarGymEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 urdfRoot=pybullet_data.getDataPath(),
                 actionRepeat=50,
                 isEnableSelfCollision=True,
                 isDiscrete=False,
                 renders=False,
                 length=1,
                 deterministic=True,
                 r_type='neg_dist',
                 lod=0):
        self._timeStep = 0.01
        self._urdfRoot = urdfRoot
        self._lod = lod
        if lod > 0:
            actionRepeat = 10
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._ballUniqueId = -1
        self._envStepCounter = 0
        self._renders = renders
        self._isDiscrete = isDiscrete
        if self._renders:
            self._p = bullet_client.BulletClient(
            connection_mode=pybullet.GUI)
        else:
            self._p = bullet_client.BulletClient()

        self._seed()
        #self.reset()
        observationDim = 5
        observation_high = np.ones(observationDim) * 1000 #np.inf

        if (isDiscrete):
            self.action_space = spaces.Discrete(3)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None
        self.length = length
        self.deterministic = deterministic
        self.switch = None
        self.max_x, self.min_y, self.max_y = None, None, None
        self.r_type = r_type

    def _build_tmaze(self):
        self.wall_block_ids = []

        self._p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[-1, 0, 0.5])
        self.wall_block_ids.append(cube_id)

        for i in range(-1, self.length):
            if i == 0 and self.switch == -1:
                cube_id = self._p.loadURDF("cube_red.urdf", basePosition=[i, -1, 0.5])
                self.wall_block_ids.append(cube_id)
            else:
                cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[i, -1, 0.5])
                self.wall_block_ids.append(cube_id)
            if i == 0 and self.switch == 1:
                cube_id = self._p.loadURDF("cube_red.urdf", basePosition=[i, 1, 0.5])
                self.wall_block_ids.append(cube_id)
            else:
                cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5])
                self.wall_block_ids.append(cube_id)

        cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length, -2, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length, 2, 0.5])
        self.wall_block_ids.append(cube_id)

        cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length-1, -2, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length-1, 2, 0.5])
        self.wall_block_ids.append(cube_id)

        for i in range(5):
            cube_id = self._p.loadURDF("cube_black.urdf", basePosition=[self.length+1, i-2, 0.5])
            self.wall_block_ids.append(cube_id)

    def _reset(self):
        self._p.resetSimulation()
        #p.setPhysicsEngineParameter(numSolverIterations=300)
        self._p.setTimeStep(self._timeStep)
        #self._p.loadURDF(os.path.join(self._urdfRoot,"plane.urdf"))
        stadiumobjects = self._p.loadSDF(os.path.join(self._urdfRoot,"stadium.sdf"))
        #move the stadium objects slightly above 0
        for i in stadiumobjects:
            pos,orn = self._p.getBasePositionAndOrientation(i)
            newpos = [pos[0],pos[1],pos[2]-0.1]
            self._p.resetBasePositionAndOrientation(i,newpos,orn)

        self._p.setGravity(0,0,-10)
        self._racecar = racecar.Racecar(self._p,urdfRootPath=self._urdfRoot, timeStep=self._timeStep, lod=self._lod)
        self._envStepCounter = 0
        for i in range(100):
            self._p.stepSimulation()

        # if self.deterministic:
            # self.switch = 1 #if np.random.random() < 0.5 else -1
        # else:
        self.switch = -1 if np.random.random() < 0.5 else 1
        print ("Goal: y = ", self.switch)

        # Build after switch so can set colored block
        self._build_tmaze()

        self.goal = np.array([self.length, self.switch])
        self._observation = self.getExtendedObservation()
        self.max_x, self.min_y, self.max_y = 0, 0, 0
        return np.array(self._observation)

    def __del__(self):
        self._p = 0

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        """
        carx, cary, [carorn], signal
        """
        carpos,carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        # print (carpos)
        # print (carorn)
        signal = [0, 0]
        if self._envStepCounter == 0:
            # if on first step, also give goal signal
            signal = [1, 0] if self.switch == -1 else [0, 1]

        self._observation = []
        # if make this relative normed x and relative normed y then should be able to generalize
        # also add zorientation
        self._observation.extend([carpos[0]/(self.length+0.5), (carpos[1]+1)/2., (carorn[2]+1)/2.])
        # self._observation.extend(carorn[0:3])
        self._observation.extend(signal)
        return self._observation

    def _step(self, action):
        carPos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)

        # if (self._renders):
        #     basePos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        #     #self._p.resetDebugVisualizerCamera(1, 30, -40, basePos)

        if (self._isDiscrete):
	        fwd = [1,1,1]
	        steerings = [-0.6,0,0.6]
	        # fwd = [-1,-1,-1,0,0,0,1,1,1]
	        # steerings = [-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6]
	        forward = fwd[action]
	        steer = steerings[action]
	        realaction = [forward,steer]
        else:
            realaction = action

        self._racecar.applyAction(realaction)
        for i in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            self._observation = self.getExtendedObservation()

            if self._termination():
                break
        self._envStepCounter += 1

        # compute reward based on movement before updated max and min x,y
        reward = self._reward()

        # check movement bounds
        x, y, z = carPos
        if x > self.max_x:
            self.max_x = x
        if x > self.length - 0.5 and y > self.max_y:
            self.max_y = y
        elif x > self.length - 0.5 and y < self.min_y:
            self.min_y = y

        done = self._termination()

        return np.array(self._observation), reward, done, {}

    def _render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos,orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _in_goal_box(self):
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        x, y, z = carpos
        own, other = False, False
        if x > self.length - 0.5 and x < self.length + 0.5:
            if y > self.switch - 0.5 and y < self.switch + 0.5:
                own = True
            elif y > (-1*self.switch) - 0.5 and y < (-1*self.switch) + 0.5:
                other = True
        return own, other

    def _is_wall_contact(self):
        contacts = self._p.getContactPoints(self._racecar.racecarUniqueId)
        # print (contacts)
        for c in contacts:
            # c[2] is bodyB id and c[-2] is distance
            if c[2] in self.wall_block_ids and c[-2] < 1e-8:
                return True
        return False

    def _closest_wall_dist(self):
        closest_dist = np.inf
        for wid in self.wall_block_ids:
            closest = self._p.getClosestPoints(self._racecar.racecarUniqueId, wid, 10.0)
            for c in closest:
                # print (c[-2], closest_dist)
                # input("")
                if c[-2] < closest_dist:
                    closest_dist = c[-2]
        if closest_dist < 0:
            closest_dist = 1e-8
        return closest_dist

    def _termination(self):
        # check if in goal box
        own, other = self._in_goal_box()
        if own:
            print ("Car reached goal")
            return True
        elif other:
            print ("Car wrong goal")
            return True
        if self._is_wall_contact():
            print ("Wall contact")
            return True
        return False

    # def _reward(self):
    #     """
    #     negative dist to goal
    #     """
    #     own, other = self._in_goal_box()
    #     if own:
    #         reward = 1.0
    #     else:
    #         # if not in goal box can have negative for being in wrong positoin
    #         # or reward for moevement
    #         if other:
    #             reward = -1.0
    #         else:
    #             reward = 0.0
    #             carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    #             x, y, z = carpos
    #             if x > self.max_x: # along path
    #                 reward += 0.1
    #             if self.switch == -1 and y < self.min_y and x > self.length + 0.5:
    #                 reward += 0.1
    #             elif self.switch == 1 and y > self.max_y and x > self.length + 0.5:
    #                 reward += 0.1
    #
    #     return reward

    def _reward(self):
        # print ("using r_type: ", self.r_type)
        if self.r_type == 'neg_dist':
            return self._reward_neg_dist_wall_potential()
        elif self.r_type == 'neg_dist_shaped':
            return self._reward_neg_dist_shaped()
        elif self.r_type == 'rew_pos_movement':
            return self._reward_pos_movement_wall_cost()
        raise ValueError
        # if self.r_type == 'neg_dist':
        #     return self._reward_neg_dist()
        # elif self.r_type == 'neg_dist_shaped':
        #     return self._reward_neg_dist_shaped()
        # raise ValueError

    def _reward_pos_movement_wall_cost(self):
        """
        negative dist to goal
        """
        own, other = self._in_goal_box()

        if own:
            reward = 1.0
        elif other:
            # if not in goal box can have negative for being in wrong positoin
            # or reward for moevement
            reward = -1.0
        else:
            # if not at either goal, check movement
            reward = 0.0
            if self._is_wall_contact():
                reward -= 1.0
            carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            x, y, z = carpos
            if x > self.max_x and x < self.length: # along path
                # if moving foward
                reward += 0.1
            if y < self.min_y and x > self.length - 0.5:
                if self.switch == -1:
                    # if moving to neg y after reaching proper x
                    reward += 0.1
                elif self.switch == 1:
                    # moving wrong dir
                    reward -= 0.1
            elif y > self.max_y and x > self.length - 0.5:
                if self.switch == 1:
                    # if moving to pos y after reaching proper x
                    reward += 0.1
                elif self.switch == -1:
                    reward -= 0.1

        return reward

    # def _reward_neg_dist(self):
    #     """
    #     negative dist to goal
    #     """
    #     if self._in_goal_box()[0]:
    #         reward = 1.0
    #     else:
    #         carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    #         carxy = np.array(carpos[0:2])
    #         dist = np.linalg.norm(carxy - self.goal)
    #         reward = -dist
    #     return reward

    def _reward_neg_dist_wall_potential(self):
        """
        negative dist to goal
        """
        reward = 0.0
        carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
        x, y, z = carpos

        if self._in_goal_box()[0]:
            reward += self.length # make reward proportional to length
        else:
            carxy = np.array(carpos[0:2])
            dist = np.linalg.norm(carxy - self.goal)
            reward = -dist

            if self._is_wall_contact():
                reward -= 100.0

        return reward

    def _reward_neg_dist_shaped(self):
        """
        negative dist to goal
        """
        reward = 0.0
        own, other = self._in_goal_box()
        if own:
            reward += 1.0
        elif other:
            # if not in goal box can have negative for being in wrong positoin
            # or reward for moevement
            reward += -1.0

        if not own:
            carpos, carorn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
            carxy = np.array(carpos[0:2])
            if carxy[0] < self.length - 0.5:
                tmp_goal = np.array([self.length, 0])
                dist = np.linalg.norm(carxy - tmp_goal) + np.sqrt(2.)
            else:
                # if at last block
                dist = np.linalg.norm(carxy - self.goal)

            if self._is_wall_contact():
                reward -= 100.0
            reward += -dist

        return reward


from itertools import count
if __name__ == '__main__':
    env = TMazeRacecarGymEnv(isDiscrete=True, renders=True, length=1)

    while True:
        env.reset()
        env.render()
        done = False

        # input("")
        while not done:
            env.render()
            obs, rew, done, info = env.step(0) #env.action_space.sample())
            print ("Reward: ", rew)
            import time
            time.sleep(0.1)

            # for i in count(1):
            # # while not done:
            #     env.render()
            #     obs, rew, done, info = env.step(6) #env.action_space.sample())
            #     import time
            #     time.sleep(0.1)

            input("")
