"""
Adapted from: https://raw.githubusercontent.com/bulletphysics/bullet3/master/examples/pybullet/gym/pybullet_envs/bullet/racecarGymEnv.py
"""

import os #inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(os.path.dirname(currentdir))
# os.sys.path.insert(0,parentdir)

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

from gym_x.envs.pomdp import racecar
from gym_x.wrappers import DataLoggerEnv

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class TMazeRacecarGymEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 50
    }

    def __init__(self,
                 urdf_root=pybullet_data.getDataPath(),
                 action_repeat=10,
                 # action_repeat=50,
                 is_enable_self_collision=True,
                 is_discrete=False,
                 renders=False,
                 length=1, # inner length of tmaze / grid section
                 width=2, # width from center point between walls
                 deterministic=False,
                 alternate=True,
                 r_type='neg_dist',
                 map_type='tmaze', # tmaze or grid
                 randomize_start=True,
                 randomize_signals=False):
        self.time_step = 0.01
        # self.time_step = 0.005
        self.urdf_root = urdf_root
        self.action_repeat = action_repeat
        self.is_enable_self_collision = is_enable_self_collision
        self.env_step_counter = 0
        self.renders = renders
        self.is_discrete = is_discrete
        if self.renders:
            self.p = bullet_client.BulletClient(connection_mode=pybullet.GUI)
        else:
            self.p = bullet_client.BulletClient()

        observation_dim = 27 #14 #5
        observation_high = np.ones(observation_dim) * 1000 #np.inf

        if (is_discrete):
            self.action_space = spaces.Discrete(3)
        else:
            action_dim = 2
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high)

        self.observation_space = spaces.Box(-observation_high, observation_high)
        self.viewer = None
        self.length = length
        self.width = width
        self.deterministic = deterministic
        self.alternate = alternate
        self.map_type = map_type
        self.randomize_start = randomize_start
        self.randomize_signals = randomize_signals

        self.wall_block_ids = []
        self.signal_block_ids = set()
        self.radar_ids = []

        self.switch = None
        self.max_x, self.min_y, self.max_y = None, None, None
        self.r_type = r_type

    def __del__(self):
        self.p = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def make_rect(self, x, y, w, h, reds=[]):
        """
        Make an empty block rect
        w, h must be ints
        """
        for i in range(w):
            for j in range(h):
                px = x+i
                py = y+j
                if i == 0 or i == w-1:
                    if [px, py] in reds:
                        cube_id = self.p.loadURDF("cube_red.urdf", basePosition=[px, py, 0.5])
                    else:
                        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[px, py, 0.5])
                elif j == 0 or j == h-1:
                    if [px, py] in reds:
                        cube_id = self.p.loadURDF("cube_red.urdf", basePosition=[px, py, 0.5])
                    else:
                        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[px, py, 0.5])
                self.wall_block_ids.append(cube_id)

    def make_wall(self, x, y, d, length=5):
        for i in range(length):
            cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[x, y, 0.5])
            self.wall_block_ids.append(cube_id)
            if d == 0:
                x += 1
            elif d == 1:
                y += 1

    def successors(self, prev, curr):
        """
        Get successor locs for current position, prev matters since can't do u-turn
        """
        xpos = curr % 3
        ypos = curr // 3

        succs = []
        if xpos != 2:
            succs.append([xpos+1, ypos])
        if xpos != 0:
            succs.append([xpos-1, ypos])
        if ypos != 2:
            succs.append([xpos, ypos+1])
        if ypos != 0:
            succs.append([xpos, ypos-1])

        print (prev, curr, succs)
        if prev != -1:
            xpos_prev = prev % 3
            ypos_prev = prev // 3
            succs.remove([xpos_prev, ypos_prev])

        indices = []
        for i in range(len(succs)):
            ind = succs[i][1]*3 + succs[i][0]
            indices.append(ind)
        return indices

    def generate_random_path(self, steps=3):
        path = []
        start = np.random.randint(0, 9)
        prev = -1
        curr = start
        path.append(start)
        for i in range(steps):
            succ = self.successors(prev, curr)
            # print (succ)
            nxt = succ[np.random.randint(0, len(succ))]
            # print (nxt)
            path.append(nxt)
            prev = curr
            curr = nxt
        return path

    def get_grid_position(self, i):
        xpos = i % 3
        ypos = i // 3

        truex = -1
        truey = -1
        if xpos == 0:
            truey = 9
        elif xpos == 1:
            truey = 0
        elif xpos == 2:
            truey = -9

        if ypos == 0:
            truex = 18
        elif ypos == 1:
            truex = 9
        elif ypos == 2:
            truex = 0
        return [truex, truey]

    def get_starting_orientation(self, path):
        i1 = path[0]
        xpos1 = i1 % 3
        ypos1 = i1 // 3

        i2 = path[1]
        xpos2 = i2 % 3
        ypos2 = i2 // 3

        zorient = 0.0

        if xpos2 < xpos1:
            zorient = np.pi/2.
        if xpos2 > xpos1:
            zorient = -np.pi/2.
        if ypos2 > ypos1:
            zorient = np.pi
        if ypos2 < ypos1:
            zorient = 0
        return self.p.getQuaternionFromEuler([0, 0, zorient])

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def get_turn_direction(self, i1, i2, i3):
        """
        return 0 if left, 1 if straight, 2 if right
        """
        xpos1 = i1 % 3
        ypos1 = i1 // 3

        xpos2 = i2 % 3
        ypos2 = i2 // 3

        xpos3 = i3 % 3
        ypos3 = i3 // 3

        if xpos1 == xpos2 and xpos2 == xpos3 or ypos1 == ypos2 and ypos2 == ypos3:
            return 1

        vec_turn = self.unit_vector(np.array([xpos3 - xpos1, ypos3 - ypos1]))
        vec_straight = self.unit_vector(np.array([xpos2 - xpos1, ypos2 - ypos1]))
        # print (vec_turn, vec_straight)
        # theta = np.arccos(np.dot(vec_turn, vec_straight))

        # https://stackoverflow.com/questions/21483999/using-atan2-to-find-angle-between-two-vectors
        theta = np.arctan2(vec_turn[1], vec_turn[0]) - np.arctan2(vec_straight[1], vec_straight[0]);

        if theta < 0:
            theta += 2 * np.pi;

        print ("turn: ", i1, i2, i3, theta)
        if theta > np.pi:
            return 0
        return 2

    def get_red_block_position(self, i1, i2, turn_dir):
        grid_pos1 = self.get_grid_position(i1)
        grid_pos2 = self.get_grid_position(i2)

        block_pos = [grid_pos1[0], grid_pos1[1]]

        # one of these should be 0
        dx = grid_pos2[0] - grid_pos1[0]
        dy = grid_pos2[1] - grid_pos1[1]

        # need 2.5 block shift toward pos2
        rand = 1 if self.randomize_signals else 0
        delta = rand * np.random.randint(3)

        if dx != 0:
            block_pos[0] += np.sign(dx) * (2.5 + delta)
        elif dy != 0:
            block_pos[1] += np.sign(dy) * (2.5 + delta)

        if turn_dir == 0:
            # left
            if dx != 0:
                if np.sign(dx) == 1:
                    block_pos[1] += 1.5
                elif np.sign(dx) == -1:
                    block_pos[1] -= 1.5
            elif dy != 0:
                if np.sign(dy) == 1:
                    block_pos[0] -= 1.5
                elif np.sign(dy) == -1:
                    block_pos[0] += 1.5

        elif turn_dir == 2:
            # right
            if dx != 0:
                if np.sign(dx) == 1:
                    block_pos[1] -= 1.5
                elif np.sign(dx) == -1:
                    block_pos[1] += 1.5
            elif dy != 0:
                if np.sign(dy) == 1:
                    block_pos[0] += 1.5
                elif np.sign(dy) == -1:
                    block_pos[0] -= 1.5

        return block_pos

    def build_grid(self, path):
        # use path to set red blocks
        self.wall_block_ids = []
        self.signal_block_ids = set()
        self.p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF

        block_pos = []
        for i in range(len(path)-2):
            turn_dir = self.get_turn_direction(*path[i:i+3])
            # if dir == 1, straight, do nothing
            # if dir == 0, left, get block on left side of path and add red block
            if turn_dir != 1:
                bpos = self.get_red_block_position(path[i], path[i+1], turn_dir)
                block_pos.append(bpos)

        self.make_rect(1.5, 1.5, 7, 7, reds=block_pos)
        self.make_rect(1.5, 1.5-2.0-7.0, 7, 7, reds=block_pos)

        self.make_rect(1.5+2.0+7.0, 1.5, 7, 7, reds=block_pos)
        self.make_rect(1.5+2.0+7.0, 1.5-2.0-7.0, 7, 7, reds=block_pos)

        self.make_wall(-1.5, -1.5-8.0, 1, length=20)
        self.make_wall(1.5+18, -1.5-8.0, 1, length=20)

        self.make_wall(-1.5, -2.5-8.0, 0, length=22)
        self.make_wall(-1.5, 2.5+8.0, 0, length=22)


    def build_tmaze(self):
        self.wall_block_ids = []
        self.signal_block_ids = set()

        self.p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[-1, -0.5, 0.5])
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[-1, 0.5, 0.5])
        self.wall_block_ids.append(cube_id)

        for i in range(-1, self.length):
            # place right block...
            if i == 1 and self.switch == -1:
                cube_id = self.p.loadURDF("cube_red.urdf", basePosition=[i, -(self.width+1)/2.0, 0.5])
                self.signal_block_ids.add(cube_id)
                self.wall_block_ids.append(cube_id)
            else:
                cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[i, -(self.width+1)/2.0, 0.5])
                self.wall_block_ids.append(cube_id)

            # place left block..
            if i == 1 and self.switch == 1:
                cube_id = self.p.loadURDF("cube_red.urdf", basePosition=[i, (self.width+1)/2.0, 0.5])
                self.signal_block_ids.add(cube_id)
                self.wall_block_ids.append(cube_id)
            else:
                cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[i, (self.width+1)/2.0, 0.5])
                self.wall_block_ids.append(cube_id)

        # right end corner
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length, -(self.width+1)/2-2, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length+1, -(self.width+1)/2-2, 0.5])
        self.wall_block_ids.append(cube_id)

        # left end corner
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length, (self.width+1)/2+2, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length+1, (self.width+1)/2+2, 0.5])
        self.wall_block_ids.append(cube_id)

        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length-1, -(self.width+1)/2-1, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length-1, -(self.width+1)/2-2, 0.5])
        self.wall_block_ids.append(cube_id)

        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length-1, (self.width+1)/2+1, 0.5])
        self.wall_block_ids.append(cube_id)
        cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length-1, (self.width+1)/2+2, 0.5])
        self.wall_block_ids.append(cube_id)

        for i in range(8):
            cube_id = self.p.loadURDF("cube_black.urdf", basePosition=[self.length+(self.width+1)-1, i-(self.width+1)-0.5, 0.5])
            self.wall_block_ids.append(cube_id)

    def reset(self):
        self.p.resetSimulation()
        self.p.setPhysicsEngineParameter(numSolverIterations=10)
        self.p.setTimeStep(self.time_step)
        stadiumobjects = self.p.loadSDF(os.path.join(self.urdf_root,"stadium.sdf"))
        #move the stadium objects slightly above 0
        for i in stadiumobjects:
            pos,orn = self.p.getBasePositionAndOrientation(i)
            newpos = [pos[0],pos[1],pos[2]-0.1]
            self.p.resetBasePositionAndOrientation(i,newpos,orn)

        self.p.setGravity(0, 0, -10)
        if self.map_type == 'grid':
            path = self.generate_random_path()
            print ("PATH: ", path)
            car_position = self.get_grid_position(path[0])
            car_position.append(0.2) # z
            # print (car_position)
            if self.randomize_start:
                car_position[0] += np.random.random() - 0.5
                car_position[1] += np.random.random() - 0.5
            car_orient = self.get_starting_orientation(path)
            # print (car_orient)
            self.racecar = racecar.Racecar(self.p, position=car_position, orientation=car_orient, urdfRootPath=self.urdf_root, timeStep=self.time_step)
        elif self.map_type == 'tmaze':
            car_position = [0, 0, 0.2]
            if self.randomize_start:
                car_position[0] += np.random.random() - 0.5
                car_position[1] += np.random.random() - 0.5
            self.racecar = racecar.Racecar(self.p, position=car_position, urdfRootPath=self.urdf_root, timeStep=self.time_step)

        self.env_step_counter = 0
        for i in range(100):
            self.p.stepSimulation()

        if self.deterministic:
            self.switch = 1 #if np.random.random() < 0.5 else -1
        elif self.alternate and self.switch is not None:
            self.switch = self.switch * -1
        else:
            self.switch = -1 if np.random.random() < 0.5 else 1
        print ("Goal: y = ", self.switch)

        # Build after switch so can set colored block
        if self.map_type == 'tmaze':
            self.build_tmaze()
            self.goal = np.array([self.length+0.5, self.switch*2])
        elif self.map_type == 'grid':
            self.build_grid(path)
            self.goal = self.get_grid_position(path[-1]) #p.array([self.length, self.switch])

        print ("Goal: ", self.goal)
        observation = self.get_extended_observation()
        self.max_x, self.min_y, self.max_y = 0, 0, 0
        return np.array(observation)


    ###
    # Observations
    ###

    def get_radar_observation(self):
        carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        xinit, yinit, zinit = carpos
        zinit += 0.5
        euler = self.p.getEulerFromQuaternion(carorn)
        # print (euler)
        zorient = euler[2]
        nrays = 13
        ray_cast_starts = [(xinit, yinit, zinit) for _ in range(nrays)]
        ray_cast_ends = []
        ray_length = 3.0
        sangle = -np.pi / 2 + zorient

        # remove old radars
        # for i in range(len(self.radar_ids)):
        #     self.p.removeUserDebugItem(self.radar_ids[i])
        # self.p.removeAllUserDebugItems()
        # self.radar_ids = []
        for i in range(nrays):
            theta = sangle #(float(i)/(nrays-1)) / (np.pi/10)
            x = ray_length * np.cos(theta)
            y = ray_length * np.sin(theta)
            ray_cast_ends.append([xinit+x, yinit+y, zinit])
            # rid = self.p.addUserDebugLine(lineFromXYZ=ray_cast_starts[i], lineToXYZ=ray_cast_ends[i])
            # self.radar_ids.append(rid)
            sangle += (np.pi/2.) / float(13 // 2)
        ray_casts = self.p.rayTestBatch(ray_cast_starts, ray_cast_ends)

        obj_ids = [ray_casts[i][0] for i in range(len(ray_casts))]
        signals = [1 if obj_ids[i] in self.signal_block_ids else 0 for i in range(len(obj_ids))]
        hits = [ray_casts[i][2] for i in range(len(ray_casts))]
        hits.extend(signals)
        return hits

    def get_extended_observation(self):
        """
        carx, cary, [carorn], signal
        """
        carpos,carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        # print (carpos)
        # print (carorn)
        # signal = [0, 0]
        # if self.env_step_counter == 0:
        #     # if on first step, also give goal signal
        #     signal = [1, 0] if self.switch == -1 else [0, 1]

        observation = []
        # if make this relative normed x and relative normed y then should be able to generalize
        # also add zorientation
        # self._observation.extend([carpos[0]/(self.length+0.5), (carpos[1]+1)/2., (carorn[2]+1)/2.])

        radars = self.get_radar_observation()
        observation.extend(radars)
        observation.extend([(carorn[2]+1)/2.])
        # self._observation.extend([carpos[0]/(self.length+0.5), (carpos[1]+1)/2., (carorn[2]+1)/2.])
        # self._observation.extend(signal)
        return observation

    def step(self, action):
        carPos,orn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        # if (self._renders):
        #     basePos,orn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        #     #self.p.resetDebugVisualizerCamera(1, 30, -40, basePos)

        if (self.is_discrete):
	        fwd = [2,2,2]
	        steerings = [-0.5,0,0.5]
	        # fwd = [-1,-1,-1,0,0,0,1,1,1]
	        # steerings = [-0.6,0,0.6,-0.6,0,0.6,-0.6,0,0.6]
	        forward = fwd[action]
	        steer = steerings[action]
	        realaction = [forward,steer]
        else:
            realaction = action

        self.racecar.applyAction(realaction)
        for i in range(self.action_repeat):
            self.p.stepSimulation()
            # if self.renders:
            #     time.sleep(self.time_step)
            # self._observation = self.getExtendedObservation()

            carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
            euler = self.p.getEulerFromQuaternion(carorn)
            self.p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=-90+np.rad2deg(euler[2]), cameraPitch=-60, cameraTargetPosition=carpos)

            if self._termination():
                break
        self.env_step_counter += 1
        observation = self.get_extended_observation()
        # compute reward based on movement before updated max and min x,y
        reward = self.reward()
        # check movement bounds
        x, y, z = carPos
        if x > self.max_x:
            self.max_x = x
        if x > self.length - 0.5 and y > self.max_y:
            self.max_y = y
        elif x > self.length - 0.5 and y < self.min_y:
            self.min_y = y

        done = self._termination()
        return np.array(observation), reward, done, {}

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        base_pos,orn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        view_matrix = self.p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=base_pos,
            distance=self._cam_dist,
            yaw=self._cam_yaw,
            pitch=self._cam_pitch,
            roll=0,
            upAxisIndex=2)
        proj_matrix = self.p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0)
        (_, _, px, _, _) = self.p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT, viewMatrix=view_matrix,
            projectionMatrix=proj_matrix, renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _in_goal_box(self):
        carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
        x, y, z = carpos

        if self.map_type == 'grid' or self.map_type == 'tmaze':
            if x > self.goal[0] - 1.0 and x < self.goal[0] + 1.0:
                if y > self.goal[1] - 1.0 and y < self.goal[1] + 1.0:
                    return True, False
            return False, False
        elif self.map_type == 'tmaze':
            own, other = False, False
            if x > self.length - self.width/2. and x < self.length + self.width/2.: # made it down the path...
                if y > self.switch*self.width - 0.5 and y < self.switch*self.width + 0.5:
                    own = True
                elif y > (-1*self.switch*self.width) - 0.5 and y < (-1*self.switch*self.width) + 0.5:
                    other = True
            return own, other

    def _is_wall_contact(self):
        contacts = self.p.getContactPoints(self.racecar.racecarUniqueId)
        # print (contacts)
        for c in contacts:
            # c[2] is bodyB id and c[-2] is distance
            if c[2] in self.wall_block_ids and c[-2] < 1e-8:
                return True
        return False

    def _closest_wall_dist(self):
        closest_dist = np.inf
        for wid in self.wall_block_ids:
            closest = self.p.getClosestPoints(self.racecar.racecarUniqueId, wid, 10.0)
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

    ###
    # Reward computation
    ###

    def reward(self):
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
            carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
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
    #         carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
    #         carxy = np.array(carpos[0:2])
    #         dist = np.linalg.norm(carxy - self.goal)
    #         reward = -dist
    #     return reward

    def _reward_neg_dist_wall_potential(self):
        """
        negative dist to goal
        """
        reward = 0.0
        carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
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
            carpos, carorn = self.p.getBasePositionAndOrientation(self.racecar.racecarUniqueId)
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
    env = TMazeRacecarGymEnv(is_discrete=True, renders=True, length=10, alternate=True, map_type='grid', randomize_start=False, randomize_signals=False)
    env = DataLoggerEnv(env)

    pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_GUI, 0)

    for t in range(20):
        print ("Run: ", t)
        pybullet.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=-90, cameraPitch=-60, cameraTargetPosition=[0, 0, 0])
        # pybullet.resetDebugVisualizerCamera(cameraDistance=12, cameraYaw=-90, cameraPitch=-80, cameraTargetPosition=[8, 0, 0])
        env.reset()

        done = False

        # input("")
        while not done:
            env.render()

            keys = pybullet.getKeyboardEvents()
            # print (keys)
            a = 1
            for k, v in keys.items():
                if k == pybullet.B3G_LEFT_ARROW and v == pybullet.KEY_IS_DOWN:
                    # print ('left')
                    a = 2
                elif k == pybullet.B3G_RIGHT_ARROW and v == pybullet.KEY_IS_DOWN:
                    # print ("right")
                    a = 0
                elif k == pybullet.B3G_UP_ARROW and v == pybullet.KEY_IS_DOWN:
                    # print ("up")
                    a = 1
                elif k == pybullet.B3G_DOWN_ARROW and v == pybullet.KEY_IS_DOWN:
                    print ("no brake!")

            # input("")
            obs, rew, done, info = env.step(a) #env.action_space.sample())
            # print ("Reward: ", rew)
            # import time
            # time.sleep(0.05)

            # for i in count(1):
            # # while not done:
            #     env.render()
            #     obs, rew, done, info = env.step(6) #env.action_space.sample())
            #     import time
            #     time.sleep(0.1)

            # input("")
        # save after each episode so can quit early...
        data = env.to_numpy()
        s, a, r, t = data
        # print (s.shape, a.shape, r.shape, t.shape)
        np.save('states.npy', s)
        np.save('actions.npy', a)
        np.save('rewards.npy', r)
        np.save('terminals.npy', t)

        # done here
    data = env.to_numpy()
    s, a, r, t = data
    print (s.shape, a.shape, r.shape, t.shape)
    np.save('states.npy', s)
    np.save('actions.npy', a)
    np.save('rewards.npy', r)
    np.save('terminals.npy', t)
