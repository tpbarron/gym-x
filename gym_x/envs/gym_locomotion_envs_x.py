import os
import cv2
import numpy as np
import pybullet as p
# import pybullet_data
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, HalfCheetahBulletEnv, HopperBulletEnv, AntBulletEnv
from gym import spaces

class AntBulletEnvX(AntBulletEnv):

    def __init__(self):
        AntBulletEnv.__init__(self)
        self.electricity_cost = -0.25 #2.0	# cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
        # self.stall_torque_cost = 0. #-0.1	# cost for running electric current through a motor even at zero rotational speed, small
        # self.foot_collision_cost  = 0. #-1.0	# touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
        # self.foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
        # self.joints_at_limit_cost = 0. #-0.1	# discourage stuck joints

    # def _get_obs(self):
    #     return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        # potential_old = self.potential
        obs, rew, done, info = super()._step(a)
        # state = self.robot.calc_state()
        # alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))
        # alive *= 0.05
        # cost = 0.01 * -np.square(a).sum()
        # progress = float(self.potential - potential_old)
        # # print ("Rewarsd", alive, progress)
        # rew = alive + progress + cost
        # # if self.robot.body_xyz[0] > 5:
        # #     rew = 1.0
        # # else:
        # #     rew = 0.0
        # # print ("ROBOT: ", self.robot.body_xyz[2] < 0.3)
        # # if done:
        # #     print ("DONE")
        return obs, rew, done, info

    def _reset(self):
        state = super()._reset()
        return state

class Walker2DBulletEnvX(Walker2DBulletEnv):

    def __init__(self):
        Walker2DBulletEnv.__init__(self)
        # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,))
        self.electricity_cost = -0.5 #2.0	# cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
        # self.stall_torque_cost = 0. #-0.1	# cost for running electric current through a motor even at zero rotational speed, small
        # self.foot_collision_cost  = 0. #-1.0	# touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
        # self.foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
        # self.joints_at_limit_cost = 0. #-0.1	# discourage stuck joints

    # def _get_obs(self):
    #     return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        # potential_old = self.potential
        obs, rew, done, info = super()._step(a)
        # state = self.robot.calc_state()
        # alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))
        # alive *= 0.01

        # cost = 0.001 * -np.square(a).sum()

        # progress = float(self.potential - potential_old)
        # print ("Rewarsd", alive, progress)
        # rew = alive + progress + cost
        # if self.robot.body_xyz[0] > 5:
        #     rew = 1.0
        # else:
        #     rew = 0.0
        return obs, rew, done, info

    def _reset(self):
        state = super()._reset()
        return state

class Walker2DVisionBulletEnvX(Walker2DBulletEnvX):

    def __init__(self, render_dims=(84, 84)):
        Walker2DBulletEnvX.__init__(self)
        self.render_dims = render_dims
        # The observation is a combination of joints and image
        # print (self.observation_space, self.observation_space.low, self.observation_space.high)
        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(1, *render_dims)),
        #                                         self.observation_space))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))

    def get_render_obs(self):
        """
        Compute first-person view from robot
        """
        x, y, z = self.robot.body_xyz
        # print (x, y, z)
        cameraEyePosition = list([x, y-0.75, 1.0])
        cameraTargetPosition = [x, y, 1.0]
        cameraUpVector = [0, 0, 1]

        fov = 120
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.05 # this ensures outside body, may see limbs
        farPlane = 100.0

        # TODO: fix me to be along moving axis
        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=self.physicsClientId)
        # viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = p.getCameraImage(self.render_dims[0], self.render_dims[1], viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        # w=img_arr[0] #width of the image, in pixels
        # h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape((1, *self.render_dims))
        gray[gray > 0] = 255
        return gray

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        state, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        return render, sum(self.rewards), bool(done), {}

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.001], physicsClientId=self.physicsClientId)

        for i in range(-2, 6):
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 1.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 2.5], physicsClientId=self.physicsClientId)

        # self.cube_id = p.loadURDF("cube_red.urdf", basePosition=[-1, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_red.urdf", basePosition=[-1, 1, 1.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_lime.urdf", basePosition=[0, 1, 1.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_lime.urdf", basePosition=[0, 1, 0.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_blue.urdf", basePosition=[1, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_blue.urdf", basePosition=[1, 1, 1.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_yellow.urdf", basePosition=[2, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_yellow.urdf", basePosition=[2, 1, 1.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_cyan.urdf", basePosition=[3, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_cyan.urdf", basePosition=[3, 1, 1.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_magenta.urdf", basePosition=[4, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_magenta.urdf", basePosition=[4, 1, 1.5], physicsClientId=self.physicsClientId)
        #
        # self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 1, 0.5], physicsClientId=self.physicsClientId)
        # self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 1, 1.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        obs = super()._reset()
        self.build_path()
        render = self.get_render_obs()
        return render
        # return (render, obs)


class HopperBulletEnvX(HopperBulletEnv):
        def __init__(self):
            HopperBulletEnv.__init__(self)
            # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,))
            self.electricity_cost = -0.5 #2.0	# cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            # self.stall_torque_cost = 0. #-0.1	# cost for running electric current through a motor even at zero rotational speed, small
            # self.foot_collision_cost  = 0. #-1.0	# touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
            # self.foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
            # self.joints_at_limit_cost = 0. #-0.1	# discourage stuck joints
            # self.max_episode_steps = max_episode_steps
            # self.steps = 0
            # self.threshold = 1

        # def _get_obs(self):
        #     return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

        def _step(self, a):
            """
            Duplicate of super class so that can modify rewards
            """
            obs, rew, done, info = super()._step(a)
            # rew = +1 if past int threshold for first time in episode
            # if self.robot.body_xyz[0] > self.threshold:
            #     self.threshold += 1
            #     rew = 1.0
            # else:
            #     rew = 0.0
            # self.steps += 1
            # if self.steps > self.max_episode_steps:
            #     done = True
            return obs, rew, done, info

        def _reset(self):
            state = super()._reset()
            return state

class HopperVisionBulletEnvX(HopperBulletEnvX):

    def __init__(self,
                 render_dims=(64, 64),
                 camera_type='fixed'):
        """ Valid camera types are
            'follow': move camera perfectly with robot
            'fixed': do not move at all
            'incremental': move when robot exits frame
        """
        HopperBulletEnvX.__init__(self)
        self.render_dims = render_dims
        self.camera_type = camera_type
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))

    def get_render_obs(self):
        """
        Compute first-person view from robot
        """
        x, y, z = self.robot.body_xyz
        # print (x, y, z)

        # if self.camera_type == 'follow':
        cameraEyePosition = [x, y-1.25, 1.0]
        cameraTargetPosition = [x, y, 1.0]
        # elif self.camera_type == 'fixed':
        #     cameraEyePosition = [1.0, y-2.0, 1.0]
        #     cameraTargetPosition = [1.0, y, 1.0]

        cameraUpVector = [0, 0, 1]

        fov = 90
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.05 # this ensures outside body, may see limbs
        farPlane = 100.0

        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=self.physicsClientId)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = p.getCameraImage(self.render_dims[0], self.render_dims[1], viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb=img_arr[2] #color data RGB
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape((1, *self.render_dims))
        gray[gray > 0] = 255

        # assign patch at bottom to show distance, this is to differentiate frames
        # bar_width_pix = int(y/5.0*self.render_dims[1])
        # bar_height_pix = 10
        # gray[0][self.render_dims[0]-bar_height_pix:, 0:bar_width_pix] = 255
        return gray

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        state, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        if self.robot.body_xyz[0] > 5.0:
            done = True
        return render, rew, done, info

    def build_path(self):
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.0005], physicsClientId=self.physicsClientId)
        ground_plane_mjcf = p.loadMJCF("ground_plane.xml") # at 0, 0, 0.001
        for i in ground_plane_mjcf:
            p.changeVisualShape(i,-1,rgbaColor=[0,0,0,0])
        for i in range(-5, 6):
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 1.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 2.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 3.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        obs = super()._reset()
        self.build_path()
        render = self.get_render_obs()
        return render


class HalfCheetahVisionBulletEnv(HalfCheetahBulletEnv):

    def __init__(self, render_dims=(32, 32)):
        HalfCheetahBulletEnv.__init__(self)
        self.render_dims = render_dims
        # The observation is a combination of joints and image
        # print (self.observation_space, self.observation_space.low, self.observation_space.high)
        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(1, *render_dims)),
        #                                         self.observation_space))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))

    def get_render_obs(self):
        """
        Compute first-person view from robot
        """
        x, y, z = self.robot.body_xyz
        # print (x, y, z)
        cameraEyePosition = list([x, y-0.75, z])
        cameraTargetPosition = [x, y, z]
        cameraUpVector = [0, 0, 1]

        fov = 120
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.05 # this ensures outside body, may see limbs
        farPlane = 100.0

        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=self.physicsClientId)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = p.getCameraImage(self.render_dims[0], self.render_dims[1], viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb=img_arr[2] #color data RGB
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape((1, *self.render_dims))
        gray[gray > 0] = 255

        # assign patch at bottom to show distance, this is to differentiate frames
        bar_width_pix = int(y/5.0*self.render_dims[1])
        bar_height_pix = 10
        gray[0][self.render_dims[0]-bar_height_pix:, 0:bar_width_pix] = 255
        return gray

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        state, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        return render, rew, done, info

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        # self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.0005], physicsClientId=self.physicsClientId)
        # ground_plane_mjcf = p.loadMJCF("ground_plane.xml") # at 0, 0, 0.001
        # for i in ground_plane_mjcf:
            # p.changeVisualShape(i,-1,rgbaColor=[0,0,0,0])
        for i in range(-2, 6):
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 1.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 2.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        obs = super()._reset()
        # self.build_path()
        render = self.get_render_obs()
        return render

class HalfCheetahBulletEnvX(HalfCheetahBulletEnv):

        def __init__(self):
            HalfCheetahBulletEnv.__init__(self)

            self.electricity_cost = -0.5 #2.0	# cost for using motors -- this parameter should be carefully tuned against reward for making progress, other values less improtant
            # self.stall_torque_cost = 0. #-0.1	# cost for running electric current through a motor even at zero rotational speed, small
            # self.foot_collision_cost  = 0. #-1.0	# touches another leg, or other objects, that cost makes robot avoid smashing feet into itself
            # self.foot_ground_object_names = set(["floor"])  # to distinguish ground and other objects
            # self.joints_at_limit_cost = 0. #-0.1	# discourage stuck joints
            # self.max_episode_steps = max_episode_steps
            # self.steps = 0
            # self.threshold = 1

        # def _get_obs(self):
        #     return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

        def _step(self, a):
            """
            Duplicate of super class so that can modify rewards
            """
            obs, rew, done, info = super()._step(a)
            # if self.robot.body_xyz[0] > self.threshold:
            #     rew = 1.0
            #     self.threshold += 1
            # else:
            #     rew = 0.0
            # self.steps += 1
            # if self.steps > self.max_episode_steps:
            #     done = True
            return obs, rew, done, info

        def _reset(self):
            # self.steps = 0
            # self.threshold = 1
            state = super()._reset()
            return state

class HalfCheetahVisionBulletEnvX(HalfCheetahBulletEnvX):

    def __init__(self, render_dims=(32, 32)):
        HalfCheetahBulletEnvX.__init__(self)
        self.render_dims = render_dims
        # The observation is a combination of joints and image
        # print (self.observation_space, self.observation_space.low, self.observation_space.high)
        # self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(1, *render_dims)),
        #                                         self.observation_space))
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))

    def get_render_obs(self):
        """
        Compute first-person view from robot
        """
        x, y, z = self.robot.body_xyz
        # print (x, y, z)
        cameraEyePosition = list([x, y-0.75, z])
        cameraTargetPosition = [x, y, z]
        cameraUpVector = [0, 0, 1]

        fov = 120
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.05 # this ensures outside body, may see limbs
        farPlane = 100.0

        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=self.physicsClientId)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = p.getCameraImage(self.render_dims[0], self.render_dims[1], viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        rgb=img_arr[2] #color data RGB
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = gray.reshape((1, *self.render_dims))
        gray[gray > 0] = 255

        # assign patch at bottom to show distance, this is to differentiate frames
        bar_width_pix = int(y/5.0*self.render_dims[1])
        bar_height_pix = 10
        gray[0][self.render_dims[0]-bar_height_pix:, 0:bar_width_pix] = 255
        return gray

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        state, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        return render, rew, done, info

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        # self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.0005], physicsClientId=self.physicsClientId)
        # ground_plane_mjcf = p.loadMJCF("ground_plane.xml") # at 0, 0, 0.001
        # for i in ground_plane_mjcf:
            # p.changeVisualShape(i,-1,rgbaColor=[0,0,0,0])
        for i in range(-2, 6):
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 1.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 2.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        obs = super()._reset()
        self.build_path()
        render = self.get_render_obs()
        return render
        # return (render, obs)
