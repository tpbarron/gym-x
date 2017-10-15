import os
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv, HalfCheetahBulletEnv
from gym import spaces

class Walker2DBulletEnvX(Walker2DBulletEnv):

    def __init__(self):
        Walker2DBulletEnv.__init__(self)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,))

    def _get_obs(self):
        return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        obs, rew, done, info = super()._step(a)
        if self.robot.body_xyz[0] > 5:
            rew = 1.0
        else:
            rew = 0.0
        return obs, rew, done, info

        # if not self.scene.multiplayer:
        #     # if multiplayer, action first applied to all robots,
        #     # then global step() called, then _step() for all robots with the same actions
        #     self.robot.apply_action(a)
        #     self.scene.global_step()
        #
        # state = self.get_state()
        # _ = self.robot.calc_state()  # also calculates self.joints_at_limit
        # # state[0] is body height above ground, body_rpy[1] is pitch
        # # if self.robot.initial_z is None:
        # #     self.robot.initial_z = self.robot.body_xyz[2]
        # # body_height = self.body_xyz[2] - self.initial_z
        # # print (self.robot.body_xyz)
        # alive = float(self.robot.alive_bonus(self.robot.body_xyz[2], self.robot.body_rpy[1]))
        # # print ("alive: ", alive)
        # done = alive < 0
        # if not np.isfinite(state).all():
        #     print("~INF~", state)
        #     done = True
        #
        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        # progress = float(self.potential - potential_old)
        # # NOTE: progress = 0 in expl case
        # feet_collision_cost = 0.
        #
        # electricity_cost = 0 # self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        # electricity_cost += self.stall_torque_cost * float(np.square(a).sum())
        #
        # joints_at_limit_cost = 0 #float(self.joints_at_limit_cost * self.robot.joints_at_limit)
        # debugmode=0
        # if(debugmode):
        #     print("alive=", alive)
        #     print("progress=",progress)
        #     print("electricity_cost=",electricity_cost)
        #     print("joints_at_limit_cost=",joints_at_limit_cost)
        #     print("feet_collision_cost=",feet_collision_cost)
        #
        # self.rewards = [
        #     alive,
        #     progress,
        #     electricity_cost,
        #     joints_at_limit_cost,
        #     feet_collision_cost
        # ]
        # if (debugmode):
        #     print("rewards=",self.rewards)
        #     print("sum rewards=",sum(self.rewards))
        #
        # self.HUD(state, a, done)
        # self.reward += sum(self.rewards)
        # return state, sum(self.rewards), bool(done), {}

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


# from pybullet_envs.robot_locomotors import WalkerBase
# from pybullet_envs.gym_locomotion_envs import WalkerBaseBulletEnv
# class MujocoHalfCheetah(WalkerBase):
#     foot_list = ["ffoot", "fshin", "fthigh",  "bfoot", "bshin", "bthigh"]  # track these contacts with ground
#
#     def __init__(self):
#         WalkerBase.__init__(self, "mujoco_half_cheetah.xml", "torso", action_dim=6, obs_dim=26, power=0.90)
#
#     def alive_bonus(self, z, pitch):
#         # Use contact other than feet to terminate episode: due to a lot of strange walks using knees
#         return +1 if np.abs(pitch) < 1.0 and not self.feet_contact[1] and not self.feet_contact[2] and not self.feet_contact[4] and not self.feet_contact[5] else -1
#
#     def robot_specific_reset(self):
#         WalkerBase.robot_specific_reset(self)
#         self.jdict["bthigh"].power_coef = 120.0
#         self.jdict["bshin"].power_coef  = 90.0
#         self.jdict["bfoot"].power_coef  = 60.0
#         self.jdict["fthigh"].power_coef = 140.0
#         self.jdict["fshin"].power_coef  = 60.0
#         self.jdict["ffoot"].power_coef = 30.0
#
# class HalfCheetahBulletEnv(WalkerBaseBulletEnv):
#
#     def __init__(self):
#         self.robot = MujocoHalfCheetah()
#         WalkerBaseBulletEnv.__init__(self, self.robot)
#         print (dir(self.robot), self.robot.parts)


class HalfCheetahBulletEnvX(HalfCheetahBulletEnv):

        def __init__(self):
            HalfCheetahBulletEnv.__init__(self)
            # self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(22,))

        def _get_obs(self):
            return np.array([j.current_relative_position() for j in self.robot.ordered_joints], dtype=np.float32).flatten()

        def _step(self, a):
            """
            Duplicate of super class so that can modify rewards
            """
            obs, rew, done, info = super()._step(a)
            # if self.robot.body_xyz[0] > 5:
            #     rew = 1.0
            # else:
            #     rew = 0.0
            return obs, rew, done, info

        def _reset(self):
            state = super()._reset()
            return state

class HalfCheetahVisionBulletEnvX(HalfCheetahBulletEnvX):

    def __init__(self, render_dims=(84, 84)):
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
        self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.0005], physicsClientId=self.physicsClientId)
        ground_plane_mjcf = p.loadMJCF("ground_plane.xml") # at 0, 0, 0.001
        for i in ground_plane_mjcf:
            p.changeVisualShape(i,-1,rgbaColor=[0,0,0,0])
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
