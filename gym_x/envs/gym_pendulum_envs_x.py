
import pybullet as p
import cv2
import os
import numpy as np
from gym import spaces
from pybullet_envs.gym_pendulum_envs import InvertedPendulumSwingupBulletEnv

class InvertedPendulumSwingupVisionBulletEnv(InvertedPendulumSwingupBulletEnv):

    def __init__(self, render_dims=(64, 64)):
        super(InvertedPendulumSwingupVisionBulletEnv, self).__init__()
        # InvertedPendulumSwingupBulletEnv.__init__(self)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))
        self.render_dims = render_dims

    def get_render_obs(self):
        x, y, z = self.robot.robot_body.current_position() #self.robot.body_xyz
        # print (x, y, z)
        cameraEyePosition = list([0, y-0.65, 0.])
        cameraTargetPosition = [0, y, 0.0]
        cameraUpVector = [0, 0, 1]

        fov = 120
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.05 # this ensures outside body, may see limbs
        farPlane = 100.0

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

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        # self.plane_id = p.loadURDF("plane_black.urdf", basePosition=[0, 0, 0.001], physicsClientId=self.physicsClientId)

        for i in range(-2, 3):
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, -1.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, -0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 0.5], physicsClientId=self.physicsClientId)
            self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[i, 1, 1.5], physicsClientId=self.physicsClientId)


    def _step(self, a):
        obs, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        # print ("Reward: ", rew)
        # if done:
        #     print ("!!!!!!!!!!!!!!!!!!!PENDULUM DONE!!!!!!!!!!!!!!!!")
        return render, rew, done, info

    def _reset(self):
        obs = super()._reset()
        self.build_path()
        render = self.get_render_obs()
        return render


class InvertedPendulumSwingupBulletEnvX(InvertedPendulumSwingupBulletEnv):

    def __init__(self, max_episode_steps=500):
        InvertedPendulumSwingupBulletEnv.__init__(self)
        self.max_episode_steps = max_episode_steps
        self.steps = 0

    def _step(self, a):
        obs, rew, done, info = super()._step(a)
        if done:
            rew = 1.0
        else:
            rew = 0.0
        self.steps += 1
        if self.steps > self.max_episode_steps:
            done = True
        return obs, rew, done, info

    def _reset(self):
        self.steps = 0
        return super()._reset()

class InvertedPendulumSwingupVisionBulletEnvX(InvertedPendulumSwingupBulletEnvX):

    def __init__(self, render_dims=(32, 32)):
        InvertedPendulumSwingupBulletEnvX.__init__(self)
        self.observation_space = spaces.Box(low=0, high=255, shape=(1, *render_dims))
        self.render_dims = render_dims

    def get_render_obs(self):
        x, y, z = self.robot.robot_body.current_position() #self.robot.body_xyz
        # print (x, y, z)
        cameraEyePosition = list([x, y-0.65, 0.])
        cameraTargetPosition = [x, y, 0.0]
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
        # gray[gray > 0] = 255
        return gray

    def _step(self, a):
        obs, rew, done, info = super()._step(a)
        render = self.get_render_obs()
        return render, rew, done, info

    def _reset(self):
        obs = super()._reset()
        render = self.get_render_obs()
        return render
