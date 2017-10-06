"""
Linear chain env in pybullet where reward is given for reaching end of path
"""

import os
import numpy as np
import cv2
import gym
from gym import spaces
import pybullet as p
import pybullet_data
# import pybullet_envs
from pybullet_envs.gym_locomotion_envs import AntBulletEnv

print("\n".join(['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet')>=0]))

# if __name__ == '__main__':
#     env = AntBulletEnv() #gym.make('AntBulletEnv-v0')
#     env.render(mode='human')
#     env.reset()
#     done = False
#     while not done:
#         env.step(env.action_space.sample())

i = 0
#TODO: update pybullet
class ChainEnvX(AntBulletEnv):

    def __init__(self,
                 render_dims=(84, 84)):
        super(ChainEnvX, self).__init__()
        self.render_dims = render_dims
        # The observation is a combination of joints and image
        self.observation_space = spaces.Tuple((self.observation_space,
                                               spaces.Box(low=0, high=255, shape=render_dims)))

    def get_render_obs(self):
        """
        Compute first-person view from robot

        """
        euler = p.getEulerFromQuaternion(self.robot.robot_body.current_orientation())
        yaw, pitch, roll = euler
        cameraEyePosition = list(self.robot.body_xyz)
        cameraTargetPosition = [np.cos(yaw)*np.cos(pitch), np.sin(yaw)*np.cos(pitch), np.sin(pitch)]
        cameraTargetPosition = [10*x for x in cameraTargetPosition]
        # cameraTargetPosition = [10, 0., 0.5]
        cameraUpVector = [0, 0, 1]

        fov = 120
        aspect = self.render_dims[0] / self.render_dims[1]
        nearPlane = 0.26 # this ensures outside body, may see limbs
        farPlane = 2.0 # TODO: perhaps changing this will make the movement more "surprising?"

        # TODO: fix me to be along moving axis
        viewMatrix = p.computeViewMatrix(cameraEyePosition, cameraTargetPosition, cameraUpVector, physicsClientId=self.physicsClientId)
        # viewMatrix = p.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        projectionMatrix = p.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = p.getCameraImage(self.render_dims[0], self.render_dims[1], viewMatrix, projectionMatrix, renderer=p.ER_BULLET_HARDWARE_OPENGL, physicsClientId=self.physicsClientId)

        # w=img_arr[0] #width of the image, in pixels
        # h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        # global i
        # cv2.imwrite('imgs/test'+str(i)+'.png', gray)
        # print (rgb.shape) (w, h, 4)
        # from PIL import Image
        # I = Image.fromarray(rgb)
        # I = I.convert("L")
        # I.save('imgs/test'+str(i)+'.png')
        # rgb = np.array(I)
        # print (rgb.shape)
        # I.show()
        # input("")
        # I.close()
        # dep=img_arr[3] #depth data
        # i += 1

        return gray

    def _step(self, a):
        """
        Mostly directly copied from `WalkerBaseBulletEnv` at
        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/gym/pybullet_envs/gym_locomotion_envs.py

        But wanted to have access to all rewards so could manually determine which ones to use
        """
        if not self.scene.multiplayer:  # if multiplayer, action first applied to all robots, then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit

        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))   # state[0] is body height above ground, body_rpy[1] is pitch
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        # progress = float(self.potential - potential_old)

        feet_collision_cost = 0.0
        for i,f in enumerate(self.robot.feet): # TODO: Maybe calculating feet contacts could be done within the robot code
            contact_ids = set((x[2], x[4]) for x in f.contact_list())
            #print("CONTACT OF '%d' WITH %d" % (contact_ids, ",".join(contact_names)) )
            if (self.ground_ids & contact_ids):
                #see Issue 63: https://github.com/openai/roboschool/issues/63
                #feet_collision_cost += self.foot_collision_cost
                self.robot.feet_contact[i] = 1.0
            else:
                self.robot.feet_contact[i] = 0.0
        electricity_cost  = self.electricity_cost  * float(np.abs(a*self.robot.joint_speeds).mean())  # let's assume we have DC motor with controller, and reverse current braking
        electricity_cost += self.stall_torque_cost * float(np.square(a).mean())

        joints_at_limit_cost = float(self.joints_at_limit_cost * self.robot.joints_at_limit)

        debugmode=0
        if(debugmode):
            print("alive=")
            print(alive)
            print("progress")
            print(progress)
            print("electricity_cost")
            print(electricity_cost)
            print("joints_at_limit_cost")
            print(joints_at_limit_cost)
            print("feet_collision_cost")
            print(feet_collision_cost)

        # set progress cost to 0 for exploration task
        progress = 0

        # check if reached terminal
        goal_rew = 0.
        if self.robot.body_xyz[0] > 3:
            goal_rew = 1.
            done = True

        self.rewards = [
            alive,
            progress,
            goal_rew,
            electricity_cost,
            joints_at_limit_cost,
            feet_collision_cost
        ]

        if (debugmode):
            print("rewards=")
            print(self.rewards)
            print("sum rewards")
            print(sum(self.rewards))

        self.HUD(state, a, done)
        self.reward += sum(self.rewards)

        render = self.get_render_obs()

        obs = (state, render)
        assert self.observation_space.contains(obs)
        return obs, sum(self.rewards), bool(done), {}

    # def _step(self, action):
    #     obs, rew, done, info = AntBulletEnv._step(self, action)
    #     # TODO: concat observation with render in tuple
    #     # TODO: specify reward = 1 iff at terminal
    #     # TODO: decide whether to cut out movement costs as well?
    #     print (info)
    #     if self.robot.body_xyz[0] > 4:
    #         done = True
    #     if done:
    #         rew += 1
    #     render = self.get_render_obs()
    #     # input("")
    #     return obs, rew, done, info

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath("assets/") #used by loadURDF
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, -1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, 0, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_red.urdf", basePosition=[-1, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_red.urdf", basePosition=[-1, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_lime.urdf", basePosition=[0, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_lime.urdf", basePosition=[0, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_blue.urdf", basePosition=[1, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_blue.urdf", basePosition=[1, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_yellow.urdf", basePosition=[2, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_yellow.urdf", basePosition=[2, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_cyan.urdf", basePosition=[3, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_cyan.urdf", basePosition=[3, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_magenta.urdf", basePosition=[4, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_magenta.urdf", basePosition=[4, 2, 0.5], physicsClientId=self.physicsClientId)

        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, -2, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, -1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 0, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 2, 0.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        AntBulletEnv._reset(self)
        self.build_path()

    def _render(self, **kwargs):
        AntBulletEnv._render(self, **kwargs)

if __name__ == '__main__':
    env = ChainEnvX()
    env.render(mode='human')
    env.reset()
    while True:
        env.step(env.action_space.sample())
        # break

    # env.reset()
