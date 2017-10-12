import os
import cv2
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_envs.gym_locomotion_envs import Walker2DBulletEnv
from gym import spaces

class Walker2DBulletEnvX(Walker2DBulletEnv):

    def __init__(self):
        Walker2DBulletEnv.__init__(self)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(14,)) # 22 - 8

    def _step(self, a):
        """
        Duplicate of super class so that can modify rewards
        """
        if not self.scene.multiplayer:
            # if multiplayer, action first applied to all robots,
            # then global step() called, then _step() for all robots with the same actions
            self.robot.apply_action(a)
            self.scene.global_step()

        state = self.robot.calc_state()  # also calculates self.joints_at_limit
        # state[0] is body height above ground, body_rpy[1] is pitch
        alive = float(self.robot.alive_bonus(state[0]+self.robot.initial_z, self.robot.body_rpy[1]))
        # alive *= 0.1
        done = alive < 0
        if not np.isfinite(state).all():
            print("~INF~", state)
            done = True

        # potential_old = self.potential
        # self.potential = self.robot.calc_potential()
        # progress = float(self.potential - potential_old)
        # NOTE: progress = 0 in expl case
        progress = 0.0
        # if self.robot.body_xyz[0] > self.max_x_dist:
        #     print (self.robot.body_xyz)
        #     self.max_x_dist = self.robot.body_xyz[0]
        # if self.robot.body_xyz[0] > 2:
        #     # abuse of variabe name, progress now +1 if movement > 5 units
        #     # this is the same as the VIME paper
        #     print ("X pos > 5!!!")
        #     progress = 1.0

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

        self.rewards = [
            alive,
            progress,
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
        state = state[8:]
        return state, sum(self.rewards), bool(done), {}

    def _reset(self):
        state = super()._reset()[8:]
        return state


class Walker2DVisionBulletEnvX(Walker2DBulletEnvX):

    def __init__(self, render_dims=(84, 84)):
        Walker2DBulletEnvX.__init__(self)
        self.render_dims = render_dims
        # The observation is a combination of joints and image
        print (self.observation_space, self.observation_space.low, self.observation_space.high)
        self.observation_space = spaces.Tuple((spaces.Box(low=0, high=255, shape=(1, *render_dims)),
                                                self.observation_space))

    def get_render_obs(self):
        """
        Compute first-person view from robot
        """
        euler = p.getEulerFromQuaternion(self.robot.robot_body.current_orientation())
        yaw, pitch, roll = euler
        x, y, z = self.robot.body_xyz
        print (x, y, z)
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
        Duplicate of super class so that can modify rewards
        """
        state, rew, done, info = super()._step(a)
        print (state.shape)
        print (state)
        input("")
        render = self.get_render_obs()
        return (render, state), sum(self.rewards), bool(done), {}

    def build_path(self):
        # print (pybullet_data.getDataPath())
        # p.setAdditionalSearchPath(pybullet_data.getDataPath()) #used by loadURDF
        p.setAdditionalSearchPath(os.path.join(os.path.dirname(__file__), "assets/")) #used by loadURDF
        self.cube_id = p.loadURDF("cube_black.urdf", basePosition=[-2, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_red.urdf", basePosition=[-1, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_lime.urdf", basePosition=[0, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_blue.urdf", basePosition=[1, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_yellow.urdf", basePosition=[2, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_cyan.urdf", basePosition=[3, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_magenta.urdf", basePosition=[4, 1, 0.5], physicsClientId=self.physicsClientId)
        self.cube_id = p.loadURDF("cube_white.urdf", basePosition=[5, 1, 0.5], physicsClientId=self.physicsClientId)

    def _reset(self):
        obs = super()._reset()
        self.build_path()
        render = self.get_render_obs()
        return (render, obs)
