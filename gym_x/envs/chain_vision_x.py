"""
Linear chain env in pybullet where reward is given for reaching end of path
"""

import os
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

#TODO: update pybullet
class ChainEnvX(AntBulletEnv):

    def __init__(self,
                 render_dims=(84, 84)):
        super(ChainEnvX, self).__init__()

        self.render_dims = render_dims
        # The observation is a combination of joints and image
        self.observation_space = spaces.Tuple((AntBulletEnv.observation_space,
                                               spaces.Box(low=0., high=1., shape=render_dims)))

    def get_render_obs(self):
        """
        Compute first-person view from robot

        https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/testrender_np.py
        """
        # TODO: fix me to be along moving axis
        viewMatrix = pybullet.computeViewMatrixFromYawPitchRoll(camTargetPos, camDistance, yaw, pitch, roll, upAxisIndex)
        aspect = pixelWidth / pixelHeight;
        projectionMatrix = pybullet.computeProjectionMatrixFOV(fov, aspect, nearPlane, farPlane);
        img_arr = pybullet.getCameraImage(pixelWidth, pixelHeight, viewMatrix,projectionMatrix, [0,1,0],renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
        stop = time.time()
        print ("renderImage %f" % (stop - start))

        w=img_arr[0] #width of the image, in pixels
        h=img_arr[1] #height of the image, in pixels
        rgb=img_arr[2] #color data RGB
        dep=img_arr[3] #depth data

    def _step(self, action):
        obs, rew, done, info = AntBulletEnv._step(self, action)
        # TODO: concat observation with render in tuple
        # TODO: specify reward = 1 iff at terminal
        # TODO: decide whether to cut out movement costs as well?
        if self.robot.body_xyz[0] > 4:
            done = True
        if done:
            rew += 1
        return obs, rew, done, info

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
        break

    env.reset()
