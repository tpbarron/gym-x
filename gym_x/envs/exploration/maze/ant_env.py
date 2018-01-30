import numpy as np
import os
import roboschool
from roboschool.gym_mujoco_xml_env import RoboschoolMujocoXmlEnv
from roboschool_x.envs.maze.gym_walker import RoboschoolWalker
from roboschool.scene_stadium import SinglePlayerStadiumScene
from roboschool.scene_abstract import cpp_household

class RoboschoolWalkerMujocoXmlEnv(RoboschoolWalker, RoboschoolMujocoXmlEnv):

    def __init__(self, fn, robot_name, action_dim, obs_dim, power):
        RoboschoolMujocoXmlEnv.__init__(self, fn, robot_name, action_dim, obs_dim)
        RoboschoolWalker.__init__(self, power)

        # pass
    # def robot_specific_reset(self):
        # RoboschoolWalker.robot_specific_reset(self)
        # RoboschoolMujocoXmlEnv.robot_specific_reset(self)

class RoboschoolAntEnv(RoboschoolWalkerMujocoXmlEnv):

    foot_list = ['front_left_foot', 'front_right_foot', 'left_back_foot', 'right_back_foot']

    def __init__(self):
        RoboschoolWalkerMujocoXmlEnv.__init__(self, "ant.xml", "torso", action_dim=8, obs_dim=28, power=2.5)

    def alive_bonus(self, z, pitch):
        return 0.
        # return +1 if z > 0.26 else -1  # 0.25 is central sphere rad, die if it scrapes the ground

    def _reset(self):
        print ("Resetting")
        obs = RoboschoolWalkerMujocoXmlEnv._reset(self)
        self.camera_fpv = self.scene.cpp_world.new_camera_free_float(10, 10, "first_person_cam")
        return obs

    def _step(self, action):
        RoboschoolWalkerMujocoXmlEnv._step(self, action)
        # self.camera_adjust()
        self.camera_fpv.move_and_look_at(1,1,1, 0,0,0)

        print ("Post adjust")
        rgb, _, _, _, _ = self.camera_fpv.render(False, False, False) # render_depth, render_labeling, print_timing)
        print ("Post render")
        rendered_rgb = np.fromstring(rgb, dtype=np.uint8).reshape( (self.VIDEO_H,self.VIDEO_W,3) )
        from PIL import Image
        I = Image.fromarray(rendered_rgb)
        I.show()
        input("")
        return rendered_rgb

    def robot_specific_reset(self):
        RoboschoolWalkerMujocoXmlEnv.robot_specific_reset(self)
        cpose = cpp_household.Pose()
        cpose.set_rpy(0, 0, 0)
        cpose.set_xyz(-1.5, 1.0, 0.25)
        self.box = self.scene.cpp_world.load_urdf(os.path.join(os.path.dirname(__file__), "assets/cube.urdf"), cpose, True, False)

    def create_single_player_scene(self):
        scene = SinglePlayerStadiumScene(gravity=9.8, timestep=0.0165/4, frame_skip=4)
        scene.zero_at_running_strip_start_line = False
        return scene
