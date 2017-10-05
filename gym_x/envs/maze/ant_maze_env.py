from maze_env import MazeEnv
from roboschool.gym_mujoco_walkers import RoboschoolAnt


class AntMazeEnv(RoboschoolAnt):

    MODEL_CLASS = RoboschoolAnt
    ORI_IND = 2

    MAZE_HEIGHT = 0.5
    MAZE_SIZE_SCALING = 4
    MAZE_MAKE_CONTACTS = True
