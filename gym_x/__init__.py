from gym.envs.registration import register

# Acrobot X
register(
    id='AcrobotX-v0',
    entry_point='gym_x.envs:AcrobotEnvX'
)
register(
    id='AcrobotContinuousX-v0',
    entry_point='gym_x.envs:AcrobotContinuousEnvX'
)
register(
    id='AcrobotVisionX-v0',
    entry_point='gym_x.envs:AcrobotVisionEnvX'
)
register(
    id='AcrobotVisionContinuousX-v0',
    entry_point='gym_x.envs:AcrobotVisionContinuousEnvX'
)

# Mountain Car X
register(
    id='MountainCarContinuousX-v0',
    entry_point='gym_x.envs:MountainCarContinuousEnvX'
)

# register(
#     id='RoboschoolAntPlain-v0',
#     entry_point='roboschool_x.envs.maze:RoboschoolAntEnv',
# )
# register(
#     id='RoboschoolHumanoidFlagrunHarderX-v0',
#     entry_point='roboschool_x.envs:RoboschoolHumanoidFlagrunHarderX',
# )
