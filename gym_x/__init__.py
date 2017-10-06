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
    id='AcrobotContinuousVisionX-v0',
    entry_point='gym_x.envs:AcrobotContinuousVisionEnvX'
)

# Mountain Car X
register(
    id='MountainCarContinuousX-v0',
    entry_point='gym_x.envs:MountainCarContinuousEnvX'
)

register(
    id='MountainCarContinuousVisionX-v0',
    entry_point='gym_x.envs:MountainCarContinuousVisionEnvX'
)

# Chain
register(
    id='ChainVisionX-v0',
    entry_point='gym_x.envs:ChainEnvX'
)

# register(
#     id='RoboschoolAntPlain-v0',
#     entry_point='roboschool_x.envs.maze:RoboschoolAntEnv',
# )
# register(
#     id='RoboschoolHumanoidFlagrunHarderX-v0',
#     entry_point='roboschool_x.envs:RoboschoolHumanoidFlagrunHarderX',
# )
