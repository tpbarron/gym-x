from gym.envs.registration import register

# Acrobot X
register(
    id='AcrobotX-v0',
    entry_point='gym_x.envs.exploration:AcrobotEnvX'
)
register(
    id='AcrobotContinuousX-v0',
    entry_point='gym_x.envs.exploration:AcrobotContinuousEnvX'
)
register(
    id='AcrobotVisionX-v0',
    entry_point='gym_x.envs.exploration:AcrobotVisionEnvX'
)
register(
    id='AcrobotContinuousVisionX-v0',
    entry_point='gym_x.envs.exploration:AcrobotContinuousVisionEnvX'
)

# Mountain Car X
register(
    id='MountainCarContinuousX-v0',
    entry_point='gym_x.envs.exploration:MountainCarContinuousEnvX'
)

register(
    id='MountainCarContinuousVisionX-v0',
    entry_point='gym_x.envs.exploration:MountainCarContinuousVisionEnvX'
)

# Chain
register(
    id='ChainVisionX-v0',
    entry_point='gym_x.envs.exploration:ChainVisionEnvX'
)

register(
    id='ChainX-v0',
    entry_point='gym_x.envs.exploration:ChainEnvX'
)

register(
    id='AntBulletX-v0',
    entry_point='gym_x.envs.exploration:AntBulletEnvX'
)

# Walker2d
register(
    id='Walker2DBulletX-v0',
    entry_point='gym_x.envs.exploration:Walker2DBulletEnvX'
)

register(
    id='Walker2DVisionBulletX-v0',
    entry_point='gym_x.envs.exploration:Walker2DVisionBulletEnvX'
)

# Half Cheetah
register(
    id='HalfCheetahBulletX-v0',
    entry_point='gym_x.envs.exploration:HalfCheetahBulletEnvX'
)

register(
    id='HalfCheetahVisionBulletX-v0',
    entry_point='gym_x.envs.exploration:HalfCheetahVisionBulletEnvX'
)

# Inverted Pendulum
register(
    id='InvertedPendulumSwingupVisionBulletEnv-v0',
    entry_point='gym_x.envs.exploration:InvertedPendulumSwingupVisionBulletEnv'
)

register(
    id='InvertedPendulumSwingupBulletX-v0',
    entry_point='gym_x.envs.exploration:InvertedPendulumSwingupBulletEnvX'
)

register(
    id='InvertedPendulumSwingupVisionBulletX-v0',
    entry_point='gym_x.envs.exploration:InvertedPendulumSwingupVisionBulletEnvX'
)

# hopper
register(
    id='HopperBulletX-v0',
    entry_point='gym_x.envs.exploration:HopperBulletEnvX'
)

register(
    id='HopperVisionBulletX-v0',
    entry_point='gym_x.envs.exploration:HopperVisionBulletEnvX'
)
