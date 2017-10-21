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
    entry_point='gym_x.envs:ChainVisionEnvX'
)

register(
    id='ChainX-v0',
    entry_point='gym_x.envs:ChainEnvX'
)

register(
    id='AntBulletX-v0',
    entry_point='gym_x.envs:AntBulletEnvX'
)

# Walker2d
register(
    id='Walker2DBulletX-v0',
    entry_point='gym_x.envs:Walker2DBulletEnvX'
)

register(
    id='Walker2DVisionBulletX-v0',
    entry_point='gym_x.envs:Walker2DVisionBulletEnvX'
)

# Half Cheetah
register(
    id='HalfCheetahBulletX-v0',
    entry_point='gym_x.envs:HalfCheetahBulletEnvX'
)

register(
    id='HalfCheetahVisionBulletX-v0',
    entry_point='gym_x.envs:HalfCheetahVisionBulletEnvX'
)

# Inverted Pendulum
register(
    id='InvertedPendulumSwingupVisionBulletEnv-v0',
    entry_point='gym_x.envs:InvertedPendulumSwingupVisionBulletEnv'
)

register(
    id='InvertedPendulumSwingupBulletX-v0',
    entry_point='gym_x.envs:InvertedPendulumSwingupBulletEnvX'
)

register(
    id='InvertedPendulumSwingupVisionBulletX-v0',
    entry_point='gym_x.envs:InvertedPendulumSwingupVisionBulletEnvX'
)

# hopper
register(
    id='HopperBulletX-v0',
    entry_point='gym_x.envs:HopperBulletEnvX'
)

register(
    id='HopperVisionBulletX-v0',
    entry_point='gym_x.envs:HopperVisionBulletEnvX'
)
