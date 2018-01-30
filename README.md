# Gym extension for exploration and partially observed tasks

# Exploration tasks
* AcrobotX-v0: modified gym acrobot task where reward given only at terminal state
* AcrobotContinuousX-v0: continuous acrobot
* AcrobotVisionX-v0: discrete acrobot from pixels
* AcrobotContinuousVisionX-v0: continuous acrobot from pixels
* MountainCarContinuousX-v0: continuous mountain car with reward given at terminal state
* MountainCarContinuousVisionX-v0: same as previous from pixel obs

# POMDP tasks
* TMaze
* TmazeRaceCar

# Wrappers
* TimeHorizonEnv: reset env after desired timesteps
* VisionEnv: return rendered env as obs
* PomdpCarpoleEnv: hide pole linear and angular velocity to require memory
