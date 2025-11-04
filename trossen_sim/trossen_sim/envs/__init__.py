import gymnasium as gym

gym.register(
    id="TrossenPickCube-v0",
    entry_point="trossen_sim.envs.trossen_pick_gym_env:TrossenPickCubeGymEnv",
    max_episode_steps=100,
)
