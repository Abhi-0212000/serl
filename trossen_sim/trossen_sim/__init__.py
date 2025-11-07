# Trossen Sim - MuJoCo simulation for Trossen WidowX AI

"""
Trossen Sim Package

A simulation environment for Trossen bimanual robotic arms using MuJoCo.
"""

import gym
from gym.envs.registration import register

from trossen_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__version__ = "0.0.1"

__all__ = [
    "__version__",
    "MujocoGymEnv",
    "GymRenderingSpec",
]

# Register bimanual pick and place environment (state-based)
register(
    id="TrossenBimanualPickPlace-v0",
    entry_point="trossen_sim.envs.trossen_bimanual_gym_env:TrossenBimanualPickPlaceGymEnv",
    max_episode_steps=100,
)

# Register bimanual pick and place environment (vision-based)
register(
    id="TrossenBimanualPickPlaceVision-v0",
    entry_point="trossen_sim.envs.trossen_bimanual_gym_env:TrossenBimanualPickPlaceGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},
)
