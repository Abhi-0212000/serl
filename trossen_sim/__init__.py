# Trossen Sim - MuJoCo simulation for Trossen WidowX AI

print("DEBUG: Loading trossen_sim/__init__.py")

import gym
from gym.envs.registration import register

from trossen_sim.mujoco_gym_env import GymRenderingSpec, MujocoGymEnv

__all__ = [
    "MujocoGymEnv",
    "GymRenderingSpec",
]

print("DEBUG: About to register environment")

# Register bimanual transfer cube environment
register(
    id="TrossenBimanualTransferCube-v0",
    entry_point="trossen_sim.envs.trossen_bimanual_gym_env:TrossenBimanualGymEnv",
    max_episode_steps=1000,
    kwargs={
        "image_obs": True,
        "action_scale": 0.05,
        "control_dt": 0.02,
        "physics_dt": 0.002,
        "time_limit": 20.0,
        "render_spec": GymRenderingSpec(
            height=128,
            width=128,
            camera_id="cam_high",
        ),
    },
)

print("DEBUG: Environment registered successfully")
