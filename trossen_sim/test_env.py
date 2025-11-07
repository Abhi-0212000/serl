#!/usr/bin/env python3
"""Test script for Trossen bimanual environment."""

import gym
import trossen_sim

def test_env():
    print("Creating TrossenBimanualPickPlace-v0 environment...")
    env = gym.make('TrossenBimanualPickPlace-v0')
    
    print("✓ Environment created successfully!")
    print(f"Action space: {env.action_space}")
    print(f"Observation space keys: {env.observation_space.keys()}")
    
    print("\nResetting environment...")
    obs, info = env.reset()
    print(f"✓ Reset successful!")
    print(f"Observation state keys: {obs['state'].keys()}")
    print(f"Cube position: {obs['state']['cube_pos']}")
    print(f"Left TCP position: {obs['state']['left/tcp_pos']}")
    print(f"Right TCP position: {obs['state']['right/tcp_pos']}")
    
    print("\nTaking random actions...")
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: reward={reward:.4f}, cube_z={obs['state']['cube_pos'][2]:.4f}")
        
        if terminated or truncated:
            print("Episode ended, resetting...")
            obs, info = env.reset()
            break
    
    env.close()
    print("\n✓ All tests passed!")

if __name__ == "__main__":
    test_env()
