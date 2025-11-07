#!/usr/bin/env python3
"""Visual test for Trossen bimanual environment - displays environment info."""

import gym
import numpy as np
import trossen_sim

def visual_test():
    print("=" * 70)
    print("TROSSEN BIMANUAL SIMULATION - ENVIRONMENT TEST")
    print("=" * 70)
    
    env = gym.make('TrossenBimanualPickPlace-v0')
    
    print("\n📋 ENVIRONMENT SPECIFICATIONS")
    print("-" * 70)
    print(f"Action Space: {env.action_space}")
    print(f"  → Shape: {env.action_space.shape}")
    print(f"  → Range: [{env.action_space.low[0]}, {env.action_space.high[0]}]")
    print(f"\nObservation Space: Dict with 'state' key")
    for key, space in env.observation_space['state'].spaces.items():
        print(f"  → {key:20s}: {space}")
    
    print("\n🔄 RESET ENVIRONMENT")
    print("-" * 70)
    obs, info = env.reset(seed=42)
    
    print("Initial State:")
    print(f"  Left Arm Joint Pos:  {obs['state']['left/joint_pos']}")
    print(f"  Right Arm Joint Pos: {obs['state']['right/joint_pos']}")
    print(f"  Left TCP (xyz):      {obs['state']['left/tcp_pos']}")
    print(f"  Right TCP (xyz):     {obs['state']['right/tcp_pos']}")
    print(f"  Left Gripper:        {obs['state']['left/gripper_pos'][0]:.4f}")
    print(f"  Right Gripper:       {obs['state']['right/gripper_pos'][0]:.4f}")
    print(f"  Cube Position (xyz): {obs['state']['cube_pos']}")
    
    print("\n🎮 RUNNING SIMULATION (50 random steps)")
    print("-" * 70)
    
    rewards = []
    cube_heights = []
    
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        
        rewards.append(reward)
        cube_heights.append(obs['state']['cube_pos'][2])
        
        if step % 10 == 0:
            print(f"Step {step:3d}: reward={reward:.4f}, cube_z={obs['state']['cube_pos'][2]:.4f}")
        
        if terminated or truncated:
            print(f"\n  Episode ended at step {step}")
            break
    
    print("\n📊 STATISTICS")
    print("-" * 70)
    print(f"Total Steps:        {len(rewards)}")
    print(f"Average Reward:     {np.mean(rewards):.6f}")
    print(f"Max Reward:         {np.max(rewards):.6f}")
    print(f"Min Cube Height:    {np.min(cube_heights):.6f}")
    print(f"Max Cube Height:    {np.max(cube_heights):.6f}")
    print(f"Cube Moved (z):     {np.max(cube_heights) - np.min(cube_heights):.6f}")
    
    env.close()
    
    print("\n✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print("\nThe environment is ready for SERL training!")
    print("Next step: Run async_sac_state_sim.py with --env TrossenBimanualPickPlace-v0")
    print("=" * 70)

if __name__ == "__main__":
    visual_test()
