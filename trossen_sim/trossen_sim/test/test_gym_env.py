"""
Test script to verify the Trossen MuJoCo environment loads and runs correctly.
"""

import numpy as np
import gymnasium as gym
import trossen_sim

def test_environment_loading():
    """Test that the environment can be loaded."""
    print("Testing environment loading...")
    try:
        env = gym.make("TrossenPickCube-v0", render_mode="rgb_array")
        print("✓ Environment loaded successfully")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Failed to load environment: {e}")
        return False

def test_reset():
    """Test that reset works correctly."""
    print("\nTesting reset...")
    try:
        env = gym.make("TrossenPickCube-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        assert "state" in obs, "Observation missing 'state' key"
        assert "trossen/tcp_pos" in obs["state"], "State missing tcp_pos"
        assert "trossen/tcp_vel" in obs["state"], "State missing tcp_vel"
        assert "trossen/gripper_pos" in obs["state"], "State missing gripper_pos"
        assert "block_pos" in obs["state"], "State missing block_pos"
        
        print("✓ Reset works correctly")
        print(f"  TCP position: {obs['state']['trossen/tcp_pos']}")
        print(f"  Block position: {obs['state']['block_pos']}")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Reset failed: {e}")
        return False

def test_step():
    """Test that step works correctly."""
    print("\nTesting step...")
    try:
        env = gym.make("TrossenPickCube-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        # Take a few random actions
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"  Step {i+1}: reward={reward:.4f}, terminated={terminated}")
        
        print("✓ Step works correctly")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Step failed: {e}")
        return False

def test_rendering():
    """Test that rendering works."""
    print("\nTesting rendering...")
    try:
        env = gym.make("TrossenPickCube-v0", render_mode="rgb_array")
        obs, info = env.reset()
        
        img = env.render()
        assert img is not None, "Render returned None"
        assert img.shape[2] == 3, f"Expected RGB image, got shape {img.shape}"
        
        print(f"✓ Rendering works correctly (image shape: {img.shape})")
        env.close()
        return True
    except Exception as e:
        print(f"✗ Rendering failed: {e}")
        return False

def test_image_observations():
    """Test image observation mode."""
    print("\nTesting image observations...")
    try:
        env = gym.make("TrossenPickCube-v0", render_mode="rgb_array", image_obs=True)
        obs, info = env.reset()
        
        assert "images" in obs, "Observation missing 'images' key"
        assert "front" in obs["images"], "Images missing 'front' camera"
        
        img = obs["images"]["front"]
        print(f"✓ Image observations work (shape: {img.shape})")
        
        env.close()
        return True
    except Exception as e:
        print(f"✗ Image observations failed: {e}")
        return False

def main():
    print("=" * 60)
    print("Trossen Sim Environment Test Suite")
    print("=" * 60)
    
    tests = [
        test_environment_loading,
        test_reset,
        test_step,
        test_rendering,
        test_image_observations,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Test Results: {sum(results)}/{len(results)} passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed! The environment is ready to use.")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit(main())
