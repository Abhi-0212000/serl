"""Utility functions for Trossen bimanual environments."""

import numpy as np


def sample_box_pose() -> np.ndarray:
    """
    Generate a random pose for a cube within predefined position ranges.
    
    Returns:
        np.ndarray: A 7D array [x, y, z, qw, qx, qy, qz] representing 
                    the cube's position and orientation as a quaternion.
    """
    # Position ranges (start near left arm workspace)
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.0125, 0.0125]  # Fixed height (sitting on table)
    
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    
    # Identity quaternion (no rotation)
    cube_quat = np.array([1, 0, 0, 0])  # [w, x, y, z]
    
    return np.concatenate([cube_position, cube_quat])


def denormalize_action(action: np.ndarray, joint_limits: np.ndarray) -> np.ndarray:
    """
    Denormalize actions from [-1, 1] to actual joint limits.
    
    Args:
        action: Normalized action in [-1, 1]
        joint_limits: Array of shape (n_joints, 2) with [min, max] for each joint
        
    Returns:
        np.ndarray: Denormalized joint positions
    """
    action_clipped = np.clip(action, -1.0, 1.0)
    joint_min = joint_limits[:, 0]
    joint_max = joint_limits[:, 1]
    
    # Map from [-1, 1] to [joint_min, joint_max]
    joint_positions = (action_clipped + 1.0) / 2.0 * (joint_max - joint_min) + joint_min
    
    return joint_positions


def compute_distance_reward(pos1: np.ndarray, pos2: np.ndarray, temperature: float = 20.0) -> float:
    """
    Compute exponential distance-based reward.
    
    Args:
        pos1: Position 1 (e.g., gripper)
        pos2: Position 2 (e.g., target)
        temperature: Exponential decay rate
        
    Returns:
        float: Reward in [0, 1]
    """
    dist = np.linalg.norm(pos1 - pos2)
    return np.exp(-temperature * dist)
