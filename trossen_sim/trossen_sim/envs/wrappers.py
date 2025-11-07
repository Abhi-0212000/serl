"""Gym wrappers for Trossen environments."""

import gym
import numpy as np
from typing import Dict, Any, Tuple


class FlattenStateWrapper(gym.ObservationWrapper):
    """
    Wrapper to flatten nested Dict observation space into a single flat array.
    
    Converts observations like:
        {"state": {"joint_pos": [...], "tcp_pos": [...], ...}}
    Into:
        np.array([...])  # flat concatenated array
        
    This is needed for SAC agents that expect flat observation spaces.
    """
    
    def __init__(self, env):
        super().__init__(env)
        
        if not isinstance(env.observation_space, gym.spaces.Dict):
            raise ValueError("FlattenStateWrapper requires a Dict observation space")
        
        if "state" not in env.observation_space.spaces:
            raise ValueError("FlattenStateWrapper requires a 'state' key in observation_space")
        
        # Calculate flattened size
        state_space = env.observation_space["state"]
        flat_size = self._calculate_flat_size(state_space)
        
        # Create flattened observation space
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(flat_size,),
            dtype=np.float32,
        )
        
        # Store keys for consistent ordering
        self._state_keys = self._get_sorted_keys(state_space)
    
    def _calculate_flat_size(self, space: gym.spaces.Dict) -> int:
        """Calculate total size of flattened space."""
        total = 0
        for key in sorted(space.spaces.keys()):
            subspace = space[key]
            if isinstance(subspace, gym.spaces.Box):
                total += int(np.prod(subspace.shape))
            elif isinstance(subspace, gym.spaces.Dict):
                total += self._calculate_flat_size(subspace)
        return total
    
    def _get_sorted_keys(self, space: gym.spaces.Dict) -> list:
        """Get sorted keys for consistent ordering."""
        keys = []
        for key in sorted(space.spaces.keys()):
            subspace = space[key]
            if isinstance(subspace, gym.spaces.Dict):
                # For nested dicts, create tuples like ("left", "joint_pos")
                for subkey in sorted(subspace.spaces.keys()):
                    keys.append((key, subkey))
            else:
                keys.append(key)
        return keys
    
    def _flatten_dict(self, obs_dict: Dict, prefix: str = "") -> np.ndarray:
        """Recursively flatten a nested dict observation."""
        arrays = []
        
        for key in sorted(obs_dict.keys()):
            value = obs_dict[key]
            if isinstance(value, dict):
                # Recursively flatten nested dict
                arrays.append(self._flatten_dict(value, prefix=f"{prefix}{key}/"))
            elif isinstance(value, np.ndarray):
                arrays.append(value.flatten())
            else:
                arrays.append(np.array([value], dtype=np.float32))
        
        return np.concatenate(arrays)
    
    def observation(self, observation: Dict) -> np.ndarray:
        """
        Flatten the observation dict.
        
        Args:
            observation: Dict with "state" key containing nested state dict
            
        Returns:
            Flattened numpy array
        """
        state_dict = observation["state"]
        return self._flatten_dict(state_dict)


class FlattenActionWrapper(gym.ActionWrapper):
    """
    Wrapper to handle action scaling/clipping.
    
    Not currently needed since TrossenBimanualPickPlaceGymEnv
    already handles action normalization, but kept for future use.
    """
    
    def __init__(self, env):
        super().__init__(env)
    
    def action(self, action: np.ndarray) -> np.ndarray:
        """Pass through action (already normalized)."""
        return np.clip(action, -1.0, 1.0)
