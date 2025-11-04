# Trossen Sim

MuJoCo simulation environment for the Trossen WidowX AI robotic arm.

## Installation

```bash
cd trossen_sim
pip install -e .
```

## Structure

- `trossen_sim/envs/`: Gym environment implementations
  - `trossen_pick_gym_env.py`: Pick-and-place task environment
  - `xmls/`: MuJoCo XML files for the robot and scene
- `trossen_sim/controllers/`: Control algorithms
  - `opspace.py`: Operational space controller
- `trossen_sim/test/`: Test scripts for validation

## Robot Specifications

- **Robot**: Trossen Robotics WidowX AI
- **DOF**: 6 (revolute joints)
- **Gripper**: 2-finger parallel gripper (prismatic)
- **Workspace**: ~0.5m radius

## Usage

```python
import gymnasium as gym
import trossen_sim

env = gym.make("TrossenPickCube-v0", render_mode="human")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```
