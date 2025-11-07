# Trossen Arm Python API Guide for Dual Robot Teleoperation

## Overview
This guide documents the Python API for the `trossen-arm` package for your **WINDOWX dual robot setup** (2 leaders + 2 followers). The Python bindings mirror the C++ API directly.

## Installation
```bash
pip install trossen-arm
```

## Core Classes and Enums

### 1. TrossenArmDriver
Main driver class for controlling the robot arms.

### 2. Model (Enum)
Robot models available:
- `trossen_arm.Model.wxai_v0` - WXAI V0
- `trossen_arm.Model.vxai_v0_right` - VXAI V0 RIGHT
- `trossen_arm.Model.vxai_v0_left` - VXAI V0 LEFT

### 3. StandardEndEffector
Pre-configured end effector types:
- `trossen_arm.StandardEndEffector.wxai_v0_leader` - Leader end effector
- `trossen_arm.StandardEndEffector.wxai_v0_follower` - Follower end effector
- `trossen_arm.StandardEndEffector.wxai_v0_base` - Base variant
- `trossen_arm.StandardEndEffector.vxai_v0_base` - VXAI base variant
- `trossen_arm.StandardEndEffector.no_gripper` - No gripper

### 4. Mode (Enum)
Joint operation modes:
- `trossen_arm.Mode.idle` - All joints braked (default safe mode)
- `trossen_arm.Mode.position` - Position control
- `trossen_arm.Mode.velocity` - Velocity control
- `trossen_arm.Mode.external_effort` - External effort/force feedback control
- `trossen_arm.Mode.effort` - Direct effort/torque control

### 5. InterpolationSpace (Enum)
- `trossen_arm.InterpolationSpace.joint` - Interpolate in joint space
- `trossen_arm.InterpolationSpace.cartesian` - Interpolate in Cartesian space

---

## Key API Methods

### Driver Initialization and Configuration

#### `__init__()`
```python
driver = trossen_arm.TrossenArmDriver()
```

#### `configure(model, end_effector, server_ip, clear_error, timeout=20.0)`
Configure and connect to a robot.
```python
driver.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_leader,
    "192.168.1.2",
    False,  # clear_error
    20.0    # timeout in seconds
)
```

**Parameters:**
- `model`: Robot model (Model enum)
- `end_effector`: End effector configuration (EndEffector struct)
- `server_ip`: IP address of the robot (string)
- `clear_error`: Whether to clear error state (bool)
- `timeout`: Connection timeout in seconds (float)

#### `cleanup(reboot_controller=False)`
Clean up driver resources.
```python
driver.cleanup(reboot_controller=False)
```

---

### Setting Modes

#### `set_all_modes(mode)`
Set all joints to the same mode.
```python
driver.set_all_modes(trossen_arm.Mode.position)
```

#### `set_arm_modes(mode)`
Set only arm joints (excluding gripper).
```python
driver.set_arm_modes(trossen_arm.Mode.position)
```

#### `set_gripper_mode(mode)`
Set gripper mode separately.
```python
driver.set_gripper_mode(trossen_arm.Mode.external_effort)
```

#### `set_joint_modes(modes)`
Set each joint individually.
```python
modes = [
    trossen_arm.Mode.position,
    trossen_arm.Mode.position,
    trossen_arm.Mode.position,
    trossen_arm.Mode.position,
    trossen_arm.Mode.position,
    trossen_arm.Mode.position,
    trossen_arm.Mode.external_effort  # gripper
]
driver.set_joint_modes(modes)
```

---

### Position Control

#### `set_all_positions(positions, goal_time=2.0, blocking=True, feedforward_velocities=None, feedforward_accelerations=None)`
Set positions for all joints.
```python
import numpy as np

# Move to home position
driver.set_all_positions(
    np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0]),
    2.0,    # goal_time: reach in 2 seconds
    True    # blocking: wait until complete
)

# Non-blocking with feedforward
driver.set_all_positions(
    positions,
    0.0,    # 0 time = immediate
    False,  # non-blocking for real-time control
    feedforward_velocities,
    feedforward_accelerations
)
```

#### `set_arm_positions(positions, goal_time=2.0, blocking=True, ...)`
Set only arm joints (6 DOF).
```python
driver.set_arm_positions(
    np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0]),
    2.0,
    True
)
```

#### `set_gripper_position(position, goal_time=2.0, blocking=True, ...)`
Set gripper position.
```python
driver.set_gripper_position(0.045, 2.0, True)  # Open gripper
driver.set_gripper_position(0.0, 2.0, True)    # Close gripper
```

#### `set_cartesian_positions(positions, interpolation_space, goal_time=2.0, blocking=True, ...)`
Set end-effector position in Cartesian space.
```python
# positions = [x, y, z, rx, ry, rz]
# First 3: translation in meters
# Last 3: angle-axis rotation in radians
cartesian_pos = driver.get_cartesian_positions()
cartesian_pos[2] += 0.1  # Move up 10cm

driver.set_cartesian_positions(
    cartesian_pos,
    trossen_arm.InterpolationSpace.cartesian
)
```

---

### External Effort Control (Force Feedback)

#### `set_all_external_efforts(external_efforts, goal_time=0.0, blocking=False)`
Set external efforts for force feedback.
```python
# Get efforts from follower
follower_efforts = np.array(driver_follower.get_all_external_efforts())

# Apply to leader with gain
force_feedback_gain = 0.1
driver_leader.set_all_external_efforts(
    -force_feedback_gain * follower_efforts,
    0.0,
    False
)
```

#### `set_gripper_external_effort(external_effort, goal_time=5.0, blocking=True)`
Control gripper with force feedback.
```python
driver.set_gripper_external_effort(20.0, 5.0, True)   # Open with force
driver.set_gripper_external_effort(-20.0, 5.0, True)  # Close with force
```

---

### Velocity Control

#### `set_all_velocities(velocities, goal_time=2.0, blocking=True, ...)`
```python
velocities = np.array([0.1, 0.1, 0.1, 0.0, 0.0, 0.0, 0.0])
driver.set_all_velocities(velocities, 2.0, True)
```

---

### Getting State Information

#### Position Getters
```python
# All joints (7 DOF including gripper)
positions = driver.get_all_positions()  # Returns list

# Arm only (6 DOF)
arm_positions = driver.get_arm_positions()

# Gripper only
gripper_pos = driver.get_gripper_position()

# Individual joint
joint_pos = driver.get_joint_position(0)  # Joint index 0-6

# Cartesian
cartesian_pos = driver.get_cartesian_positions()  # Returns array[6]
```

#### Velocity Getters
```python
velocities = driver.get_all_velocities()
arm_velocities = driver.get_arm_velocities()
gripper_vel = driver.get_gripper_velocity()
joint_vel = driver.get_joint_velocity(0)
cartesian_vel = driver.get_cartesian_velocities()
```

#### Effort Getters
```python
efforts = driver.get_all_efforts()
external_efforts = driver.get_all_external_efforts()  # For force feedback
compensation_efforts = driver.get_all_compensation_efforts()
```

#### Temperature Monitoring
```python
rotor_temps = driver.get_all_rotor_temperatures()
driver_temps = driver.get_all_driver_temperatures()
```

#### Other Info
```python
num_joints = driver.get_num_joints()  # Returns 7 for WXAI
driver_version = driver.get_driver_version()
controller_version = driver.get_controller_version()
is_configured = driver.get_is_configured()
```

---

## Dual Robot Teleoperation Example (2 Leader + 2 Follower)

```python
import time
import numpy as np
import trossen_arm

# IP addresses - adjust for your setup
LEADER_1_IP = '192.168.1.2'
LEADER_2_IP = '192.168.1.3'
FOLLOWER_1_IP = '192.168.1.4'
FOLLOWER_2_IP = '192.168.1.5'

# Teleoperation parameters
TELEOPERATION_TIME = 60  # seconds
FORCE_FEEDBACK_GAIN = 0.1

# Initialize all 4 drivers
print("Initializing drivers...")
driver_leader_1 = trossen_arm.TrossenArmDriver()
driver_leader_2 = trossen_arm.TrossenArmDriver()
driver_follower_1 = trossen_arm.TrossenArmDriver()
driver_follower_2 = trossen_arm.TrossenArmDriver()

# Configure all robots
print("Configuring robots...")
driver_leader_1.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_leader,
    LEADER_1_IP,
    False
)
driver_leader_2.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_leader,
    LEADER_2_IP,
    False
)
driver_follower_1.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_follower,
    FOLLOWER_1_IP,
    False
)
driver_follower_2.configure(
    trossen_arm.Model.wxai_v0,
    trossen_arm.StandardEndEffector.wxai_v0_follower,
    FOLLOWER_2_IP,
    False
)

# Move all to home positions
print("Moving to home positions...")
home_positions = np.array([0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0])

for driver in [driver_leader_1, driver_leader_2, driver_follower_1, driver_follower_2]:
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(home_positions, 2.0, True)

# Start teleoperation
print("Starting teleoperation...")
time.sleep(1)

# Set leaders to external effort mode (force feedback)
driver_leader_1.set_all_modes(trossen_arm.Mode.external_effort)
driver_leader_2.set_all_modes(trossen_arm.Mode.external_effort)

# Set followers to position mode (follow leaders)
driver_follower_1.set_all_modes(trossen_arm.Mode.position)
driver_follower_2.set_all_modes(trossen_arm.Mode.position)

# Teleoperation loop
start_time = time.time()
end_time = start_time + TELEOPERATION_TIME

while time.time() < end_time:
    # Leader 1 -> Follower 1
    driver_leader_1.set_all_external_efforts(
        -FORCE_FEEDBACK_GAIN * np.array(driver_follower_1.get_all_external_efforts()),
        0.0,
        False
    )
    driver_follower_1.set_all_positions(
        driver_leader_1.get_all_positions(),
        0.0,
        False,
        driver_leader_1.get_all_velocities()
    )
    
    # Leader 2 -> Follower 2
    driver_leader_2.set_all_external_efforts(
        -FORCE_FEEDBACK_GAIN * np.array(driver_follower_2.get_all_external_efforts()),
        0.0,
        False
    )
    driver_follower_2.set_all_positions(
        driver_leader_2.get_all_positions(),
        0.0,
        False,
        driver_leader_2.get_all_velocities()
    )

print("Teleoperation complete!")

# Return to home
print("Returning to home positions...")
for driver in [driver_leader_1, driver_leader_2, driver_follower_1, driver_follower_2]:
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(home_positions, 2.0, True)

# Move to sleep positions
print("Moving to sleep positions...")
sleep_positions = np.zeros(7)
for driver in [driver_leader_1, driver_leader_2, driver_follower_1, driver_follower_2]:
    driver.set_all_positions(sleep_positions, 2.0, True)

print("Done!")
```

---

## Configuration Management

### Save/Load Configurations
```python
# Save all configurations to YAML
driver.save_configs_to_file("robot_config.yaml")

# Load configurations from YAML
driver.load_configs_from_file("robot_config.yaml")
```

### Joint Characteristics
```python
# Get current characteristics
characteristics = driver.get_joint_characteristics()

# Modify and set
for char in characteristics:
    char.effort_correction = 1.2
    char.friction_constant_term = 0.5
    
driver.set_joint_characteristics(characteristics)
```

### End Effector Configuration
```python
end_effector = driver.get_end_effector()

# Modify tool frame (e.g., custom gripper offset)
end_effector.t_flange_tool[0] = 0.15  # X offset in meters
end_effector.t_flange_tool[1] = 0.0   # Y offset
end_effector.t_flange_tool[2] = 0.0   # Z offset

driver.set_end_effector(end_effector)
```

### Joint Limits
```python
limits = driver.get_joint_limits()

# Modify limits
limits[0].position_min = -3.14
limits[0].position_max = 3.14
limits[0].velocity_max = 2.0
limits[0].effort_max = 10.0

driver.set_joint_limits(limits)
```

---

## Error Handling and Recovery

```python
try:
    driver.configure(...)
except trossen_arm.RuntimeError as e:
    print(f"Runtime error: {e}")
except trossen_arm.LogicError as e:
    print(f"Logic error: {e}")

# Get error information
error_info = driver.get_error_information()
print(error_info)

# Clear error state
driver.configure(..., clear_error=True, ...)
```

---

## Logging

```python
import logging

# Get the logger
logger_name = trossen_arm.TrossenArmDriver.get_default_logger_name()
logger = logging.getLogger(logger_name)

# Or for specific robot
logger_name = trossen_arm.TrossenArmDriver.get_logger_name(
    trossen_arm.Model.wxai_v0,
    "192.168.1.2"
)
logger = logging.getLogger(logger_name)

# Configure logging
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
logger.addHandler(handler)
```

---

## Important Notes

### 1. **Real-time Control**
For teleoperation, use:
- `goal_time=0.0` for immediate execution
- `blocking=False` to avoid waiting
- Feed positions and velocities together for smooth motion

### 2. **Force Feedback**
- Use `Mode.external_effort` on leaders
- Apply negative gain: `-force_feedback_gain * follower_efforts`
- Typical gain: 0.05 to 0.2

### 3. **Thread Safety**
The driver has internal mutexes. You can safely call methods from one thread at a time, but the driver manages a daemon thread internally.

### 4. **Cleanup**
Always cleanup properly:
```python
try:
    # Your code
    pass
finally:
    driver.cleanup()
```

Or rely on destructor (automatic when driver goes out of scope).

### 5. **Units**
- **Positions**: radians (arm joints), meters (gripper)
- **Velocities**: rad/s (arm), m/s (gripper)
- **Efforts**: Nm (arm), N (gripper)
- **Cartesian positions**: [x, y, z] in meters, [rx, ry, rz] angle-axis in radians
- **Cartesian velocities**: [vx, vy, vz] in m/s, [wx, wy, wz] in rad/s

### 6. **Number of Joints**
- WXAI V0: 7 joints (6 arm + 1 gripper)
- Indexing: 0-5 are arm joints, 6 is gripper

---

## Advanced: Trajectory Generation with Feedforward

```python
from scipy.interpolate import PchipInterpolator
import time
import numpy as np

# Define waypoints
waypoints = np.array([
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Start
    [0.0, np.pi/2, np.pi/2, 0.0, 0.0, 0.0, 0.0],  # Home
    [0.5, np.pi/2, np.pi/2, 0.3, 0.0, 0.0, 0.0],  # Mid
])
timepoints = np.array([0, 2, 4])

# Create interpolators
pchip_pos = PchipInterpolator(timepoints, waypoints, axis=0)
pchip_vel = pchip_pos.derivative()
pchip_acc = pchip_vel.derivative()

# Execute trajectory
driver.set_all_modes(trossen_arm.Mode.position)

start_time = time.time()
end_time = start_time + timepoints[-1]

while time.time() < end_time:
    current_time = time.time() - start_time
    
    pos = pchip_pos(current_time)
    vel = pchip_vel(current_time)
    acc = pchip_acc(current_time)
    
    driver.set_all_positions(pos, 0.0, False, vel, acc)
```

---

## Quick Reference: Common Patterns

### Pattern 1: Simple Move
```python
driver.set_all_modes(trossen_arm.Mode.position)
driver.set_all_positions(target_positions, 2.0, True)
```

### Pattern 2: Real-time Teleoperation
```python
# Setup
leader.set_all_modes(trossen_arm.Mode.external_effort)
follower.set_all_modes(trossen_arm.Mode.position)

# Loop
while running:
    follower.set_all_positions(
        leader.get_all_positions(),
        0.0, False,
        leader.get_all_velocities()
    )
    leader.set_all_external_efforts(
        -gain * np.array(follower.get_all_external_efforts()),
        0.0, False
    )
```

### Pattern 3: Gripper Control
```python
# Force-based
driver.set_gripper_mode(trossen_arm.Mode.external_effort)
driver.set_gripper_external_effort(20.0, 2.0, True)  # Open

# Position-based
driver.set_gripper_mode(trossen_arm.Mode.position)
driver.set_gripper_position(0.045, 2.0, True)  # Open
```

---

This guide should give you everything you need for your dual robot teleoperation setup! The Python API directly mirrors the C++ API, so you can reference the C++ demos for additional patterns.
