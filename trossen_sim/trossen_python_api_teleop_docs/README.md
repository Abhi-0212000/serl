# WINDOWX Dual Robot Teleoperation Setup

Python-based teleoperation system for WINDOWX dual robot setup (2 leaders + 2 followers).

## Overview

This repository contains Python scripts for bilateral teleoperation with force feedback using Trossen robotic arms. The setup consists of:
- **2 Leader robots**: Provide force feedback to the operator
- **2 Follower robots**: Mirror the leader movements in real-time

## Files

- **`PYTHON_API_GUIDE.md`**: Complete Python API documentation for `trossen-arm` package
- **`dual_robot_teleop.py`**: Main teleoperation script for 2 leader + 2 follower setup
- **`test_single_robot.py`**: Test script to verify individual robot connections
- **`trossen_arm/`**: Reference C++ library and demos (cloned from Trossen Robotics)

## Installation

### 1. Install the Python Package

```bash
pip install trossen-arm
```

### 2. Install Additional Dependencies

```bash
pip install numpy scipy
```

### 3. Make Scripts Executable (Optional)

```bash
chmod +x dual_robot_teleop.py
chmod +x test_single_robot.py
```

## Quick Start

### Step 1: Test Individual Robots

Before running the full teleoperation, test each robot individually:

```bash
# Test Leader 1
python test_single_robot.py --ip 192.168.1.2 --type leader

# Test Leader 2
python test_single_robot.py --ip 192.168.1.3 --type leader

# Test Follower 1
python test_single_robot.py --ip 192.168.1.4 --type follower

# Test Follower 2
python test_single_robot.py --ip 192.168.1.5 --type follower
```

### Step 2: Run Dual Robot Teleoperation

```bash
# Basic usage (with default IPs)
python dual_robot_teleop.py

# Custom configuration
python dual_robot_teleop.py \
    --leader1-ip 192.168.1.2 \
    --leader2-ip 192.168.1.3 \
    --follower1-ip 192.168.1.4 \
    --follower2-ip 192.168.1.5 \
    --duration 60 \
    --force-gain 0.1
```

## Configuration

### Default IP Addresses

The scripts use the following default IP addresses:
- Leader 1: `192.168.1.2`
- Leader 2: `192.168.1.3`
- Follower 1: `192.168.1.4`
- Follower 2: `192.168.1.5`

Update these in your commands if your setup differs.

### Network Setup

Ensure all robots are on the same network and accessible from your control computer. Test connectivity:

```bash
ping 192.168.1.2
ping 192.168.1.3
ping 192.168.1.4
ping 192.168.1.5
```

## Command Line Options

### dual_robot_teleop.py

```bash
python dual_robot_teleop.py --help
```

Options:
- `--leader1-ip`: IP address of leader 1 (default: 192.168.1.2)
- `--leader2-ip`: IP address of leader 2 (default: 192.168.1.3)
- `--follower1-ip`: IP address of follower 1 (default: 192.168.1.4)
- `--follower2-ip`: IP address of follower 2 (default: 192.168.1.5)
- `--duration`: Duration of teleoperation in seconds (default: 60)
- `--force-gain`: Force feedback gain, 0.05-0.2 typical (default: 0.1)
- `--model`: Robot model - wxai_v0, vxai_v0_right, or vxai_v0_left (default: wxai_v0)
- `--no-home`: Skip moving to home position
- `--no-sleep`: Skip moving to sleep position at end

### test_single_robot.py

```bash
python test_single_robot.py --help
```

Required options:
- `--ip`: IP address of robot to test
- `--type`: Robot type (leader or follower)

Optional:
- `--model`: Robot model (default: wxai_v0)

## Usage Examples

### Example 1: Quick Test

Test all robots then run 30-second teleoperation:

```bash
# Test all robots
for ip in 192.168.1.{2,3,4,5}; do
    type=$([ "$ip" == "192.168.1.2" ] || [ "$ip" == "192.168.1.3" ] && echo "leader" || echo "follower")
    python test_single_robot.py --ip $ip --type $type
done

# Run teleoperation
python dual_robot_teleop.py --duration 30
```

### Example 2: Custom Force Feedback

Adjust force feedback gain for different feel:

```bash
# Lighter force feedback
python dual_robot_teleop.py --force-gain 0.05

# Stronger force feedback
python dual_robot_teleop.py --force-gain 0.2
```

### Example 3: VXAI Robot Model

If using VXAI robots:

```bash
python dual_robot_teleop.py --model vxai_v0_right
```

### Example 4: Skip Home Position

Start directly from current position:

```bash
python dual_robot_teleop.py --no-home --no-sleep
```

## How It Works

### Teleoperation Loop

The teleoperation system runs a continuous control loop at ~1000 Hz:

1. **Leader → Follower Position Tracking**
   - Read leader joint positions and velocities
   - Command follower to track leader positions with velocity feedforward
   - Provides smooth, real-time mirroring

2. **Follower → Leader Force Feedback**
   - Read external forces sensed by follower
   - Apply scaled forces to leader in opposite direction
   - Allows operator to "feel" what the follower touches

### Control Modes

- **Leaders**: `Mode.external_effort` - Force feedback control
- **Followers**: `Mode.position` - Position tracking control

### Safety Features

- All robots move to home position before teleoperation
- Robots move to sleep position after teleoperation
- Clean shutdown on Ctrl+C interrupt
- Error logging and reporting
- Temperature monitoring

## Adjusting Parameters

### Force Feedback Gain

The `force_feedback_gain` parameter (default 0.1) scales the force feedback:

- **Lower values (0.05-0.08)**: Lighter, more delicate feel
- **Medium values (0.1-0.15)**: Balanced feedback (recommended)
- **Higher values (0.15-0.2)**: Stronger, more pronounced feedback

**Warning**: Gains above 0.2 may cause instability!

### Control Loop Rate

The control loop runs as fast as possible (~1000 Hz). To adjust timing:

1. Edit `dual_robot_teleop.py`
2. Add sleep in the `start_teleoperation()` loop:
   ```python
   time.sleep(0.001)  # 1ms delay = ~1000 Hz max
   ```

## Troubleshooting

### Connection Issues

**Problem**: Cannot connect to robot
```
RuntimeError: Failed to connect to 192.168.1.2
```

**Solutions**:
1. Check network connectivity: `ping 192.168.1.2`
2. Verify robot is powered on
3. Check IP address is correct
4. Ensure no firewall blocking ports
5. Try increasing timeout: edit script and change `timeout=20.0` to `timeout=30.0`

### Robot Not Moving

**Problem**: Robot configured but not moving

**Solutions**:
1. Check if robot is in error state
2. Try configuring with `clear_error=True`
3. Check joint limits and temperature
4. Verify mode is set correctly

### Force Feedback Too Strong/Weak

**Problem**: Force feedback feels wrong

**Solutions**:
1. Adjust `--force-gain` parameter
2. Check follower is properly calibrated
3. Verify external effort sensing is working

### Robots Moving Erratically

**Problem**: Jerky or unstable motion

**Solutions**:
1. Reduce force feedback gain
2. Check network latency
3. Ensure feedforward velocities are being used
4. Check for joint limit violations

### Temperature Warnings

**Problem**: High rotor/driver temperatures

**Solutions**:
1. Reduce operation duration
2. Allow cooling time between runs
3. Check for mechanical binding
4. Reduce force feedback gain

## API Reference

For complete API documentation, see **`PYTHON_API_GUIDE.md`**.

Quick reference:

### Core Methods

```python
# Configuration
driver = trossen_arm.TrossenArmDriver()
driver.configure(model, end_effector, ip, clear_error)

# Set modes
driver.set_all_modes(trossen_arm.Mode.position)

# Position control
driver.set_all_positions(positions, goal_time, blocking)

# Force feedback
driver.set_all_external_efforts(efforts, goal_time, blocking)

# Get state
positions = driver.get_all_positions()
velocities = driver.get_all_velocities()
efforts = driver.get_all_external_efforts()

# Cleanup
driver.cleanup()
```

## Safety Notes

⚠️ **IMPORTANT SAFETY INFORMATION**

1. **Emergency Stop**: Keep Ctrl+C ready to stop immediately
2. **Clear Workspace**: Ensure robots have clear space to move
3. **Force Limits**: Don't exceed force-gain of 0.2
4. **Temperature**: Monitor robot temperatures during operation
5. **Release Leaders**: Let go of leaders when teleoperation ends
6. **Home Position**: Always return to home before shutdown

## Advanced Usage

### Custom Trajectories

See `PYTHON_API_GUIDE.md` for trajectory generation with scipy interpolation.

### Configuration Management

Save/load robot configurations:

```python
driver.save_configs_to_file("my_config.yaml")
driver.load_configs_from_file("my_config.yaml")
```

### Multi-Threading

The driver manages internal threads. For external threading:

```python
import threading

def teleop_thread():
    # Your teleoperation code
    pass

thread = threading.Thread(target=teleop_thread)
thread.start()
thread.join()
```

## Development

### Modifying the Scripts

The scripts are designed to be modular. Key classes:

- `DualRobotTeleop`: Main teleoperation class
  - `initialize_drivers()`: Setup robots
  - `start_teleoperation()`: Main control loop
  - `_update_leader_follower_pair()`: Single pair update

### Adding New Features

Example: Add data logging

```python
import csv

class DualRobotTeleop:
    def start_teleoperation(self, duration):
        # ... existing code ...
        
        with open('teleop_data.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'leader1_pos', 'follower1_pos'])
            
            while time.time() < end_time:
                # ... control loop ...
                
                writer.writerow([
                    time.time(),
                    self.driver_leader_1.get_all_positions(),
                    self.driver_follower_1.get_all_positions()
                ])
```

## References

- **Trossen Robotics Official Docs**: Check `trossen_arm/docs/` directory
- **Python API Guide**: `PYTHON_API_GUIDE.md` in this repo
- **C++ API Reference**: `trossen_arm/include/libtrossen_arm/trossen_arm.hpp`
- **Demo Scripts**: `trossen_arm/demos/python/`

## License

- Scripts in this directory: MIT License (or your preferred license)
- `trossen-arm` package: See package license
- Trossen library: See `trossen_arm/LICENSE`

## Support

For issues with:
- **These scripts**: Open an issue in this repository
- **trossen-arm package**: Contact Trossen Robotics support
- **Hardware**: Refer to Trossen Robotics documentation

## Version Info

- Python Package: `trossen-arm` from PyPI
- Compatible Python: 3.7+
- Required: numpy, scipy (for trajectory generation)

---

**Happy Teleoperating! 🤖**
