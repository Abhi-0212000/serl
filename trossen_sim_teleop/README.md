# Trossen Real-to-Sim Teleoperation

Control MuJoCo sim robots using real physical Trossen leader robots.

## How It Works

```
Physical Leader Robot           MuJoCo Sim Follower
┌────────────────┐             ┌────────────────┐
│  You move this │             │  Mirrors leader│
│  (free motion) │────USB──────►│  (qpos update) │
└────────────────┘             └────────────────┘
     ↑ trossen_arm                  ↑ mujoco
     get_all_positions()            qpos injection
```

**Key MuJoCo concept:**
- `data.qpos[:] = joint_angles` - Directly set robot pose
- `mj_forward()` - Update kinematics/physics
- `viewer.sync()` - Render frame

No physics simulation during teleop - just kinematic mirroring!

## Files

- `single_robot_teleop_sim.py` - 1 leader → 1 sim robot
- `dual_robot_teleop_sim.py` - 2 leaders → 2 sim robots

## Prerequisites

```bash
# Install trossen_arm Python bindings
pip install trossen-arm

# Already have mujoco (comes with trossen_sim)
```

## Usage

### Single Robot Teleop
```bash
cd /home/qte9489/personal_abhi/serl/trossen_sim_teleop

# Connect leader via USB, then:
python single_robot_teleop_sim.py \
    --leader-ip 192.168.1.2 \
    --duration 60

# Or skip home position:
python single_robot_teleop_sim.py --no-home --duration 30
```

### Dual Robot Teleop  
```bash
# Connect both leaders, then:
python dual_robot_teleop_sim.py \
    --left-ip 192.168.1.2 \
    --right-ip 192.168.1.3 \
    --duration 60
```

## What Happens

1. **Initialize**: Connect to leader robots via `trossen_arm` SDK
2. **Load Sim**: Load MuJoCo model from `trossen_ai_scene_joint.xml`
3. **Randomize Cube**: Spawn red cube at random table position
4. **Home Position**: Move leaders to safe home pose (optional)
5. **Teleop Loop**:
   ```python
   while running:
       # Read from leader
       joints = leader.get_all_positions()  # 6D array (radians)
       gripper = leader.get_gripper_position()  # Scalar (meters)
       
       # Inject into sim
       data.qpos[0:6] = joints
       data.qpos[6] = gripper
       data.qvel[:] = 0  # No velocities (static pose)
       
       # Update sim
       mujoco.mj_forward(model, data)
       viewer.sync()
   ```
6. **Cleanup**: Close connections

## Joint Mapping

**Leader (physical)** → **Follower (sim)**

```
Leader qpos indices:          Sim qpos indices:
─────────────────────        ───────────────────
Single robot:                Single robot:
  joints[0-5]: arm             qpos[0-5]: left arm
  gripper: gripper             qpos[6]: left gripper

Dual robots:                 Dual robots:
  left_joints[0-5]            qpos[0-5]: left arm
  left_gripper                qpos[6]: left gripper
  right_joints[0-5]           qpos[7-12]: right arm  
  right_gripper               qpos[13]: right gripper
```

## Troubleshooting

**"No module named 'trossen_arm'"**
→ `pip install trossen-arm`

**"XML file not found"**
→ Check path to `trossen_ai_scene_joint.xml` in code

**"Leader connection failed"**
→ Check USB connection and IP address
→ Try: `ping 192.168.1.2`

**Sim robot doesn't move**
→ Check `sim_joint_start_idx` matches your XML joint order
→ Print `mj_model.joint_names` to see joint order

**Gripper not working**
→ Adjust gripper index in code (should be after 6 arm joints)

**Cube not visible**
→ Check cube joint name in XML (should be "cube_joint")

## Next Steps

To record demos for training:
1. Run teleop script
2. Add data recording (save qpos, qvel, gripper states)
3. Convert to RLDS format
4. Use with `--preload_rlds_path` in training

## Reference

Based on ChatGPT's MuJoCo teleop pattern:
```python
while viewer.is_running():
    q_real = get_real_robot_joint_positions()
    data.qpos[:] = q_real
    mujoco.mj_forward(model, data)
    viewer.sync()
```
