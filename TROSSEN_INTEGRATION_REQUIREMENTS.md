# Trossen Robot Integration with SERL - Requirements Document

**Date:** November 3, 2025  
**Purpose:** Adapt SERL (Sample-Efficient Robotic Reinforcement Learning) for Trossen robot arms

---

## Executive Summary

After thoroughly analyzing the SERL repository, I've identified exactly what you need to provide to create a working Trossen simulation similar to the Franka implementation. This document outlines all requirements in order of priority.

---

## 🎯 Critical Requirements (Must Have)

### 1. **Trossen Robot URDF or MJCF File**

**What I need:**
- Robot description file in one of these formats:
  - **URDF** (Unified Robot Description Format) - Preferred
  - **MJCF** (MuJoCo XML format) - Can use directly
  - **SDF** (Simulation Description Format) - Can convert

**What it should contain:**
- Complete kinematic chain (all links and joints)
- Joint limits (min/max angles)
- Link masses and inertias
- Collision geometries
- Visual meshes (STL, OBJ, or DAE files)

**Example from Franka:**
```
franka_emika_panda/
├── assets/
│   ├── link0.stl
│   ├── link1.stl
│   ├── ... (more link meshes)
│   └── link7.stl
└── panda.xml  # MJCF file with robot definition
```

**What I'll do with it:**
- If URDF: Convert to MJCF using `mjcf` Python library
- If MJCF: Use directly
- Place in `trossen_sim/trossen_sim/envs/xmls/trossen_robot/`

---

### 2. **Robot Specifications**

**What I need to know:**

#### A. Joint Information
```yaml
number_of_joints: 5 or 6 or 7?  # How many DOF?
joint_names:
  - joint1  # What are they called?
  - joint2
  - ...
joint_types:
  - revolute or prismatic  # For each joint
joint_limits:
  - [-π, π]  # Min/max for each joint (radians)
  - ...
```

#### B. End-Effector Information
```yaml
tcp_name: "tool_center_point" or "ee_link"  # What's the end-effector link called?
tcp_offset: [0, 0, 0.1]  # Offset from last link to actual TCP (x, y, z in meters)
```

#### C. Gripper Information
```yaml
gripper_type: "parallel_jaw" or "suction" or "none"
gripper_joint_name: "gripper_joint"  # If parallel jaw
gripper_control_range: [0, 0.08]  # Min/max opening (meters)
gripper_force_limit: 50  # Maximum grip force (Newtons)
```

#### D. Physical Properties
```yaml
workspace_bounds:
  x: [0.2, 0.6]  # Reachable space (meters)
  y: [-0.3, 0.3]
  z: [0.0, 0.5]
  
base_height: 0.0  # Height of robot base above ground (meters)
max_payload: 2.0  # Maximum load (kg)
```

---

### 3. **Home/Reset Position**

**What I need:**
- Default joint angles when robot resets
- Should be a "neutral" pose in middle of workspace

**Format:**
```python
TROSSEN_HOME = np.array([
    0.0,    # joint1 angle (radians)
    -0.785, # joint2 angle
    0.0,    # joint3 angle
    # ... for all joints
])
```

**How to provide:**
- Can be in degrees (I'll convert)
- Or describe: "all joints at 0°" or "elbow bent 90°, rest straight"

---

## ⚙️ Important Requirements (Should Have)

### 4. **Control Parameters**

**What I need to know:**

```yaml
# PD control gains (for operational space control)
position_gains: [200, 200, 200]  # Kp for x, y, z
orientation_gains: [200, 200, 200]  # Kp for roll, pitch, yaw
damping_ratio: 1.0  # Usually between 0.7-1.5

# Action scaling
max_position_delta: 0.1  # Max TCP movement per timestep (meters)
max_gripper_delta: 1.0   # Max gripper change per timestep

# Control frequency
control_dt: 0.02  # Control timestep (20ms = 50Hz)
physics_dt: 0.002  # Physics timestep (2ms = 500Hz)
```

**If you don't know:**
- I'll use Franka's values as defaults (they're pretty standard)
- We can tune later

---

### 5. **Camera Setup**

**What I need to know:**

```yaml
cameras:
  front:
    position: [0.8, 0.0, 0.5]  # Where to mount (x, y, z)
    look_at: [0.3, 0.0, 0.2]   # What to look at
    fov: 45  # Field of view (degrees)
    
  wrist:  # Optional: Camera on end-effector
    position: [0.05, 0, 0.05]  # Relative to TCP
    orientation: [0, -90, 0]   # Roll, pitch, yaw (degrees)
    fov: 50
    
  side:  # Optional: Additional view
    position: [0.3, 0.8, 0.5]
    look_at: [0.3, 0.0, 0.2]
    fov: 45
```

**Defaults if not provided:**
- Front camera: Similar to Franka setup
- No wrist camera initially

---

### 6. **Task-Specific Information**

**For the initial task (cube picking):**

```yaml
cube:
  size: 0.02  # Half-size (meters) - makes 4cm cube
  mass: 0.1   # kg
  color: [0.6, 0.3, 0.6, 1.0]  # RGBA
  
table:
  height: 0.0  # meters above ground
  material: "wood" or "metal"
  
workspace:
  cube_spawn_area:
    x: [0.25, 0.55]  # Where to randomly place cube
    y: [-0.25, 0.25]
```

---

## 📦 Optional Enhancements (Nice to Have)

### 7. **Visual Assets**

**If you have:**
- High-quality mesh files (STL/OBJ) for realistic rendering
- Textures for the robot links
- CAD models

**If you don't have:**
- I'll use primitive shapes (cylinders, boxes) for visualization
- Still fully functional, just less pretty

---

### 8. **Real Robot Interface Information**

**For future real robot integration (not needed for sim):**

```yaml
ros_info:
  robot_namespace: "/trossen"
  state_topic: "/trossen/joint_states"
  command_topic: "/trossen/joint_command"
  controller_name: "position_controller" or "impedance_controller"
  
communication:
  protocol: "ROS" or "SDK" or "USB"
  control_mode: "position" or "velocity" or "torque"
```

---

## 📋 Information Collection Template

**Please fill out what you can:**

```yaml
# ========== ROBOT BASIC INFO ==========
robot_name: "Trossen WidowX" or "ViperX" or "ReactorX"?
degrees_of_freedom: ?
has_gripper: yes/no?

# ========== FILES ==========
urdf_file_location: "/path/to/robot.urdf" or "I'll email it"
mesh_files_location: "/path/to/meshes/" or "included in URDF"

# ========== JOINT CONFIG ==========
joint_names: [?, ?, ?, ...]
joint_limits_degrees:
  - [?, ?]  # For each joint
  - ...
  
home_position_degrees: [?, ?, ?, ...]  # Or describe in words

# ========== WORKSPACE ==========
typical_workspace_description: "Can reach 30cm forward, 20cm to sides, 40cm up"
base_mounted_on: "table" or "floor" or "wall"?

# ========== GRIPPER ==========
gripper_type: "parallel" or "suction" or "none"
gripper_max_opening_mm: ?  # If parallel
gripper_max_force_N: ?

# ========== CONTROL ==========
# (Leave blank if unknown - I'll use defaults)
recommended_control_frequency_Hz: ?
typical_max_speed_m_per_sec: ?

# ========== YOUR GOALS ==========
intended_tasks: "Pick and place cubes" or "Cable routing" or ...
single_robot_or_dual: "single" or "dual arm setup"
```

---

## 🚀 What I'll Deliver

Once you provide the above information, I will create:

### 1. **Trossen Simulation Package**
```
trossen_sim/
├── setup.py
├── requirements.txt
├── trossen_sim/
│   ├── __init__.py  # Gym registration
│   ├── mujoco_gym_env.py  # Base class (copied from franka)
│   ├── controllers/
│   │   └── opspace.py  # Operational space controller
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── trossen_pick_gym_env.py  # Main environment
│   │   ├── utils.py
│   │   └── xmls/
│   │       ├── arena.xml  # Complete scene
│   │       ├── trossen.xml  # Robot definition
│   │       └── trossen_robot/  # Robot assets
│   │           ├── assets/
│   │           │   ├── link0.stl
│   │           │   └── ...
│   │           └── robot.xml
│   └── test/
│       ├── test_gym_env_human.py
│       └── test_gym_env_render.py
```

### 2. **Training Examples**
```
examples/
└── async_trossen_pick_drq/
    ├── async_drq_trossen.py  # Training script
    ├── run_learner.sh
    ├── run_actor.sh
    ├── tmux_launch.sh
    └── README.md
```

### 3. **Documentation**
- Installation guide
- Quick start tutorial
- Configuration reference
- Troubleshooting guide

---

## 📝 Step-by-Step Process

### Step 1: You Provide (Priority Order)

1. **Minimum to start:**
   - URDF file (or MJCF)
   - Number of joints
   - Home position

2. **To make it work well:**
   - Joint limits
   - Workspace bounds
   - Gripper specs (if applicable)

3. **To make it realistic:**
   - Control parameters
   - Camera positions
   - Visual meshes

### Step 2: I Create

1. Convert URDF to MJCF (if needed)
2. Create `trossen_sim` package structure
3. Adapt `TrossenPickCubeGymEnv` from `PandaPickCubeGymEnv`
4. Configure operational space controller
5. Set up cameras and rendering
6. Create training scripts
7. Test everything

### Step 3: We Iterate

1. You test the simulation
2. We tune parameters (gains, limits, rewards)
3. We verify behavior matches real robot (if you have one)
4. We add more tasks/features as needed

---

## 🎓 Technical Details (For Reference)

### What Gets Adapted from Franka

| Component | Adaptation Needed | Difficulty |
|-----------|-------------------|------------|
| Base gym environment | None (reuse) | ✅ Easy |
| Operational space controller | Update DOF, joint IDs | ✅ Easy |
| MuJoCo XML scene | Replace robot model | ⚠️ Moderate |
| Joint/actuator definitions | Match Trossen specs | ⚠️ Moderate |
| Gripper control | Adapt to Trossen gripper | ⚠️ Moderate |
| Camera setup | Reposition for workspace | ✅ Easy |
| Observation space | Match Trossen DOF | ✅ Easy |
| Action space | Keep same (delta control) | ✅ Easy |
| Training scripts | Minimal changes | ✅ Easy |
| RL algorithms | No changes | ✅ Easy |

### Estimated Timeline

- **With complete information:** 2-3 days
- **With minimal information:** 1 week (more iteration)
- **With dual-arm setup:** Add 1-2 weeks

---

## ❓ Common Questions

**Q: I don't have a URDF file, what do I do?**  
A: Check:
1. Trossen's GitHub (they usually provide URDF)
2. ROS packages for your robot model
3. Contact Trossen support
4. I can create a simplified model from specifications

**Q: My robot is different from Franka (fewer/more joints)**  
A: No problem! The code is modular - I'll adjust the DOF everywhere needed.

**Q: Can we start with a simple version and add features later?**  
A: Absolutely! I recommend:
1. Phase 1: Basic cube picking (state-based, no images)
2. Phase 2: Add image observations
3. Phase 3: Add real robot interface
4. Phase 4: Add dual-arm support

**Q: I want to use this for a different task (not cube picking)**  
A: Easy! Just tell me:
- What objects are involved?
- What's the goal?
- How do we detect success?

---

## 📧 What to Send Me

**Minimum:**
1. Robot name/model
2. URDF file (or link to where I can download it)
3. Number of DOF
4. Does it have a gripper?

**Helpful:**
- Filled template from above
- Photos/videos of the robot
- Any existing code/ROS packages
- Your intended use case

**Format:**
- Email with attachments
- GitHub link
- Google Drive share
- Or just paste specifications in chat

---

## 🎯 Next Steps

1. **You:** Gather the information above (especially URDF)
2. **Me:** Review and ask clarifying questions
3. **Us:** Agree on scope for Phase 1
4. **Me:** Create the simulation package
5. **You:** Test and provide feedback
6. **Us:** Iterate until it works perfectly!

---

**Ready to start? Send me the robot specifications and URDF file!** 🚀

*Note: Even if you can't provide everything, send what you have and we'll figure out the rest together.*
