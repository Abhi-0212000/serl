# SERL Understanding Summary: Direct Answers to Your Questions

*Last updated: November 3, 2025*

This document directly answers all the confusing questions a beginner might have when starting with SERL, especially for someone coming from a background without RL experience and wanting to use it with a Trossen robot setup.

---

## Table of Contents
1. [What is SERL? What's the Purpose of This Repo?](#what-is-serl-whats-the-purpose-of-this-repo)
2. [Human Intervention During Training - How Does That Work?](#human-intervention-during-training---how-does-that-work)
3. [MuJoCo and Gym Environment Explained](#mujoco-and-gym-environment-explained)
4. [Training Process: Sim vs Real, Data Collection](#training-process-sim-vs-real-data-collection)
5. [Available Algorithms and Their Inputs/Outputs](#available-algorithms-and-their-inputsoutputs)
6. [Hardware Requirements: Do I Need Real Robot to Train?](#hardware-requirements-do-i-need-real-robot-to-train)
7. [Adapting SERL for Trossen Robot Setup](#adapting-serl-for-trossen-robot-setup)
8. [Multi-Robot Setup Capability](#multi-robot-setup-capability)
9. [Data Format and Recording](#data-format-and-recording)

---

## What is SERL? What's the Purpose of This Repo?

**SERL = Sample-Efficient Robotic Reinforcement Learning**

### Core Purpose
SERL is a **practical RL training framework** for robot manipulation tasks that focuses on:
- Learning policies **quickly** (30 mins to 2 hours) with minimal data
- Learning from **both demonstrations and online interaction**
- Supporting **human intervention during training** (though this repo has limited support - see next section)
- Working with both **simulation and real robots**

### What This Repo Provides
```
serl/
├── serl_launcher/          # Core RL algorithms (DrQ, SAC, BC, VICE)
├── franka_sim/            # MuJoCo simulation for Franka robot
├── serl_robot_infra/      # Real robot interface (Franka-specific)
└── examples/              # Training scripts for various tasks
```

**Think of it as:** A complete pipeline from environment setup → data collection → training → deployment, specifically designed for robotic manipulation.

---

## Human Intervention During Training - How Does That Work?

### The Confusion

You mentioned: *"Humans can intervene during training to correct mistakes"* - but this is **HIL-SERL** (Human-in-the-Loop SERL), which is a newer, separate project.

### What This SERL Repo Actually Does

**1. Pre-Training Demonstrations (Offline)**
```python
# Record demos BEFORE training starts
python record_demo.py  # Uses spacemouse to teleoperate
# Saves to: demos.pkl
```

**2. Limited Online Intervention**
During training, you can intervene via:
```python
# In the code, there's SpacemouseIntervention wrapper
env = SpacemouseIntervention(env)  # Touch spacemouse during training
```
However, this is **not the main focus**. The "intervention" here means:
- The spacemouse can trigger a pause
- You can manually correct the robot
- But it's not seamlessly integrated like HIL-SERL

**3. HIL-SERL vs SERL**

| Feature | SERL (this repo) | HIL-SERL (newer) |
|---------|------------------|------------------|
| Offline demos | ✅ Yes | ✅ Yes |
| Online corrections | ⚠️ Limited | ✅ Seamless |
| Training paradigm | Demo → RL | Demo + Continuous Intervention |
| Main philosophy | Sample-efficient RL | Human-guided RL |

**Bottom line:** This SERL repo trains primarily from:
1. Pre-recorded demonstrations (10-20 trajectories)
2. Online exploration by the policy
3. Optional human corrections (but not the main workflow)

---

## MuJoCo and Gym Environment Explained

### Why MuJoCo?

**MuJoCo = Physics Simulator**
- Simulates robot physics, contacts, dynamics
- NOT just for visualization - it's the entire simulation engine

**Gym = Standard RL Interface**
```python
env = gym.make("PandaPickCubeVision-v0")  # Creates the environment
obs, _ = env.reset()                       # Start episode
next_obs, reward, done, _, _ = env.step(action)  # Take action
```

### Two Use Cases

#### 1. **Simulation Training (Pure Sim)**
```bash
# franka_sim/ provides a MuJoCo simulation
python examples/async_drq_sim/async_drq_sim.py --actor --learner
```
- MuJoCo renders the robot and cube
- Policy interacts with simulated robot
- Training happens in simulation
- Can train without real hardware!

#### 2. **Real Robot Training**
```bash
# serl_robot_infra/ provides real robot interface
python examples/async_peg_insert_drq/async_drq_randomized.py
```
- Gym env **sends commands to real robot**
- No MuJoCo involved during execution
- MuJoCo is ONLY used if you want to visualize or debug

### Visualization vs Training

**Simulation:**
```python
# Visualization IS the environment
env = gym.make("PandaPickCubeVision-v0")  # MuJoCo sim
obs, _ = env.reset()  # Resets simulated robot
env.render()  # Shows MuJoCo visualization
```

**Real Robot:**
```python
# Gym env wraps real robot - no MuJoCo
env = gym.make("FrankaPegInsert-Vision-v0")  # Real robot
obs, _ = env.reset()  # Moves real robot to start
# No rendering - you see the real robot move!
```

---

## Training Process: Sim vs Real, Data Collection

### Entry Points and Workflow

Every training script follows this pattern:

```python
def main():
    # 1. Create environment (sim or real)
    env = gym.make("...")
    
    # 2. Create RL agent
    agent = DrQAgent.create_drq(...)
    
    # 3. Launch learner OR actor
    if FLAGS.learner:
        learner(agent, replay_buffer, ...)  # Update policy
    elif FLAGS.actor:
        actor(agent, data_store, env, ...)   # Collect data
```

### Actor vs Learner (Critical Concept!)

**Actor Node:**
- Runs the policy on the environment
- Collects experience (transitions)
- Sends data to learner over network
```python
# In actor loop
action = agent.sample_actions(obs)  # Policy decides
next_obs, reward, done = env.step(action)  # Execute
data_store.insert(transition)  # Send to learner
```

**Learner Node:**
- Receives data from actor
- Updates policy using RL algorithm
- Sends updated policy back to actor
```python
# In learner loop
batch = replay_buffer.sample(batch_size)  # Get data
agent, update_info = agent.update(batch)  # Train
# Policy automatically synced to actor
```

**Why separate?**
- Actor needs GPU for inference (fast)
- Learner needs more GPU for training (heavy)
- Can run on separate machines!

### Data Sources

**Three types of data:**

1. **Random Exploration**
```python
if step < FLAGS.random_steps:  # First 300 steps
    actions = env.action_space.sample()  # Random
```

2. **Offline Demos** (pre-recorded)
```python
with open("demos.pkl", "rb") as f:
    trajs = pkl.load(f)
    for traj in trajs:
        demo_buffer.insert(traj)  # Load into buffer
```

3. **Online Policy Rollouts**
```python
action = agent.sample_actions(obs)  # Policy acts
```

### Is It Sim Data or Real Data?

**Depends on the example:**

| Example Directory | Environment | Data Source |
|-------------------|-------------|-------------|
| `async_drq_sim/` | MuJoCo Simulation | Sim trajectories only |
| `async_sac_state_sim/` | MuJoCo Simulation | Sim trajectories only |
| `async_peg_insert_drq/` | Real Franka Robot | Teleoperated demos + real rollouts |
| `async_pcb_insert_drq/` | Real Franka Robot | Teleoperated demos + real rollouts |

**Key Point:** 
- `franka_sim` examples = pure simulation
- `serl_robot_infra` examples = real robot with real demos

---

## Available Algorithms and Their Inputs/Outputs

### 1. **DrQ (Data-Regularized Q-Learning)**

**What it is:** SAC + image augmentation for vision-based RL

**Inputs:**
```python
observations = {
    "images": {"front": np.array([128,128,3]), "wrist": np.array([128,128,3])},
    "state": {"tcp_pos": np.array([x,y,z]), ...}
}
actions = np.array([dx, dy, dz, droll, dpitch, dyaw, gripper])
```

**Outputs:**
```python
action = agent.sample_actions(observations)  # Returns action
# action shape: (action_dim,) e.g., (7,) for 6D pose + gripper
```

**When to use:** Image-based manipulation tasks

**Example:**
```bash
cd examples/async_drq_sim
bash run_learner.sh  # Starts DrQ training
bash run_actor.sh    # Starts rollouts
```

### 2. **SAC (Soft Actor-Critic)**

**What it is:** Entropy-regularized RL for continuous actions

**Inputs:**
```python
observations = {
    "state": {"joint_pos": np.array([7,]), "tcp_pos": np.array([3,]), ...}
}
actions = np.array([action_dim])
```

**Outputs:**
```python
action = agent.sample_actions(observations, deterministic=False)
```

**When to use:** State-based (no images) manipulation

**Example:**
```bash
cd examples/async_sac_state_sim
bash tmux_launch.sh  # One-liner to start training
```

### 3. **BC (Behavior Cloning)**

**What it is:** Supervised learning from demonstrations

**Inputs:**
```python
# Demonstrations: (obs, action) pairs
demos = load_demos("demos.pkl")
```

**Outputs:**
```python
action = bc_agent.predict(observation)
```

**When to use:** Baseline comparison, or when you have lots of expert demos

**Example:**
```bash
cd examples/async_peg_insert_drq
python bc_policy.py --demo_path demos.pkl
```

### 4. **RLPD (RL with Prior Data)**

**What it is:** DrQ + heavily weighted demos

**Usage:**
```bash
bash run_learner.sh --demo_path demos.pkl  # Demos loaded
# Higher sampling ratio from demo buffer vs replay buffer
```

**When to use:** You have good demos and want faster training

### 5. **VICE (Value Intervention from Corrections)**

**What it is:** Uses human corrections as value labels

**Less commonly used in this repo** - mainly for research

---

## Hardware Requirements: Do I Need Real Robot to Train?

### Can I Train Without Hardware?

**YES!** You can:

1. **Train entirely in simulation:**
```bash
cd examples/async_drq_sim
bash tmux_launch.sh  # No robot needed
# Trains a policy in MuJoCo sim
```

2. **Evaluate the simulated policy:**
```python
# After training, policy is saved
# Load and test in sim
```

### Sim-to-Real Transfer

**Question:** Can I train in sim and deploy to real robot?

**Answer:** **Partially supported, but not the main focus**

The repo is designed for:
- **Training directly on real robot** (preferred)
- Training in sim for prototyping/debugging

**Why not pure sim-to-real?**
- Sim-to-real has reality gap issues
- SERL's philosophy: train on real robot with real data
- 30-60 minutes of real robot training >> hours of sim training

**If you want sim-to-real:**
- Train in `franka_sim`
- Create matching gym env for your real robot
- Fine-tune on real robot with a few demos

---

## Adapting SERL for Trossen Robot Setup

### Current Status

SERL is **Franka-specific** in:
1. `franka_sim/` - MuJoCo model of Franka
2. `serl_robot_infra/` - ROS controllers for Franka
3. Examples - All use Franka environments

### What You Need to Change

#### 1. **Create Trossen MuJoCo Simulation**

```bash
serl/
└── trossen_sim/  # NEW: Copy from franka_sim
    ├── trossen_sim/
    │   ├── mujoco_gym_env.py  # Base gym env
    │   ├── envs/
    │   │   ├── trossen_pick_gym_env.py  # NEW: Your task
    │   │   └── xmls/
    │   │       └── trossen_arena.xml  # NEW: Trossen URDF→MJCF
    └── setup.py
```

**Steps:**
```python
# 1. Convert Trossen URDF to MuJoCo XML
# Use: https://github.com/kevinzakka/mjcf

# 2. Modify PandaPickCubeGymEnv → TrossenPickCubeGymEnv
class TrossenPickCubeGymEnv(MujocoGymEnv):
    def __init__(self, ...):
        super().__init__(
            xml_path="trossen_arena.xml",  # Your XML
            ...
        )
        # Update DOF, joint names, etc. for Trossen
```

#### 2. **Create Trossen Real Robot Interface**

```bash
serl_robot_infra/
└── trossen_env/  # NEW
    ├── __init__.py
    ├── envs/
    │   ├── trossen_base_env.py  # Interface to Trossen ROS
    │   └── trossen_pick_env.py  # Task-specific
    └── config.py
```

**Key changes:**
```python
# Instead of Franka ROS topics/services:
# /franka_control/...

# Use Trossen topics:
# /trossen_control/joint_states
# /trossen_control/command
```

#### 3. **Modify Training Scripts**

```bash
examples/
└── async_trossen_pick_drq/  # NEW: Copy from async_drq_sim
    ├── async_drq_trossen.py  # Modified script
    ├── run_actor.sh
    └── run_learner.sh
```

```python
# In async_drq_trossen.py
import trossen_sim  # Instead of franka_sim

env = gym.make("TrossenPickCube-v0")  # Your env
```

### Multi-Robot (Dual Trossen) Setup

**Is it possible?** **YES, but requires significant work**

#### Option 1: Two Separate Agents (Simpler)
```python
# Two independent robots, two policies
env_left = gym.make("TrossenLeft-v0")
env_right = gym.make("TrossenRight-v0")
agent_left = DrQAgent.create_drq(...)
agent_right = DrQAgent.create_drq(...)

# Train separately or alternating
```

#### Option 2: Single Coordinated Policy (Advanced)
```python
# Single observation includes both robots
obs = {
    "images": {"left_wrist": ..., "right_wrist": ..., "front": ...},
    "state": {
        "left_tcp": ...,
        "right_tcp": ...,
    }
}
# Action: [left_action (7,), right_action (7,)] = (14,)
actions = agent.sample_actions(obs)  # Returns (14,) vector
```

**Changes needed:**
1. Gym env returns combined observation
2. Action space is concatenated: `(7,) → (14,)`
3. MuJoCo XML has two robot arms
4. Reward function considers both arms

**Reference:** See `async_bin_relocation_fwbw_drq` - it uses forward/backward policies (similar concept)

---

## Multi-Robot Setup Capability

### Can SERL Train Multi-Robot Setups?

**Short answer:** Not out-of-the-box, but architecturally possible

### Single Robot with Dual Policies (Existing)

Example: `async_bin_relocation_fwbw_drq`
```python
# Two policies for same robot doing forward/backward task
agent_fw = DrQAgent.create_drq(...)  # Forward policy
agent_bw = DrQAgent.create_drq(...)  # Backward policy

# Actor switches between them
if mode == "forward":
    action = agent_fw.sample_actions(obs)
else:
    action = agent_bw.sample_actions(obs)
```

### Dual Robot Setup (Extension)

**Approach 1: Centralized Policy**
```python
# Concatenate everything
obs_combined = {
    "images": {
        "robot1_wrist": img1,
        "robot2_wrist": img2,
        "front": img_front,
    },
    "state": {
        "robot1_tcp": tcp1,
        "robot2_tcp": tcp2,
    }
}
action_combined = agent.sample_actions(obs_combined)
# action_combined = [action1, action2]
robot1.step(action_combined[:7])
robot2.step(action_combined[7:])
```

**Approach 2: Decentralized (MARL)**
```python
# Each robot has own policy but shares replay buffer
agent1 = DrQAgent.create_drq(...)
agent2 = DrQAgent.create_drq(...)

# Can share experiences
shared_replay_buffer = ...
```

**Challenges:**
- Observation space design
- Action space design
- Reward shaping (cooperative task)
- Synchronization

### Can It Be Done in Sim?

**YES!** 
1. Create MuJoCo XML with two robot arms
2. Modify gym env to control both
3. Train in simulation

**Example structure:**
```xml
<!-- trossen_dual.xml -->
<mujoco>
  <worldbody>
    <body name="robot1" pos="0 -0.5 0">
      <!-- Trossen arm 1 -->
    </body>
    <body name="robot2" pos="0 0.5 0">
      <!-- Trossen arm 2 -->
    </body>
    <body name="shared_object">
      <!-- Object both manipulate -->
    </body>
  </worldbody>
</mujoco>
```

---

## Data Format and Recording

### Demo Data Format

**Pickle file structure:**
```python
# demos.pkl contains list of trajectories
demos = [
    {  # Trajectory 1
        "observations": [...],  # List of obs dicts
        "actions": [...],       # List of action arrays
        "rewards": [...],       # List of floats
        "dones": [...],         # List of bools
    },
    {  # Trajectory 2
        ...
    },
]
```

**Single transition:**
```python
transition = {
    "observations": {
        "images": {"front": (128,128,3), "wrist": (128,128,3)},
        "state": {"tcp_pos": (3,), "tcp_vel": (3,), ...}
    },
    "actions": np.array([7,]),  # [dx, dy, dz, drot_x, drot_y, drot_z, gripper]
    "next_observations": { ... },
    "rewards": 0.0 or 1.0,
    "masks": 1.0,  # (1 - done)
    "dones": False or True,
}
```

### How to Record Data

#### Simulation:
```bash
cd examples/async_drq_sim
python record_demo.py --save_path demos.pkl

# Uses spacemouse or keyboard to control simulated robot
# Saves when you press 's' or end episode
```

#### Real Robot:
```bash
cd examples/async_peg_insert_drq
python record_demo.py

# Uses spacemouse to teleoperate real robot
# Records camera images + state
```

**Code example:**
```python
# From record_demo.py
env = gym.make("FrankaPegInsert-Vision-v0")
env = SpacemouseIntervention(env)  # Enable spacemouse

trajectory = []
obs, _ = env.reset()

while not done:
    action = spacemouse.get_action()  # From spacemouse
    next_obs, reward, done, _, info = env.step(action)
    
    trajectory.append({
        "observations": obs,
        "actions": action,
        "rewards": reward,
        "dones": done,
    })
    obs = next_obs

# Save
with open("demos.pkl", "wb") as f:
    pickle.dump([trajectory], f)
```

### RLDS Format (Alternative)

**For compatibility with RTX datasets:**
```bash
# Save during training
bash run_learner.sh --log_rlds_path /path/to/save

# Creates TensorFlow dataset:
# /path/to/save/
#   ├── dataset_info.json
#   ├── features.json
#   └── serl_rlds_dataset-train.tfrecord-*
```

---

## Summary: Your Learning Path

### If You're Starting with SERL on Trossen:

**Phase 1: Understand (1 week)**
1. Run simulation examples
2. Understand actor-learner separation
3. Study Franka gym envs as reference

**Phase 2: Adapt Simulation (2-3 weeks)**
1. Create Trossen MuJoCo model
2. Build `trossen_sim` gym env
3. Test with DrQ in simulation

**Phase 3: Real Robot (2-3 weeks)**
1. Set up ROS interface for Trossen
2. Create `trossen_env` gym env
3. Test spacemouse teleoperation
4. Record demos

**Phase 4: Training (1 week)**
1. Modify training scripts
2. Train on simple task (cube stacking)
3. Iterate on reward function

**Phase 5: Multi-Robot (if needed) (3-4 weeks)**
1. Design dual-arm observation/action spaces
2. Extend simulation
3. Train cooperative policy

---

## Quick Reference: Essential Commands

```bash
# Simulation Training
cd examples/async_drq_sim
bash tmux_launch.sh  # All-in-one

# Real Robot Training
cd examples/async_peg_insert_drq
python record_demo.py  # Step 1: Demos
bash run_learner.sh    # Step 2: Start learner
bash run_actor.sh      # Step 3: Start actor

# Check training progress
# Look for: eval/success_rate, eval/return in wandb or terminal logs
```

---

*This document provides direct, practical answers. For deeper architectural details, see SERL_COMPLETE_GUIDE.md*
