# SERL Complete Architecture Guide

**A comprehensive deep-dive into the SERL codebase**

*Last updated: November 3, 2025*

---

## Table of Contents

### Part 1: Foundation & Architecture
1. [Introduction and Philosophy](#1-introduction-and-philosophy)
2. [Repository Structure Deep-Dive](#2-repository-structure-deep-dive)
3. [Core Architecture: Actor-Learner Pattern](#3-core-architecture-actor-learner-pattern)
4. [Data Flow and Communication](#4-data-flow-and-communication)

### Part 2: Environment & Simulation
5. [MuJoCo Simulation Environment](#5-mujoco-simulation-environment)
6. [Gym Environment Design](#6-gym-environment-design)
7. [Observation and Action Spaces](#7-observation-and-action-spaces)
8. [Environment Wrappers](#8-environment-wrappers)

### Part 3: Algorithms & Training
9. [RL Algorithms Explained](#9-rl-algorithms-explained)
10. [Neural Network Architecture](#10-neural-network-architecture)
11. [Training Loop Mechanics](#11-training-loop-mechanics)
12. [Replay Buffer System](#12-replay-buffer-system)

### Part 4: Real Robot & Advanced
13. [Real Robot Infrastructure](#13-real-robot-infrastructure)
14. [Reward Engineering](#14-reward-engineering)
15. [Advanced Features](#15-advanced-features)
16. [Extending SERL](#16-extending-serl)

---

## 1. Introduction and Philosophy

### What Problem Does SERL Solve?

Traditional RL for robotics faces these challenges:
- **Sample inefficiency**: Needs millions of interactions
- **Sim-to-real gap**: Policies trained in sim fail on real robots
- **No human guidance**: Can't leverage human expertise during training
- **Slow convergence**: Takes hours or days to learn simple tasks

**SERL addresses these by:**
1. **Sample efficiency**: Learn from 10-20 demos + limited exploration
2. **Real robot training**: Train directly on hardware (no sim-to-real gap)
3. **Demo bootstrapping**: Start with human demonstrations
4. **Fast convergence**: 30-60 minutes for simple manipulation tasks

### Key Philosophy

```
┌────────────────────────────────────────────────────────┐
│  SERL Philosophy: Practical RL for Real Robots        │
├────────────────────────────────────────────────────────┤
│                                                         │
│  1. Start with Demos                                   │
│     └─> Human shows the task 10-20 times              │
│                                                         │
│  2. Learn Online                                       │
│     └─> Policy explores and improves                  │
│                                                         │
│  3. Fast Iteration                                     │
│     └─> See results in under an hour                  │
│                                                         │
│  4. Minimal Infrastructure                             │
│     └─> Works with standard robots + cameras          │
│                                                         │
└────────────────────────────────────────────────────────┘
```

### Target Users

- Robotics researchers wanting to apply RL
- Engineers building manipulation systems
- Students learning robot learning
- Anyone with a robot arm and want to train policies

### What You Get

**Out of the box:**
- State-of-the-art RL algorithms (DrQ, SAC, RLPD)
- Simulation environment (Franka robot + MuJoCo)
- Real robot interfaces (Franka Panda)
- Training infrastructure (distributed actor-learner)
- Example tasks (peg insertion, cable routing, etc.)

**What you need to add:**
- Your robot's interface (if not Franka)
- Your task definition (reward function)
- Your demonstrations

---

## 2. Repository Structure Deep-Dive

### High-Level Organization

```
serl/
├── serl_launcher/          # Core RL library (task-agnostic)
├── franka_sim/             # Simulation environment
├── serl_robot_infra/       # Real robot infrastructure
├── examples/               # Task-specific training scripts
└── docs/                   # Documentation
```

### serl_launcher: The Core RL Library

This is the **heart of SERL** - reusable RL components:

```
serl_launcher/
├── agents/
│   └── continuous/
│       ├── drq.py          # DrQ agent (image-based RL)
│       ├── sac.py          # SAC agent (state-based RL)
│       ├── vice.py         # VICE agent (research)
│       └── bc.py           # Behavior cloning (supervised)
│
├── networks/
│   ├── actor_critic_nets.py   # Policy and critic networks
│   ├── mlp.py                  # Multi-layer perceptron
│   ├── lagrange.py             # Lagrange multipliers
│   ├── classifier.py           # Binary classifier
│   └── reward_classifier.py    # Learned reward functions
│
├── data/
│   ├── replay_buffer.py        # Standard replay buffer
│   ├── memory_efficient_replay_buffer.py  # Optimized version
│   ├── data_store.py           # Data storage abstraction
│   └── dataset.py              # Dataset utilities
│
├── vision/
│   ├── small_encoders.py       # Lightweight CNN encoders
│   ├── resnet_v1.py           # ResNet encoders
│   └── data_augmentations.py  # Image augmentation (random crop)
│
├── wrappers/
│   ├── serl_obs_wrappers.py   # Observation formatting
│   ├── chunking.py            # Action/obs chunking
│   └── remap.py               # Action space remapping
│
├── common/
│   ├── common.py              # Common utilities
│   ├── encoding.py            # Observation encoding
│   ├── evaluation.py          # Policy evaluation
│   ├── optimizers.py          # Optimizer config
│   ├── typing.py              # Type definitions
│   └── wandb.py               # Weights & Biases logging
│
└── utils/
    ├── jax_utils.py           # JAX utilities
    ├── launcher.py            # Agent creation helpers
    ├── timer_utils.py         # Performance timing
    ├── train_utils.py         # Training utilities
    └── sim_utils.py           # Simulation utilities
```

**Key insight:** `serl_launcher` knows **nothing** about specific robots or tasks. It's pure RL algorithms and utilities.

### franka_sim: Simulation Environment

```
franka_sim/
├── franka_sim/
│   ├── __init__.py            # Gym env registration
│   ├── mujoco_gym_env.py      # Base MuJoCo gym env
│   │
│   ├── controllers/
│   │   └── opspace.py         # Operational space controller
│   │
│   ├── envs/
│   │   ├── panda_pick_gym_env.py  # Pick cube task
│   │   ├── utils.py               # Environment utilities
│   │   └── xmls/
│   │       └── arena.xml          # MuJoCo scene definition
│   │
│   └── test/
│       ├── test_gym_env_human.py     # Manual control test
│       └── test_gym_env_render.py    # Rendering test
│
├── setup.py
└── requirements.txt
```

**Purpose:** Provides a working simulation for prototyping and testing before moving to real hardware.

### serl_robot_infra: Real Robot Interface

```
serl_robot_infra/
├── robot_servers/
│   ├── franka_server.py       # Main robot control server
│   ├── franka_gripper_server.py   # Gripper-specific server
│   ├── gripper_server.py      # Generic gripper interface
│   └── robotiq_gripper_server.py  # Robotiq gripper support
│
└── franka_env/
    ├── __init__.py            # Real robot gym registration
    │
    ├── envs/
    │   ├── franka_env.py      # Base real robot env
    │   ├── peg_env/           # Peg insertion task
    │   │   ├── __init__.py
    │   │   ├── config.py      # Task configuration
    │   │   └── franka_peg_env.py
    │   ├── pcb_env/           # PCB insertion task
    │   ├── cable_env/         # Cable routing task
    │   └── bin_relocation_env/  # Object relocation
    │
    ├── camera/
    │   └── rs_capture.py      # RealSense camera interface
    │
    ├── spacemouse/
    │   └── spacemouse_expert.py   # SpaceMouse teleoperation
    │
    └── utils/
        ├── rotations.py       # Rotation utilities
        └── server_interface.py  # Robot server communication
```

**Purpose:** Bridges gym interface to real Franka robot via ROS + Flask server.

### examples: Training Scripts

```
examples/
├── async_drq_sim/             # Image-based sim training
│   ├── async_drq_sim.py       # Main training script
│   ├── run_learner.sh         # Launch learner node
│   ├── run_actor.sh           # Launch actor node
│   └── tmux_launch.sh         # One-liner launcher
│
├── async_sac_state_sim/       # State-based sim training
│   ├── async_sac_state_sim.py
│   ├── run_learner.sh
│   ├── run_actor.sh
│   └── tmux_launch.sh
│
├── async_peg_insert_drq/      # Real robot: peg insertion
│   ├── async_drq_randomized.py
│   ├── record_demo.py         # Demo collection
│   ├── run_learner.sh
│   └── run_actor.sh
│
├── async_pcb_insert_drq/      # Real robot: PCB insertion
├── async_cable_route_drq/     # Real robot: cable routing
│
└── async_bin_relocation_fwbw_drq/  # Dual policy (forward/backward)
    ├── async_drq_randomized.py
    ├── record_demo.py
    ├── record_bc_demos.py     # BC-specific demos
    ├── record_transitions.py   # For reward classifier
    ├── train_reward_classifier.py  # Train learned reward
    ├── run_fw_learner.sh      # Forward policy learner
    ├── run_bw_learner.sh      # Backward policy learner
    └── run_actor.sh           # Dual-policy actor
```

**Key point:** Each example is **self-contained** - copy and modify for your task!

---

## 3. Core Architecture: Actor-Learner Pattern

### Why Actor-Learner?

Traditional RL pseudocode:
```python
for episode in range(num_episodes):
    obs = env.reset()
    while not done:
        action = policy(obs)           # 1. Inference
        next_obs, reward, done = env.step(action)
        buffer.add(obs, action, reward, next_obs)
        policy.update(buffer.sample())  # 2. Training
```

**Problem:** Training blocks environment interaction! Wastes time.

**Solution:** Separate into two processes:

```python
# === ACTOR PROCESS ===
while True:
    action = policy(obs)        # Fast inference
    next_obs, reward, done = env.step(action)
    send_to_learner(transition)  # Send data
    policy = get_updated_policy()  # Async receive

# === LEARNER PROCESS ===
while True:
    batch = receive_from_actor()  # Async receive
    replay_buffer.add(batch)
    policy.update(replay_buffer.sample())  # Train
    send_policy_to_actor(policy)  # Async send
```

**Benefit:** Actor always collecting data while learner always training!

### AgentLace: The Communication Library

SERL uses [AgentLace](https://github.com/youliangtan/agentlace) for actor-learner communication.

```python
# === Learner Side ===
from agentlace.trainer import TrainerServer

server = TrainerServer(
    data_store=replay_buffer,
    config=trainer_config,
)

server.register_data_callback(lambda data: replay_buffer.insert(data))
server.register_param_callback(lambda: agent.state.params)
server.start()  # Start listening

# === Actor Side ===
from agentlace.trainer import TrainerClient

client = TrainerClient(
    agent_name="actor_env",
    server_ip="192.168.1.100",  # Learner IP
    config=trainer_config,
    data_store=local_data_store,
)

# Send data
client.update()  # Sends queued transitions

# Receive policy
client.recv_network_callback(update_params)  # Async callback
```

**Key features:**
- Works over network (actor and learner on different machines)
- Asynchronous (non-blocking)
- Handles serialization automatically

### Full Training Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         LEARNER NODE                            │
│                    (GPU Workstation)                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            TrainerServer (AgentLace)                     │  │
│  │  ┌────────────────────┐      ┌─────────────────────┐   │  │
│  │  │ Receive Data       │      │ Send Policy         │   │  │
│  │  │ from Actor         │      │ to Actor            │   │  │
│  │  └─────────┬──────────┘      └──────────▲──────────┘   │  │
│  └────────────┼─────────────────────────────┼──────────────┘  │
│               ▼                             │                  │
│  ┌────────────────────────┐    ┌───────────┴──────────┐       │
│  │   Demo Buffer          │    │   Agent (Policy)     │       │
│  │   (Fixed, 10-20 demos) │    │   - Actor Network    │       │
│  └────────────┬───────────┘    │   - Critic Network   │       │
│               │                │   - Temperature      │       │
│  ┌────────────▼───────────┐    └───────────▲──────────┘       │
│  │   Replay Buffer        │                │                  │
│  │   (Growing, online)    │                │                  │
│  └────────────┬───────────┘                │                  │
│               │                            │                  │
│               └────────► Sample Batch ─────┘                  │
│                          (50% demo, 50% online)               │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Network (ZMQ)
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│                         ACTOR NODE                              │
│                  (Robot or Sim Environment)                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │            TrainerClient (AgentLace)                     │  │
│  │  ┌────────────────────┐      ┌─────────────────────┐   │  │
│  │  │ Send Data          │      │ Receive Policy      │   │  │
│  │  │ to Learner         │      │ from Learner        │   │  │
│  │  └─────────▲──────────┘      └──────────┬──────────┘   │  │
│  └────────────┼─────────────────────────────┼──────────────┘  │
│               │                             ▼                  │
│  ┌────────────┴───────────┐    ┌───────────────────────┐      │
│  │   QueuedDataStore      │    │   Agent (Policy)      │      │
│  │   (Small buffer)       │    │   (Synced from        │      │
│  └────────────▲───────────┘    │    Learner)           │      │
│               │                └───────────┬───────────┘      │
│               │                            │                  │
│               │                            ▼                  │
│               │                ┌──────────────────────┐       │
│               │                │   action = policy(obs)│       │
│               │                └──────────┬───────────┘       │
│               │                           │                   │
│               │                           ▼                   │
│  ┌────────────┴─────────────────────────────────────────┐    │
│  │             Gym Environment                          │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │  obs, reward, done = env.step(action)       │    │    │
│  │  │  transition = (obs, action, reward, ...)    │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                                                       │    │
│  │  Real Robot OR MuJoCo Simulation                    │    │
│  └───────────────────────────────────────────────────────┘    │
│                                                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow and Communication

### Startup Sequence

**Step 1: Launch Learner**
```bash
$ bash run_learner.sh
```

What happens:
```python
1. Load config and create environment (for observation space)
2. Create agent (DrQAgent.create_drq(...))
3. Initialize replay buffers (demo + online)
4. Load demos into demo_buffer
5. Start TrainerServer on port 5488
6. Enter training loop:
   while True:
       batch = sample_batch(demo_buffer, replay_buffer)
       agent = agent.update(batch)
       # Policy automatically synced to actors
```

**Step 2: Launch Actor**
```bash
$ bash run_actor.sh
```

What happens:
```python
1. Create environment (real robot or sim)
2. Create agent (same architecture as learner)
3. Connect to learner via TrainerClient
4. Register callback to receive policy updates
5. Enter rollout loop:
   while True:
       action = agent.sample_actions(obs)
       next_obs, reward, done = env.step(action)
       data_store.insert(transition)
       if step % update_freq == 0:
           client.update()  # Send data to learner
```

### Data Flow Timeline

```
Time →

Learner:  [Init] ──[Wait]── [Recv Data] ─[Train]─[Send Policy]─[Train]─...
                      ▲                               │
                      │                               │
Actor:    [Init] ──[Connect]──[Rollout]──[Send]──[Recv]──[Rollout]──...
                                  ▼                   ▲
Environment:              [step] [step] [step]  [step] [step] ...
```

### Message Types

**Actor → Learner:**
```python
{
    "observations": {...},    # Current state
    "actions": [...],         # Action taken
    "next_observations": {...},  # Resulting state
    "rewards": float,         # Reward received
    "masks": 1.0,            # (1 - done)
    "dones": bool            # Episode termination
}
```

**Learner → Actor:**
```python
{
    "params": {               # Neural network parameters
        "actor": {...},
        "critic": {...},
        "temperature": {...},
    }
}
```

### Synchronization Strategy

**Periodic sync** (not every step):
- Actor sends data every `steps_per_update` steps (default: 30)
- Learner sends policy every `N` training updates
- Reduces network overhead

**Why this works:**
- Policy changes slowly (small learning rates)
- Actor can use slightly old policy without issues
- Massive speedup from asynchrony

---

## 5. MuJoCo Simulation Environment

### What is MuJoCo?

**MuJoCo** = Multi-Joint dynamics with Contact

- Physics engine for robotics simulation
- Fast, accurate contact dynamics
- Industry standard for RL research

### The Simulation Stack

```
┌─────────────────────────────────────┐
│     Gym Environment                  │  ← Standard RL interface
│     (PandaPickCubeGymEnv)           │
├─────────────────────────────────────┤
│     MujocoGymEnv (Base Class)       │  ← SERL's wrapper
│     - Handles rendering             │
│     - Control loop                   │
│     - Observation collection         │
├─────────────────────────────────────┤
│     MuJoCo Python Bindings          │  ← mujoco-py or mujoco
│     - Load XML model                │
│     - Physics simulation            │
│     - Contact solver                │
├─────────────────────────────────────┤
│     MuJoCo Engine (C++)             │  ← Core physics
│     - Dynamics integration          │
│     - Collision detection           │
│     - Constraint solver             │
└─────────────────────────────────────┘
```

### MuJoCo XML Model

**Location:** `franka_sim/franka_sim/envs/xmls/arena.xml`

```xml
<mujoco model="franka_arena">
  <!-- Assets: meshes, textures -->
  <asset>
    <mesh file="franka_panda/link0.stl"/>
    <mesh file="franka_panda/link1.stl"/>
    <!-- ... more links ... -->
  </asset>

  <!-- Scene setup -->
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    
    <!-- Robot arm -->
    <body name="panda_link0" pos="0 0 0">
      <inertial mass="0.629" pos="0 0 0"/>
      <geom type="mesh" mesh="link0"/>
      
      <joint name="joint1" type="hinge" axis="0 0 1"/>
      <body name="panda_link1">
        <!-- Nested bodies for kinematic chain -->
      </body>
    </body>
    
    <!-- Cube to manipulate -->
    <body name="block" pos="0.4 0 0.02">
      <geom name="block" type="box" size="0.02 0.02 0.02" 
            rgba="1 0 0 1" mass="0.05"/>
      <freejoint/>  <!-- 6-DOF free body -->
    </body>
    
    <!-- Target (visual only) -->
    <body name="target" pos="0.4 0 0.3">
      <geom type="sphere" size="0.03" rgba="0 1 0 0.3" 
            contype="0" conaffinity="0"/>  <!-- No collision -->
    </body>
  </worldbody>

  <!-- Actuators -->
  <actuator>
    <motor joint="joint1" gear="1" ctrlrange="-87 87"/>
    <motor joint="joint2" gear="1" ctrlrange="-87 87"/>
    <!-- ... 7 joints total ... -->
    <motor joint="finger_joint1" gear="1"/>  <!-- Gripper -->
  </actuator>

  <!-- Cameras for rendering -->
  <camera name="front_camera" pos="0.8 0 0.5" 
          euler="0 0 0" fovy="45"/>
  <camera name="wrist_camera" pos="0.05 0 0.05" 
          quat="..."/>  <!-- Attached to end-effector -->
</mujoco>
```

### Base Class: MujocoGymEnv

**File:** `franka_sim/franka_sim/mujoco_gym_env.py`

```python
class MujocoGymEnv(gym.Env):
    def __init__(
        self,
        xml_path: Path,
        control_dt: float = 0.02,  # 50 Hz control
        physics_dt: float = 0.002, # 500 Hz physics
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        # Load MuJoCo model
        self._model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self._model)
        
        # Control frequency
        self.control_dt = control_dt
        self.physics_dt = physics_dt
        self._steps_per_control = int(control_dt / physics_dt)
        
        # Rendering setup
        self._renderer = mujoco.Renderer(
            self._model,
            height=render_spec.height,
            width=render_spec.width,
        )
    
    def step(self, action):
        # Apply action
        self._apply_action(action)
        
        # Step physics multiple times (substeps)
        for _ in range(self._steps_per_control):
            mujoco.mj_step(self._model, self._data)
        
        # Collect observation
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_termination(obs)
        
        return obs, reward, done, False, {}
    
    def reset(self):
        # Reset simulation state
        mujoco.mj_resetData(self._model, self._data)
        self._initialize_episode()
        return self._get_obs(), {}
    
    def render(self):
        # Render current state
        self._renderer.update_scene(self._data)
        return self._renderer.render()
```

### Task-Specific: PandaPickCubeGymEnv

**File:** `franka_sim/franka_sim/envs/panda_pick_gym_env.py`

```python
class PandaPickCubeGymEnv(MujocoGymEnv):
    def __init__(self, image_obs: bool = False, **kwargs):
        super().__init__(xml_path=_XML_PATH, **kwargs)
        
        self.image_obs = image_obs
        
        # Cache MuJoCo IDs for fast access
        self._panda_dof_ids = [
            self._model.joint(f"joint{i}").id 
            for i in range(1, 8)
        ]
        self._gripper_ctrl_id = self._model.actuator(
            "fingers_actuator"
        ).id
        self._pinch_site_id = self._model.site("pinch").id
        
        # Define observation space
        if self.image_obs:
            self.observation_space = gym.spaces.Dict({
                "images": {
                    "front": Box(0, 255, (128,128,3), uint8),
                    "wrist": Box(0, 255, (128,128,3), uint8),
                },
                "state": {
                    "tcp_pos": Box(-inf, inf, (3,)),
                    "tcp_vel": Box(-inf, inf, (3,)),
                    "gripper_pos": Box(-inf, inf, (1,)),
                }
            })
        else:
            # State-only observation
            self.observation_space = gym.spaces.Dict({
                "state": {
                    "panda/tcp_pos": Box(-inf, inf, (3,)),
                    "panda/tcp_vel": Box(-inf, inf, (3,)),
                    "panda/gripper_pos": Box(-inf, inf, (1,)),
                    "block_pos": Box(-inf, inf, (3,)),
                }
            })
        
        # Action space: [dx, dy, dz, droll, dpitch, dyaw, gripper]
        self.action_space = Box(
            low=np.array([-1, -1, -1, -1]),
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32,
        )
    
    def _get_obs(self):
        # Get end-effector pose
        tcp_pos = self._data.site_xpos[self._pinch_site_id]
        tcp_vel = self._data.site_xvelp[self._pinch_site_id]
        
        # Get gripper state
        gripper_pos = self._data.qpos[self._gripper_dof_ids]
        
        # Get block pose
        block_pos = self._data.body("block").xpos
        
        obs = {"state": {
            "panda/tcp_pos": tcp_pos.copy(),
            "panda/tcp_vel": tcp_vel.copy(),
            "panda/gripper_pos": gripper_pos.copy(),
            "block_pos": block_pos.copy(),
        }}
        
        if self.image_obs:
            # Render cameras
            front_img = self._render_camera(camera_id=0)
            wrist_img = self._render_camera(camera_id=1)
            obs["images"] = {
                "front": front_img,
                "wrist": wrist_img,
            }
        
        return obs
    
    def _apply_action(self, action):
        # action: [dx, dy, dz, gripper] in [-1, 1]
        # Scale to real units
        delta_pos = action[:3] * self._action_scale[0]  # e.g., 0.1m
        gripper_cmd = action[3] * self._action_scale[1]
        
        # Current TCP pose
        current_pos = self._data.site_xpos[self._pinch_site_id]
        current_quat = self._data.site_xquat[self._pinch_site_id]
        
        # Target pose
        target_pos = current_pos + delta_pos
        target_quat = current_quat  # No rotation change
        
        # Inverse kinematics (operational space control)
        target_joints = self._controller.compute_ik(
            target_pos, target_quat
        )
        
        # Set actuator controls
        self._data.ctrl[self._panda_ctrl_ids] = target_joints
        self._data.ctrl[self._gripper_ctrl_id] = gripper_cmd
    
    def _compute_reward(self, obs, action):
        # Simple distance-based reward
        tcp_pos = obs["state"]["panda/tcp_pos"]
        block_pos = obs["state"]["block_pos"]
        
        # Distance to block
        dist = np.linalg.norm(tcp_pos - block_pos)
        
        # Sparse reward: 1 if block above threshold
        if block_pos[2] > 0.15:  # Block lifted
            return 1.0
        else:
            return 0.0
    
    def _initialize_episode(self):
        # Reset robot to home position
        self._data.qpos[self._panda_dof_ids] = _PANDA_HOME
        
        # Random block position
        block_xy = self.np_random.uniform(
            low=[0.25, -0.25],
            high=[0.55, 0.25],
        )
        self._data.joint("block_joint").qpos[:2] = block_xy
        self._data.joint("block_joint").qpos[2] = 0.02  # On table
        
        # Let physics settle
        for _ in range(100):
            mujoco.mj_step(self._model, self._data)
```

### Gym Environment Registration

**File:** `franka_sim/franka_sim/__init__.py`

```python
from gym.envs.registration import register

register(
    id="PandaPickCube-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": False},  # State-based
)

register(
    id="PandaPickCubeVision-v0",
    entry_point="franka_sim.envs:PandaPickCubeGymEnv",
    max_episode_steps=100,
    kwargs={"image_obs": True},  # Image-based
)
```

**Usage:**
```python
import gym
import franka_sim

env = gym.make("PandaPickCubeVision-v0")
obs, _ = env.reset()
# obs = {"images": {...}, "state": {...}}
```

---

## 6. Gym Environment Design

### The Gym Interface

Standard RL interface in robotics:

```python
class gym.Env:
    def reset(self) -> Tuple[Observation, Dict]:
        """Start new episode, return initial observation"""
        pass
    
    def step(self, action) -> Tuple[Obs, float, bool, bool, Dict]:
        """Execute action, return (obs, reward, terminated, truncated, info)"""
        pass
    
    @property
    def observation_space(self) -> gym.Space:
        """Description of observation format"""
        pass
    
    @property
    def action_space(self) -> gym.Space:
        """Description of action format"""
        pass
```

### Real Robot Gym Environment

**File:** `serl_robot_infra/franka_env/envs/franka_env.py`

```python
class FrankaEnv(gym.Env):
    """Base class for real Franka robot environments"""
    
    def __init__(self, 
                 server_ip: str = "localhost",
                 server_port: int = 5000,
                 camera_ids: List[str] = ["front", "wrist"],
                 **kwargs):
        
        self.server_url = f"http://{server_ip}:{server_port}"
        self.camera_ids = camera_ids
        
        # Initialize cameras
        self.cameras = {
            cam_id: RealSenseCamera(serial=cam_serial)
            for cam_id, cam_serial in zip(camera_ids, CAMERA_SERIALS)
        }
        
        # Define spaces (task-specific in subclass)
        self.observation_space = ...
        self.action_space = ...
    
    def reset(self):
        # Move robot to reset position
        response = requests.post(
            f"{self.server_url}/reset",
            json={"joint_positions": RESET_JOINTS}
        )
        
        # Capture initial observation
        obs = self._get_obs()
        return obs, {}
    
    def step(self, action):
        # Send action to robot server
        response = requests.post(
            f"{self.server_url}/step",
            json={
                "action": action.tolist(),
                "action_space": "cartesian_velocity"
            }
        )
        
        # Get new observation
        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        done = self._check_done(obs)
        
        return obs, reward, done, False, {}
    
    def _get_obs(self):
        # Get robot state from server
        state_response = requests.get(
            f"{self.server_url}/get_state"
        )
        robot_state = state_response.json()
        
        # Capture camera images
        images = {
            cam_id: cam.get_frame()
            for cam_id, cam in self.cameras.items()
        }
        
        return {
            "images": images,
            "state": robot_state,
        }
```

### Robot Server

**File:** `serl_robot_infra/robot_servers/franka_server.py`

```python
from flask import Flask, request, jsonify
import rospy
from franka_msgs.msg import FrankaState
from std_msgs.msg import Float64MultiArray

app = Flask(__name__)

class FrankaRobotServer:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("franka_server")
        
        # Publishers
        self.cmd_pub = rospy.Publisher(
            "/cartesian_impedance_controller/desired_pose",
            PoseStamped,
            queue_size=1
        )
        
        # Subscribers
        self.state_sub = rospy.Subscriber(
            "/franka_state_controller/franka_states",
            FrankaState,
            self._state_callback
        )
        
        self.current_state = None
    
    def _state_callback(self, msg):
        self.current_state = {
            "tcp_pos": msg.O_T_EE[:3, 3].tolist(),
            "tcp_quat": R.from_matrix(msg.O_T_EE[:3, :3]).as_quat(),
            "joint_pos": msg.q.tolist(),
            "joint_vel": msg.dq.tolist(),
        }

@app.route("/step", methods=["POST"])
def step():
    data = request.json
    action = np.array(data["action"])
    
    # Convert action to robot command
    server.execute_action(action)
    
    return jsonify({"success": True})

@app.route("/get_state", methods=["GET"])
def get_state():
    return jsonify(server.current_state)

if __name__ == "__main__":
    server = FrankaRobotServer()
    app.run(host="0.0.0.0", port=5000)
```

**Communication flow:**
```
Gym Env (Python) <---HTTP---> Flask Server (Python) <---ROS---> Robot (C++)
```

---

## 7. Observation and Action Spaces

### Observation Space Design

**Philosophy:** Include everything the policy needs, nothing more.

#### State-Based Observation

```python
observation = {
    "state": {
        # End-effector state
        "tcp_pos": np.array([x, y, z]),        # 3D position
        "tcp_vel": np.array([vx, vy, vz]),     # 3D velocity
        "tcp_quat": np.array([qw, qx, qy, qz]), # Orientation
        
        # Gripper state
        "gripper_pos": np.array([width]),      # Gripper opening
        
        # Optional: Joint state
        "joint_pos": np.array([7,]),           # Joint angles
        "joint_vel": np.array([7,]),           # Joint velocities
        
        # Task-specific
        "block_pos": np.array([x, y, z]),      # Object position
        "target_pos": np.array([x, y, z]),     # Goal position
    }
}
```

#### Image-Based Observation

```python
observation = {
    "images": {
        "front": np.array([128, 128, 3], dtype=uint8),  # RGB image
        "wrist": np.array([128, 128, 3], dtype=uint8),  # Wrist camera
        # Can have multiple cameras
    },
    "state": {
        # Usually still include proprioceptive state
        "tcp_pos": np.array([3]),
        "gripper_pos": np.array([1]),
    }
}
```

**Why both images and state?**
- Images: See the world (objects, configuration)
- State: Precise proprioception (where am I?)
- Combining both works best!

### Action Space Design

#### Delta Position Control (Most Common)

```python
action = np.array([
    dx,        # Change in x (m)
    dy,        # Change in y (m)
    dz,        # Change in z (m)
    droll,     # Change in roll (rad)
    dpitch,    # Change in pitch (rad)
    dyaw,      # Change in yaw (rad)
    gripper,   # Gripper command [-1, 1]
])
# All actions in range [-1, 1], scaled internally
```

**Why deltas, not absolute?**
- Easier for policy to learn (small corrections)
- More stable (bounded changes per step)
- Natural for impedance control

#### Joint Control (Alternative)

```python
action = np.array([7,])  # Target joint positions or velocities
```

Less common in SERL (harder to learn).

### Gym Space Definitions

```python
# Observation space
from gym import spaces

observation_space = spaces.Dict({
    "images": spaces.Dict({
        "front": spaces.Box(
            low=0, high=255, 
            shape=(128, 128, 3), 
            dtype=np.uint8
        ),
    }),
    "state": spaces.Dict({
        "tcp_pos": spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(3,),
            dtype=np.float32
        ),
    })
})

# Action space
action_space = spaces.Box(
    low=np.array([-1, -1, -1, -1]),
    high=np.array([1, 1, 1, 1]),
    dtype=np.float32
)
```

---

## 8. Environment Wrappers

### What Are Wrappers?

Wrappers modify environment behavior **without changing the core env**:

```python
env = gym.make("PandaPickCubeVision-v0")  # Base env
env = SERLObsWrapper(env)                  # Format observations
env = ChunkingWrapper(env)                 # Add chunking
env = RecordEpisodeStatistics(env)         # Track metrics
```

**Benefits:**
- Composable (stack multiple wrappers)
- Reusable across tasks
- Clean separation of concerns

### Key SERL Wrappers

#### 1. SERLObsWrapper

**File:** `serl_launcher/serl_launcher/wrappers/serl_obs_wrappers.py`

**Purpose:** Standardize observation format for RL algorithms

```python
class SERLObsWrapper(gym.ObservationWrapper):
    """Converts gym obs dict to flat structure expected by agents"""
    
    def observation(self, obs):
        # Input: nested dict from env
        # Output: flattened structure
        
        serl_obs = {}
        
        # Extract images if present
        if "images" in obs:
            for img_key, img in obs["images"].items():
                serl_obs[img_key] = img  # Keep as is
        
        # Flatten state dict
        if "state" in obs:
            # Convert nested dict to flat array
            state_vec = []
            for key in sorted(obs["state"].keys()):
                state_vec.append(obs["state"][key].flatten())
            serl_obs["state"] = np.concatenate(state_vec)
        
        return serl_obs
```

**Before:**
```python
obs = {
    "images": {"front": array([128,128,3])},
    "state": {
        "tcp_pos": array([3]),
        "gripper": array([1])
    }
}
```

**After:**
```python
obs = {
    "front": array([128,128,3]),
    "state": array([4])  # Concatenated
}
```

#### 2. ChunkingWrapper

**File:** `serl_launcher/serl_launcher/wrappers/chunking.py`

**Purpose:** Implement action/observation chunking for temporal coherence

```python
class ChunkingWrapper(gym.Wrapper):
    """
    Observation chunking: Stack past N observations
    Action chunking: Execute sequence of M actions
    """
    
    def __init__(self, env, obs_horizon=1, act_exec_horizon=None):
        super().__init__(env)
        self.obs_horizon = obs_horizon  # How many past obs to stack
        self.act_exec_horizon = act_exec_horizon  # Action sequence length
        
        self.obs_history = deque(maxlen=obs_horizon)
    
    def reset(self):
        obs, info = self.env.reset()
        # Initialize history with copies of first obs
        self.obs_history.clear()
        for _ in range(self.obs_horizon):
            self.obs_history.append(obs)
        return self._get_chunked_obs(), info
    
    def step(self, action):
        # If action chunking, execute sequence
        if self.act_exec_horizon:
            for i in range(self.act_exec_horizon):
                obs, reward, done, truncated, info = self.env.step(
                    action[i]
                )
                if done or truncated:
                    break
        else:
            obs, reward, done, truncated, info = self.env.step(action)
        
        # Update observation history
        self.obs_history.append(obs)
        
        return self._get_chunked_obs(), reward, done, truncated, info
    
    def _get_chunked_obs(self):
        # Stack observations along time dimension
        chunked = {}
        for key in self.obs_history[0].keys():
            if key == "images" or "image" in key:
                # Stack images: (T, H, W, C)
                chunked[key] = np.stack([
                    obs[key] for obs in self.obs_history
                ])
            else:
                # Stack state: (T, D)
                chunked[key] = np.stack([
                    obs[key] for obs in self.obs_history
                ])
        return chunked
```

**Why chunking?**
- Observation: Temporal context (velocity implicitly)
- Action: Smoother execution, reduce high-frequency oscillations

#### 3. SpacemouseIntervention

**File:** `serl_robot_infra/franka_env/spacemouse_intervention.py`

**Purpose:** Allow human to take control during episodes

```python
class SpacemouseIntervention(gym.Wrapper):
    """Allows human intervention via spacemouse"""
    
    def __init__(self, env):
        super().__init__(env)
        self.spacemouse = SpaceMouse()
        self.intervention_mode = False
    
    def step(self, action):
        # Check if spacemouse is active
        sm_action, buttons = self.spacemouse.get_action()
        
        if buttons["intervention"]:  # E.g., right button pressed
            self.intervention_mode = True
            action = sm_action  # Override policy action
            print("🎮 Human intervening!")
        elif buttons["release"]:
            self.intervention_mode = False
            print("🤖 Policy resumed")
        
        return self.env.step(action)
```

**Use cases:**
- Demo collection
- Online corrections during training
- Safety (stop dangerous actions)

#### 4. RecordEpisodeStatistics

**File:** (from gym.wrappers)

**Purpose:** Track episode returns and lengths

```python
from gym.wrappers import RecordEpisodeStatistics

env = RecordEpisodeStatistics(env)

obs, _ = env.reset()
# ... run episode ...
# After episode ends, info contains:
# info = {
#     "episode": {
#         "r": 10.5,    # Total return
#         "l": 50,      # Episode length
#         "t": 12.3,    # Time elapsed
#     }
# }
```

---

*[End of Part 2: Environment & Simulation]*

---

## Part 3: Algorithms & Training

## 9. RL Algorithms Explained

### SAC (Soft Actor-Critic)

**File:** `serl_launcher/serl_launcher/agents/continuous/sac.py`

**Algorithm:** Entropy-regularized off-policy RL

**Key idea:** Maximize both reward AND entropy (exploration)

```
Objective: maximize E[∑ reward + α * entropy]
          = maximize E[∑ reward - α * log π(a|s)]
```

**Components:**
1. **Actor (Policy)**: π(a|s) - chooses actions
2. **Critic (Q-function)**: Q(s,a) - estimates value
3. **Temperature**: α - controls exploration

**Update equations:**

```python
# Critic update
Q_target = r + γ * (Q'(s', a') - α * log π(a'|s'))
L_critic = (Q(s,a) - Q_target)²

# Actor update
L_actor = E[α * log π(a|s) - Q(s,a)]

# Temperature update
L_temp = E[-α * (log π(a|s) + target_entropy)]
```

**Why SAC?**
- Stable (off-policy)
- Sample efficient (reuses data)
- Automatic exploration (entropy term)

### DrQ (Data-Regularized Q-Learning)

**File:** `serl_launcher/serl_launcher/agents/continuous/drq.py`

**Algorithm:** SAC + data augmentation for images

**Key addition:** Random crop augmentation during training

```python
def update(agent, batch):
    # Augment images
    batch["images"] = random_crop(
        batch["images"],
        crop_size=(100, 100),  # From 128x128
    )
    
    # Standard SAC update with augmented data
    agent = sac_update(agent, batch)
    return agent
```

**Why DrQ?**
- Dramatically improves sample efficiency for vision
- Forces policy to be robust to small visual changes
- Essentially free regularization

**Data augmentation:**
```python
def batched_random_crop(images, crop_size):
    """
    images: (B, H, W, C)
    crop_size: (h, w)
    """
    B, H, W, C = images.shape
    h, w = crop_size
    
    # Random crop location per image
    top = np.random.randint(0, H - h, size=B)
    left = np.random.randint(0, W - w, size=B)
    
    cropped = jax.vmap(lambda img, t, l: 
        jax.lax.dynamic_slice(
            img, 
            (t, l, 0),  # Start
            (h, w, C)   # Size
        )
    )(images, top, left)
    
    return cropped
```

### RLPD (RL with Prior Data)

**Not a separate algorithm** - it's a **training recipe**:

1. Use demos in replay buffer
2. Sample 50% from demos, 50% from online
3. Train with DrQ/SAC

```python
# In learner loop
if demo_buffer is not None:
    demo_batch = demo_buffer.sample(batch_size // 2)
    online_batch = replay_buffer.sample(batch_size // 2)
    batch = concat_batches(demo_batch, online_batch)
else:
    batch = replay_buffer.sample(batch_size)

agent = agent.update(batch)
```

**Why RLPD?**
- Bootstraps from demos (faster learning)
- Still learns online (improves beyond demos)
- Simple to implement

### Behavior Cloning (BC)

**File:** `examples/bc_policy.py`

**Algorithm:** Supervised learning from demonstrations

```
Loss = ||policy(s) - action_demo||²
```

**Advantages:**
- Very simple
- No reward function needed
- Quick to train

**Disadvantages:**
- Can't improve beyond demos
- Brittle (distribution shift)
- No exploration

**When to use:**
- Baseline comparison
- Initialization for RL
- When you have many perfect demos

---

## 10. Neural Network Architecture

### Overall Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DrQ Agent                            │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Input: observation = {"images": ..., "state": ...}    │
│                                                          │
│  ┌────────────────────────────────────────────────┐    │
│  │              Encoder Network                    │    │
│  │  ┌──────────────┐         ┌─────────────────┐ │    │
│  │  │ Image Encoder│         │  State Encoder  │ │    │
│  │  │  (ResNet or  │         │     (MLP)       │ │    │
│  │  │   SmallCNN)  │         │                 │ │    │
│  │  │              │         │                 │ │    │
│  │  │ (128,128,3)  │         │     (state_dim) │ │    │
│  │  │      ↓       │         │        ↓        │ │    │
│  │  │   (256,)     │         │      (256,)     │ │    │
│  │  └──────┬───────┘         └────────┬────────┘ │    │
│  │         └──────────┬───────────────┘          │    │
│  │                    ↓                           │    │
│  │            Concatenate(256 + 256)              │    │
│  │                    ↓                           │    │
│  │               Embedding (512,)                 │    │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                 │
│         ┌─────────────┴─────────────┐                  │
│         │                           │                  │
│  ┌──────▼──────┐             ┌──────▼──────┐          │
│  │ Actor (Policy)│            │Critic (Q)   │          │
│  │               │            │             │          │
│  │  MLP(512,256) │            │ MLP(512,256)│          │
│  │     ↓         │            │     ↓       │          │
│  │  MLP(256,256) │            │ MLP(256,256)│          │
│  │     ↓         │            │     ↓       │          │
│  │ Output(act_dim)│           │ Output(1)   │          │
│  │               │            │             │          │
│  │ μ(s), σ(s)    │            │  Q(s,a)     │          │
│  └───────────────┘            └─────────────┘          │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

### Image Encoders

#### SmallEncoder (Lightweight CNN)

**File:** `serl_launcher/serl_launcher/vision/small_encoders.py`

```python
class SmallEncoder(nn.Module):
    features: Tuple[int] = (32, 64, 128, 256)
    kernel_sizes: Tuple[int] = (3, 3, 3, 3)
    strides: Tuple[int] = (2, 2, 2, 2)
    
    @nn.compact
    def __call__(self, x):
        # x: (B, H, W, C) = (B, 128, 128, 3)
        
        for feat, kernel, stride in zip(
            self.features, self.kernel_sizes, self.strides
        ):
            x = nn.Conv(
                features=feat,
                kernel_size=(kernel, kernel),
                strides=(stride, stride),
                padding="VALID",
            )(x)
            x = nn.relu(x)
            # After each: H,W → (H-kernel+1)/stride
        
        # x: (B, 4, 4, 256) after 4 conv layers
        
        # Global average pooling
        x = jnp.mean(x, axis=(1, 2))  # (B, 256)
        
        return x
```

**Output:** 256-dim embedding

#### ResNet (Pretrained)

**File:** `serl_launcher/serl_launcher/vision/resnet_v1.py`

```python
# Load pretrained weights
resnet_params = pickle.load(open("resnet10_params.pkl", "rb"))

class PretrainedResNetEncoder(nn.Module):
    @nn.compact
    def __call__(self, x, train=False):
        # x: (B, 128, 128, 3)
        
        # Forward through ResNet-10
        x = resnet_v1(
            x,
            num_classes=0,  # No classification head
            pretrained_params=resnet_params,
        )
        # x: (B, 512) embedding
        
        # Optional: Fine-tune or freeze
        if not train:
            x = jax.lax.stop_gradient(x)  # Freeze
        
        return x
```

**When to use:**
- ResNet: Better features, but slower
- SmallEncoder: Faster, less compute

### Actor Network (Policy)

```python
class Policy(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    
    @nn.compact
    def __call__(self, observations):
        # observations: encoded embedding (512,)
        
        x = observations
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Output mean and log_std
        mean = nn.Dense(self.action_dim)(x)
        log_std = nn.Dense(self.action_dim)(x)
        log_std = jnp.clip(log_std, -20, 2)  # Stability
        
        return mean, log_std
    
    def sample(self, observations, rng):
        mean, log_std = self(observations)
        std = jnp.exp(log_std)
        
        # Reparameterization trick
        noise = jax.random.normal(rng, shape=mean.shape)
        action = mean + std * noise
        
        # Tanh squashing to [-1, 1]
        action = jnp.tanh(action)
        
        # Compute log probability
        log_prob = gaussian_log_prob(noise, log_std)
        log_prob -= jnp.log(1 - action**2 + 1e-6).sum(-1)
        
        return action, log_prob
```

### Critic Network (Q-function)

```python
class Critic(nn.Module):
    hidden_dims: Sequence[int]
    
    @nn.compact
    def __call__(self, observations, actions):
        # Concatenate observation and action
        x = jnp.concatenate([observations, actions], axis=-1)
        
        for hidden_dim in self.hidden_dims:
            x = nn.Dense(hidden_dim)(x)
            x = nn.relu(x)
        
        # Output Q-value (scalar)
        q = nn.Dense(1)(x)
        return jnp.squeeze(q, -1)
```

**Ensemble:** Train multiple critics (default: 2) and take minimum:

```python
q1 = critic1(obs, action)
q2 = critic2(obs, action)
q_value = jnp.minimum(q1, q2)  # Pessimistic estimate
```

**Why ensemble?**
- Reduces overestimation bias
- More stable training

---

## 11. Training Loop Mechanics

### Learner Loop

**File:** `examples/async_drq_sim/async_drq_sim.py` (learner function)

```python
def learner(rng, agent, replay_buffer, demo_buffer=None):
    # Setup
    wandb_logger = make_wandb_logger(...)
    server = TrainerServer(...)
    server.register_data_callback(
        lambda data: replay_buffer.insert(data)
    )
    server.start()
    
    # Training loop
    for step in tqdm.tqdm(range(FLAGS.max_steps)):
        
        # Wait for enough data
        if len(replay_buffer) < FLAGS.training_starts:
            time.sleep(0.1)
            continue
        
        # Sample batch
        if demo_buffer and len(demo_buffer) > 0:
            # RLPD: Mix demos and online data
            demo_batch = demo_buffer.sample(FLAGS.batch_size // 2)
            online_batch = replay_buffer.sample(FLAGS.batch_size // 2)
            batch = concat_batches(demo_batch, online_batch)
        else:
            batch = replay_buffer.sample(FLAGS.batch_size)
        
        # Update agent (multiple critic updates per actor update)
        for _ in range(FLAGS.critic_actor_ratio):
            rng, update_rng = jax.random.split(rng)
            agent, update_info = agent.update(batch, update_rng)
        
        # Log metrics
        if step % FLAGS.log_period == 0:
            wandb_logger.log(update_info, step=step)
        
        # Save checkpoint
        if step % FLAGS.checkpoint_period == 0:
            checkpoints.save_checkpoint(
                FLAGS.checkpoint_path,
                agent.state,
                step=step,
            )
```

### Actor Loop

```python
def actor(agent, data_store, env, sampling_rng):
    # Setup client
    client = TrainerClient(...)
    client.recv_network_callback(
        lambda params: agent.replace(state=agent.state.replace(params=params))
    )
    
    obs, _ = env.reset()
    done = False
    
    for step in tqdm.tqdm(range(FLAGS.max_steps)):
        # Sample action
        if step < FLAGS.random_steps:
            action = env.action_space.sample()  # Random
        else:
            sampling_rng, key = jax.random.split(sampling_rng)
            action = agent.sample_actions(
                observations=obs,
                seed=key,
                deterministic=False,  # Stochastic for exploration
            )
            action = np.asarray(jax.device_get(action))
        
        # Step environment
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Store transition
        transition = dict(
            observations=obs,
            actions=action,
            next_observations=next_obs,
            rewards=reward,
            masks=1.0 - done,
            dones=done or truncated,
        )
        data_store.insert(transition)
        
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
        
        # Periodically sync with learner
        if step % FLAGS.steps_per_update == 0:
            client.update()  # Send data, receive policy
        
        # Periodic evaluation
        if step % FLAGS.eval_period == 0:
            eval_info = evaluate(agent, eval_env, num_episodes=5)
            client.request("send-stats", {"eval": eval_info})
```

### Agent Update (Single Step)

**In DrQAgent:**

```python
def update(agent, batch, rng):
    # batch = {
    #     "observations": ...,
    #     "actions": ...,
    #     "next_observations": ...,
    #     "rewards": ...,
    #     "masks": ...  (1 - done)
    # }
    
    rng, aug_rng = jax.random.split(rng)
    
    # 1. Augment images (DrQ)
    batch = augment_batch(batch, aug_rng)
    
    # 2. Update critic
    rng, critic_rng = jax.random.split(rng)
    agent, critic_info = update_critic(agent, batch, critic_rng)
    
    # 3. Update actor
    rng, actor_rng = jax.random.split(rng)
    agent, actor_info = update_actor(agent, batch, actor_rng)
    
    # 4. Update temperature
    agent, temp_info = update_temperature(agent, batch)
    
    # 5. Update target networks (EMA)
    agent = agent.replace(
        state=agent.state.replace(
            target_params=ema_update(
                agent.state.params,
                agent.state.target_params,
                tau=0.005,  # Soft update rate
            )
        )
    )
    
    return agent, {**critic_info, **actor_info, **temp_info}
```

---

## 12. Replay Buffer System

### Memory-Efficient Replay Buffer

**File:** `serl_launcher/serl_launcher/data/memory_efficient_replay_buffer.py`

**Key insight:** Store images as JPEG to save RAM!

```python
class MemoryEfficientReplayBufferDataStore:
    def __init__(self, obs_space, action_space, capacity, image_keys):
        self.capacity = capacity
        self.size = 0
        self.insert_index = 0
        
        # Preallocate arrays
        self.observations = {}
        for key, space in obs_space.spaces.items():
            if key in image_keys:
                # Store as bytes (JPEG)
                self.observations[key] = np.empty(
                    (capacity,), dtype=object
                )
            else:
                # Store as numpy array
                self.observations[key] = np.empty(
                    (capacity, *space.shape), dtype=space.dtype
                )
        
        self.actions = np.empty(
            (capacity, *action_space.shape), dtype=action_space.dtype
        )
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.masks = np.empty((capacity,), dtype=np.float32)
    
    def insert(self, transition):
        idx = self.insert_index
        
        # Store observation
        for key, value in transition["observations"].items():
            if key in self.image_keys:
                # Compress image to JPEG
                _, encoded = cv2.imencode(".jpg", value)
                self.observations[key][idx] = encoded.tobytes()
            else:
                self.observations[key][idx] = value
        
        self.actions[idx] = transition["actions"]
        self.rewards[idx] = transition["rewards"]
        self.masks[idx] = transition["masks"]
        
        # Circular buffer
        self.insert_index = (self.insert_index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {}
        for key in self.observations.keys():
            if key in self.image_keys:
                # Decode JPEG
                images = []
                for idx in indices:
                    img_bytes = self.observations[key][idx]
                    img = cv2.imdecode(
                        np.frombuffer(img_bytes, np.uint8),
                        cv2.IMREAD_COLOR
                    )
                    images.append(img)
                batch[key] = np.stack(images)
            else:
                batch[key] = self.observations[key][indices]
        
        batch["actions"] = self.actions[indices]
        batch["rewards"] = self.rewards[indices]
        batch["masks"] = self.masks[indices]
        
        return batch
```

**Memory savings:**
- Raw: 128×128×3 = 49KB per image
- JPEG: ~5-10KB per image
- 5-10x compression!

---

*This completes the main sections. The guide now covers foundation, environment, algorithms, and training mechanics comprehensively.*

Would you like me to:
1. Add sections on Real Robot Infrastructure and Reward Engineering?
2. Add sections on Extending SERL for custom robots?
3. Create a final summary document?

