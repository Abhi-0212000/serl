# SERL Quick Start Guide (10-Minute Overview)

**Goal:** Get you from zero to running SERL training in 10 minutes of reading.

---

## What is SERL?

**SERL = Sample-Efficient Robotic Reinforcement Learning**

- Train robot manipulation policies **in 30-60 minutes**
- Works with **images** (cameras) or **state** (joint positions)
- Can train in **simulation** or on **real robots**
- Uses minimal demonstrations (10-20 trajectories)

---

## The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                     SERL Training Loop                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│   1. Record Demos (Teleoperation)                           │
│      └─> demos.pkl (10-20 trajectories)                     │
│                                                               │
│   2. Launch Learner Node (GPU)                              │
│      ├─> Loads demos into demo_buffer                       │
│      ├─> Receives data from actor                           │
│      └─> Updates policy with RL algorithm                   │
│                                                               │
│   3. Launch Actor Node (Robot/Sim)                          │
│      ├─> Runs policy on environment                         │
│      ├─> Collects new experiences                           │
│      └─> Sends data to learner                              │
│                                                               │
│   4. Policy Converges (30-60 min)                           │
│      └─> checkpoints/ saved every N steps                   │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

**Key Concept:** Actor and Learner run **asynchronously** on separate processes (or machines!)

---

## Repository Structure (What's Where?)

```
serl/
│
├── serl_launcher/               # Core RL library
│   ├── agents/continuous/       # RL algorithms
│   │   ├── drq.py              # DrQ (image-based)
│   │   ├── sac.py              # SAC (state-based)
│   │   ├── vice.py             # VICE (research)
│   │   └── bc.py               # Behavior cloning
│   ├── networks/                # Neural network models
│   ├── data/                    # Replay buffers
│   └── wrappers/                # Gym environment wrappers
│
├── franka_sim/                  # MuJoCo simulation
│   └── franka_sim/
│       ├── mujoco_gym_env.py   # Base simulation class
│       └── envs/
│           └── panda_pick_gym_env.py  # Pick cube task
│
├── serl_robot_infra/           # Real robot interface
│   ├── robot_servers/          # ROS → Flask server
│   │   └── franka_server.py   # Robot control server
│   └── franka_env/             # Real robot gym envs
│       └── envs/
│           ├── peg_env/        # Peg insertion
│           ├── pcb_env/        # PCB insertion
│           ├── cable_env/      # Cable routing
│           └── bin_relocation_env/  # Object relocation
│
└── examples/                   # Training scripts
    ├── async_drq_sim/          # Image training (sim)
    ├── async_sac_state_sim/    # State training (sim)
    ├── async_peg_insert_drq/   # Real robot: peg
    ├── async_pcb_insert_drq/   # Real robot: PCB
    ├── async_cable_route_drq/  # Real robot: cable
    └── async_bin_relocation_fwbw_drq/  # Real robot: dual policy
```

---

## Installation (5 Minutes)

```bash
# 1. Create conda environment
conda create -n serl python=3.10
conda activate serl

# 2. Install JAX (GPU)
pip install --upgrade "jax[cuda12_pip]==0.4.35" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

# 3. Install SERL launcher (RL algorithms)
cd serl_launcher
pip install -e .
pip install -r requirements.txt

# 4. Install Franka sim (for simulation)
cd ../franka_sim
pip install -e .
pip install -r requirements.txt

# 5. Test simulation
python franka_sim/franka_sim/test/test_gym_env_human.py
# Should open a MuJoCo window with Franka arm!
```

---

## Your First Training Run (Simulation)

### Option 1: One-Liner (Recommended)

```bash
cd examples/async_drq_sim
bash tmux_launch.sh  # Launches learner + actor in tmux

# Watch training in the terminal output
# Or check wandb dashboard (if configured)

# Kill when done:
tmux kill-session -t serl_session
```

**What happens:**
- Creates 2 tmux panes: learner (left), actor (right)
- Learner trains policy on GPU
- Actor collects data in MuJoCo simulation
- Policy converges in ~10-20 minutes

### Option 2: Manual (Two Terminals)

**Terminal 1 (Learner):**
```bash
cd examples/async_drq_sim
bash run_learner.sh
```

**Terminal 2 (Actor):**
```bash
cd examples/async_drq_sim
bash run_actor.sh
```

---

## Understanding the Training Scripts

### What's in `run_learner.sh`?

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false  # Don't hog all GPU memory
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2    # Use 20% of GPU

python async_drq_sim.py \
    --learner \                    # This is the learner node
    --exp_name=my_experiment \     # Experiment name (for wandb)
    --seed 0 \
    --batch_size 256 \             # Batch size for training
    --training_starts 1000 \       # Start training after 1000 steps
    --critic_actor_ratio 4 \       # 4 critic updates per actor update
    --encoder_type resnet-pretrained \  # Use pretrained ResNet
    --checkpoint_period 10000 \    # Save checkpoint every 10k steps
    --checkpoint_path ./checkpoints  # Where to save
    # --demo_path demos.pkl \      # Optional: load demos
    # --debug                      # Add this to disable wandb
```

### What's in `run_actor.sh`?

```bash
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1  # Actor needs less GPU

python async_drq_sim.py \
    --actor \                      # This is the actor node
    --render True \                # Show MuJoCo visualization
    --exp_name=my_experiment \
    --seed 0 \
    --random_steps 1000 \          # Random exploration for 1000 steps
    --eval_period 2000 \           # Evaluate every 2000 steps
    --eval_n_trajs 5               # Run 5 eval episodes
    # --ip x.x.x.x                 # If learner is on different machine
```

---

## Key Concepts

### 1. Actor vs Learner

| **Actor** | **Learner** |
|-----------|-------------|
| Runs policy on environment | Updates policy with RL |
| Collects transitions | Samples from replay buffer |
| Lightweight (inference) | Heavy (training) |
| Can be on real robot | Usually on workstation |

### 2. Replay Buffer Architecture

```python
┌────────────────────────────────────────────┐
│           Learner Process                   │
├────────────────────────────────────────────┤
│                                             │
│  ┌──────────────┐      ┌──────────────┐   │
│  │ Demo Buffer  │      │Replay Buffer │   │
│  │  (Fixed)     │      │  (Growing)   │   │
│  │ 10-20 demos  │      │ Online data  │   │
│  └──────┬───────┘      └──────┬───────┘   │
│         │                      │            │
│         └──────────┬───────────┘            │
│                    ▼                        │
│           Sample Batch (50% each)          │
│                    │                        │
│                    ▼                        │
│            Update Policy (RL)              │
│                                             │
└────────────────────────────────────────────┘
            │
            │ (Network sync)
            ▼
┌────────────────────────────────────────────┐
│            Actor Process                    │
├────────────────────────────────────────────┤
│  Policy → Action → Env → Transition        │
│           └─────────────────┘               │
│           Send to Learner                   │
└────────────────────────────────────────────┘
```

### 3. Observation Space (Images)

```python
observation = {
    "images": {
        "front": np.array([128, 128, 3]),   # Front camera RGB
        "wrist": np.array([128, 128, 3]),   # Wrist camera RGB
    },
    "state": {
        "tcp_pos": np.array([3]),           # End-effector position
        "tcp_vel": np.array([3]),           # End-effector velocity
        "gripper_pos": np.array([1]),       # Gripper width
    }
}
```

### 4. Action Space

```python
action = np.array([
    dx,        # Change in x position
    dy,        # Change in y position
    dz,        # Change in z position
    droll,     # Change in roll
    dpitch,    # Change in pitch
    dyaw,      # Change in yaw
    gripper,   # Gripper command (-1=open, 1=close)
])
# Shape: (7,)
```

**Important:** Actions are **deltas** (changes), not absolute positions!

---

## Available Algorithms

### DrQ (Image-Based)

**Use when:** Task needs camera images

```bash
cd examples/async_drq_sim
bash tmux_launch.sh
```

**How it works:**
- CNN encoder processes images
- SAC for policy optimization
- Random crop augmentation

### SAC (State-Based)

**Use when:** You have state info (joint angles, positions)

```bash
cd examples/async_sac_state_sim
bash tmux_launch.sh
```

**How it works:**
- MLP processes state vectors
- Faster training (no images)
- More sample efficient

### RLPD (RL + Prior Data)

**Use when:** You have good demonstrations

```bash
cd examples/async_drq_sim
bash run_learner.sh --demo_path demos.pkl  # Demos heavily weighted
bash run_actor.sh
```

---

## Recording Demonstrations

### Simulation

```bash
cd examples/async_drq_sim
python record_demo.py --output demos.pkl --n_episodes 20

# Control:
# - Use spacemouse OR keyboard
# - Press 's' to save trajectory
# - Collect 10-20 successful trajectories
```

### Real Robot

```bash
cd examples/async_peg_insert_drq
python record_demo.py

# Requires:
# - Franka robot running
# - franka_server.py active
# - SpaceMouse connected
```

---

## Monitoring Training

### Terminal Output

```
Step: 5000
  eval/return: 0.85
  eval/success_rate: 0.8
  training/actor_loss: 0.023
  training/critic_loss: 0.45
  timer/total: 1.2s
```

**Key metrics:**
- `eval/success_rate` - % of successful episodes (0-1)
- `eval/return` - Average return (higher = better)
- Training time per step

### Weights & Biases (Optional)

```bash
# Remove --debug flag in run_learner.sh
wandb login
bash run_learner.sh  # Now logs to wandb
```

Visit: https://wandb.ai to see live training curves

---

## Evaluation

```bash
# In run_actor.sh, add:
python async_drq_sim.py \
    --actor \
    --eval_checkpoint_step 50000 \     # Load checkpoint at step 50k
    --eval_n_trajs 20 \                # Run 20 episodes
    --checkpoint_path ./checkpoints

# Outputs success rate and returns
```

---

## Common Flags Explained

| Flag | Purpose | Typical Value |
|------|---------|---------------|
| `--batch_size` | Training batch size | 256 |
| `--replay_buffer_capacity` | Max transitions stored | 200000 |
| `--random_steps` | Random exploration steps | 300-1000 |
| `--training_starts` | When to start training | Same as random_steps |
| `--critic_actor_ratio` | Critic updates per actor update | 4 |
| `--encoder_type` | CNN architecture | `resnet-pretrained` or `small` |
| `--checkpoint_period` | Save frequency | 10000 |
| `--eval_period` | Evaluation frequency | 2000 |

---

## Troubleshooting

### Out of Memory (OOM)

```bash
# Reduce batch size
bash run_learner.sh --batch_size 64

# Or reduce replay buffer
bash run_learner.sh --replay_buffer_capacity 50000
```

### MuJoCo Rendering Error

```bash
# Set rendering backend
export MUJOCO_GL=glfw  # Or egl for headless
python async_drq_sim.py --actor --render False
```

### Training Not Improving

1. **Check reward function** - Is it returning correct values?
2. **Check demos** - Are they good quality?
3. **Increase training** - Try `--training_starts 2000`
4. **Check observation** - Is the state/image correct?

---

## Next Steps

### Run Real Robot Example

1. Set up Franka robot with ROS
2. Follow: [`docs/real_franka.md`](docs/real_franka.md)
3. Start with peg insertion (simplest task)

### Adapt to Your Robot

1. Create MuJoCo model (`.xml`)
2. Build gym environment (see `franka_sim/envs/`)
3. Modify training script (see `examples/`)
4. Train and iterate

### Multi-Robot Setup

1. Extend observation space (2 robots)
2. Extend action space (concat actions)
3. Design cooperative reward
4. Train with modified scripts

---

## Complete Example Workflow

```bash
# ===== SETUP =====
conda activate serl
cd serl/examples/async_drq_sim

# ===== TRAIN =====
# Terminal 1
bash run_learner.sh --checkpoint_path ./my_checkpoints

# Terminal 2
bash run_actor.sh

# Wait 10-20 minutes...

# ===== EVALUATE =====
# Edit run_actor.sh:
#   --eval_checkpoint_step 50000
#   --eval_n_trajs 20
bash run_actor.sh

# ===== RESULTS =====
# Check: eval/success_rate should be > 0.9
```

---

## Summary

| Concept | Key Point |
|---------|-----------|
| **What is SERL?** | Sample-efficient RL for robot manipulation |
| **Two nodes** | Actor (collects data) + Learner (trains policy) |
| **Data sources** | Demos + random exploration + policy rollouts |
| **Algorithms** | DrQ (images), SAC (state), RLPD (with demos) |
| **Training time** | 30-60 minutes for simple tasks |
| **Hardware** | Can train in sim or on real robot |

---

## Essential Commands Cheat Sheet

```bash
# Installation
conda create -n serl python=3.10
pip install "jax[cuda12_pip]==0.4.35" -f https://...
cd serl_launcher && pip install -e .
cd franka_sim && pip install -e .

# Training (Sim)
cd examples/async_drq_sim
bash tmux_launch.sh  # All-in-one
# OR
bash run_learner.sh  # Terminal 1
bash run_actor.sh    # Terminal 2

# Recording Demos
python record_demo.py --output demos.pkl

# Evaluation
bash run_actor.sh --eval_checkpoint_step 50000 --eval_n_trajs 20

# Kill Training
tmux kill-session -t serl_session
# OR just Ctrl+C in both terminals
```

---

**You're now ready to train your first SERL policy!**

For deeper understanding, see:
- [`UNDERSTANDING_SUMMARY.md`](UNDERSTANDING_SUMMARY.md) - Detailed Q&A
- [`SERL_COMPLETE_GUIDE.md`](SERL_COMPLETE_GUIDE.md) - Complete architecture deep-dive
- [`docs/sim_quick_start.md`](docs/sim_quick_start.md) - Official sim guide
- [`docs/real_franka.md`](docs/real_franka.md) - Real robot guide
