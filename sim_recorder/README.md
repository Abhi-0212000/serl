# Sim Recorder

**Self-contained** data collection for SERL robot learning: Record demos from **MuJoCo simulation** controlled by **real robot teleoperation**.

## 🎯 Quick Start

**Option 1: Simple recording (no web UI):**

```bash
cd sim_recorder/examples
python teleop_with_server.py --no-visualize
# Data collection runs in background, cameras stream to web UI
```

**Option 2: With MuJoCo visualization:**

```bash
cd sim_recorder/examples  
python teleop_with_server.py
# Opens MuJoCo viewer + web UI at http://localhost:5000
```

**What it does:**
1. Connects to real leader robots (dual robot support)
2. Loads MuJoCo simulation (self-contained XML + meshes)
3. [Optional] Shows MuJoCo viewer
4. Streams 4 cameras to web UI in real-time
5. Records: camera views + robot states + actions
6. Saves in SERL-compatible format

## 📦 Self-Contained Package

This `sim_recorder/` folder is **completely independent** - it includes:

- ✅ **MuJoCo XML files**: `assets/trossen_ai_scene_joint.xml`, `assets/trossen_ai_joint.xml`
- ✅ **3D meshes & textures**: `assets/meshes/` (all STL files + PNG textures)
- ✅ **Web UI**: Complete Flask server + HTML/JS/CSS interface
- ✅ **Recording logic**: Camera capture, state recording, data export
- ✅ **Dual robot support**: Control two sim robots with two real leaders

**External dependencies** (install separately):
- `mujoco` (MuJoCo physics engine)
- `trossen_arm` (real robot interface)
- Python packages in `requirements.txt`

No other SERL folders required!

## How It Works

```
Real Leader Robot (192.168.1.2)
       ↓ (read joint positions)
   [Actions: 7D joint commands]
       ↓
MuJoCo Simulation
   • Apply actions to sim robot
   • Step physics
   • Render 4 cameras (128x128)
   • Get robot state (qpos, qvel)
       ↓
Save Episode Data
   • observations.npz (cameras + states)
   • actions.npy (leader commands)
   • meta.json (episode info)
```

## What Gets Recorded

Each episode folder contains:

```
recorded_episodes/episode_20240115_143022/
├── observations.npz         # All observations
│   ├── cam_high: (T, 128, 128, 3)      # High camera view
│   ├── cam_low: (T, 128, 128, 3)       # Low camera view  
│   ├── cam_left_wrist: (T, 128, 128, 3)  # Left wrist camera
│   ├── cam_right_wrist: (T, 128, 128, 3) # Right wrist camera
│   ├── qpos: (T, 8)         # Joint positions (6 arm + 2 gripper)
│   └── qvel: (T, 8)         # Joint velocities
├── actions.npy              # Actions from leader robot (T, 7)
└── meta.json                # Metadata (num_steps, duration, FPS, etc.)
```

Where `T` = number of timesteps in the episode.

## Usage Examples

### Basic Recording

```bash
# Simple: Record for 60 seconds
python examples/integrated_teleop_recorder.py --duration 60 --auto-record

# With web UI for monitoring cameras
python examples/integrated_teleop_recorder.py --duration 60 --auto-record --web-ui
# Open http://localhost:5000 in browser to see cameras

# Specify save location
python examples/integrated_teleop_recorder.py \
    --duration 90 \
    --save-dir my_demos \
    --auto-record

# Higher control frequency (30 Hz instead of 20 Hz)
python examples/integrated_teleop_recorder.py \
    --control-freq 30.0 \
    --auto-record

# Different leader robot IP
python examples/integrated_teleop_recorder.py \
    --leader-ip 192.168.1.5 \
    --auto-record
```

### Inspect Recorded Data

```bash
# List all episodes
python examples/inspect_episode.py recorded_episodes/

# Show episode details
python examples/inspect_episode.py recorded_episodes/episode_20240115_143022/

# Play episode as video
python examples/inspect_episode.py recorded_episodes/episode_20240115_143022/ \
    --play --camera cam_high --fps 20

# Show all 4 cameras in grid
python examples/inspect_episode.py recorded_episodes/episode_20240115_143022/ \
    --show-cameras
```

### Convert to SERL Pickle Format

```bash
# Convert all episodes to pickle format for SERL training
python examples/convert_to_serl_pickle.py recorded_episodes/ \
    --output-dir serl_demos
```

## Requirements

```bash
# Hardware needed:
- Real Trossen leader robot (for teleoperation)
- Computer with MuJoCo installed

# Software:
pip install mujoco numpy trossen_arm opencv-python
```

## Configuration

Edit `examples/integrated_teleop_recorder.py` if you need to customize:

```python
# Camera configuration
camera_names = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
camera_resolution = (128, 128)  # (Height, Width)

# Control frequency
control_freq = 20.0  # Hz (20 Hz default)

# Robot model
model = 'wxai_v0'  # Trossen robot model

# MuJoCo XML path
XML_PATH = Path(...) / "trossen_ai_scene_joint.xml"
```

## Troubleshooting

**Leader robot connection fails:**
```bash
# Check if robot is reachable
ping 192.168.1.2

# Verify robot is powered on and in correct mode
# Use --leader-ip flag if different IP
python integrated_teleop_recorder.py --leader-ip 192.168.1.5
```

**MuJoCo XML not found:**
```bash
# sim_recorder is self-contained - XML is in assets/
ls assets/trossen_ai_scene_joint.xml
ls assets/meshes/  # Should contain all STL and texture files

# If missing, the XML path resolution will fall back to:
# ../trossen_sim/trossen_sim/envs/xmls/trossen_ai_scene_joint.xml
```

**Cameras rendering black/wrong:**
```bash
# Check MuJoCo installation
python -c "import mujoco; print(mujoco.__version__)"

# Verify camera names in XML match script
grep "<camera" trossen_sim/trossen_sim/envs/xmls/trossen_ai_scene_joint.xml
```

**Recording is slow/laggy:**
```bash
# Reduce control frequency
python integrated_teleop_recorder.py --control-freq 10.0 --auto-record

# Or edit script to reduce camera resolution to 64x64
camera_resolution = (64, 64)

# Or record fewer cameras
camera_names = ['cam_high']  # Just one camera
```

## File Structure

```
sim_recorder/
├── README.md                          # This file
├── QUICKSTART_INTEGRATED.md           # Detailed guide
├── requirements.txt                   # Dependencies
├── assets/                            # Self-contained MuJoCo assets
│   ├── trossen_ai_scene_joint.xml     # Main scene XML
│   ├── trossen_ai_joint.xml           # Robot definition XML
│   └── meshes/                        # All 3D meshes & textures
├── examples/
│   ├── teleop_with_server.py         # Main dual-robot teleop script
│   ├── convert_to_serl_pickle.py     # Convert to pickle format
│   └── inspect_episode.py            # Visualize episodes
├── server/                            # Web UI server
│   ├── app.py                         # Flask REST API
│   ├── cameras.py                     # Camera buffer management
│   ├── recorder.py                    # Recording engine
│   └── teleop_ingest.py               # ZeroMQ teleop listener
├── ui/                                # Web interface
│   ├── index.html                     # Camera monitoring UI
│   ├── main.js                        # JavaScript controls
│   └── styles.css                     # UI styling
└── tests/                             # Unit tests
    └── test_integrated_recorder.py
```

## Using Recorded Data for Training

```python
import numpy as np

# Load episode
episode_dir = "recorded_episodes/episode_20240115_143022"

# Load observations
obs = np.load(f"{episode_dir}/observations.npz")
cam_high = obs['cam_high']        # (T, 128, 128, 3)
qpos = obs['qpos']                # (T, 8)
qvel = obs['qvel']                # (T, 8)

# Load actions
actions = np.load(f"{episode_dir}/actions.npy")  # (T, 7)

# Use for training (BC, RL, etc.)
for t in range(len(actions)):
    image = cam_high[t]
    state = np.concatenate([qpos[t], qvel[t]])
    action = actions[t]
    # ... feed to your training loop
```

## Advanced: Web UI Server (Optional)

The `--web-ui` flag automatically starts a Flask server in the background for real-time camera monitoring:

```bash
python examples/integrated_teleop_recorder.py --duration 60 --auto-record --web-ui
```

Then open browser to `http://localhost:5000` to see:
- 4 camera streams updating in real-time
- Recording status (idle/recording)
- Episode name and step count
- Start/Stop recording buttons (if not using --auto-record)

The server automatically shuts down when the script exits.

## Step-by-Step Guide

See [RECORDING_GUIDE.md](RECORDING_GUIDE.md) for detailed step-by-step instructions.

## License

Same license as parent SERL project.
