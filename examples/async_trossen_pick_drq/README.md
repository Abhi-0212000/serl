# Trossen WidowX AI Pick-and-Place with DrQ

This example demonstrates training a DrQ agent on the Trossen WidowX AI robot simulation to perform a pick-and-place task.

## Files

- `async_trossen_drq.py`: Main training script with actor and learner loops
- `run_actor.sh`: Launch script for the actor (collects experience)
- `run_learner.sh`: Launch script for the learner (trains the policy)

## Quick Start

### 1. Install Dependencies

```bash
# Install trossen_sim package
cd ../../trossen_sim
pip install -e .

# Install serl_launcher
cd ../serl_launcher
pip install -e .
```

### 2. Test the Environment

```bash
cd ../../trossen_sim
python trossen_sim/test/test_gym_env.py
```

### 3. Run Training

You need to run both the learner and actor in separate terminals:

**Terminal 1 (Learner):**
```bash
cd examples/async_trossen_pick_drq
bash run_learner.sh
```

**Terminal 2 (Actor):**
```bash
cd examples/async_trossen_pick_drq
bash run_actor.sh
```

## Configuration

Key hyperparameters in the scripts:

- `--seed`: Random seed
- `--batch_size`: Training batch size
- `--random_steps`: Number of random exploration steps
- `--training_starts`: Start training after this many steps
- `--critic_actor_ratio`: Critic to actor update ratio
- `--encoder_type`: Image encoder type (small, resnet-pretrained, mobilenet)
- `--debug`: Debug mode (disables wandb logging)

## Environment

- **Task**: Pick up a cube and lift it 15cm
- **Action Space**: [dx, dy, dz, gripper] - Delta position commands + gripper
- **Observation**: Robot state (TCP position, velocity, gripper) + block position
- **Reward**: Combination of proximity to block (30%) and lift height (70%)

## Notes

- The actor renders the environment by default (set `--render False` to disable)
- Training uses the DrQ algorithm with data augmentation
- The learner and actor communicate via AgentLace
- Checkpoints can be saved with `--checkpoint_period` and `--checkpoint_path`
