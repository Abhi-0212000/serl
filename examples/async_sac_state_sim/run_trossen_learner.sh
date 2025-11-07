#!/bin/bash
# Trossen Bimanual Pick-and-Place Training - LEARNER
# This script starts the learner process for training the Trossen dual-arm robot

# Get absolute path for checkpoint directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CHECKPOINT_DIR="${SCRIPT_DIR}/checkpoints/trossen_bimanual"

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --learner \
    --env TrossenBimanualPickPlace-v0 \
    --exp_name=trossen_bimanual_pick_place \
    --seed 42 \
    --max_steps 1000000 \
    --training_starts 300 \
    --random_steps 300 \
    --critic_actor_ratio 8 \
    --batch_size 256 \
    --checkpoint_period 10000 \
    --checkpoint_path "${CHECKPOINT_DIR}" \
    --eval_period 2000 \
    --log_period 10 \
    --debug  # Remove this line to enable wandb logging

# To enable wandb logging:
# Remove the --debug flag above and the model will log to wandb

# Notes:
# - 14D action space: [left_arm(6) + left_gripper(1) + right_arm(6) + right_gripper(1)]
# - Training starts after 300 random exploration steps
# - Checkpoints saved every 10000 steps to ./checkpoints/trossen_bimanual/
# - Evaluation runs every 2000 steps
