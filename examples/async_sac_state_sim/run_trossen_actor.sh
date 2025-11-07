#!/bin/bash
# Trossen Bimanual Pick-and-Place Training - ACTOR
# This script starts the actor process for collecting experience

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.05 && \
python async_sac_state_sim.py "$@" \
    --actor \
    --render \
    --env TrossenBimanualPickPlace-v0 \
    --exp_name=trossen_bimanual_pick_place \
    --seed 42 \
    --random_steps 300 \
    --steps_per_update 30 \
    --eval_period 2000 \
    --eval_n_trajs 5 \
    --ip localhost \
    --debug

# To run actor on different machine, change --ip to learner's IP address
# Example: --ip 192.168.1.100

# To disable rendering (faster training):
# Change --render to --render false

# Notes:
# - Actor collects data and sends to learner every 30 steps
# - Rendering shows live simulation (can be slow)
# - For faster training without visualization, set --render false
# - Random exploration for first 300 steps, then uses learned policy
