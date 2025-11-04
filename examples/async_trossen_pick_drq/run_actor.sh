#!/bin/bash

unset MUJOCO_GL && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.1 && \
python async_trossen_drq.py "$@" \
    --actor \
    --render True \
    --exp_name=serl_trossen_pick_drq \
    --seed 0 \
    --random_steps 1000 \
    --encoder_type resnet-pretrained \
    --debug
