#!/bin/bash

export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.2 && \
python async_trossen_drq.py "$@" \
    --learner \
    --exp_name=serl_trossen_pick_drq \
    --seed 0 \
    --batch_size 64 \
    --training_starts 1000 \
    --critic_actor_ratio 4 \
    --encoder_type resnet-pretrained \
    --debug
