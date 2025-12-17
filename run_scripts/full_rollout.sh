#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    python experiments/rlmpc_experiment.py \
        --horizon "$horizon" \
        --terminal_constraint \
        --terminal_penalty \
        --region_range 0.025 \
        --experiment_name full-rollout-frame-stack-terminal-penalty-0.025 \
        --frame_stack \
        --n_stack 2 \
        --model_name trim-durian-32
done
