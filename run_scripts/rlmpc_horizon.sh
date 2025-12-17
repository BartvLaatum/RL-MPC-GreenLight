#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --region_range 0.05 \
    #     --experiment_name full-rollout-frame-stack \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30 \


    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --region_range 0.05 \
    #     --experiment_name offline-frame-stack \
    #     --offline_rl \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30 \

    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --extend_ocp_region \
    #     --region_range 0.05 \
    #     --experiment_name extend-ocp-frame-stack \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30 \

    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --terminal_penalty \
    #     --region_range 0.05 \
    #     --experiment_name full-rollout-frame-stack-terminal-penalty \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30 \

    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --terminal_penalty \
    #     --extend_ocp_region \
    #     --region_range 0.05 \
    #     --experiment_name extend-ocp-frame-stack-terminal-penalty \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30 \

    python experiments/rlmpc_experiment.py \
        --horizon "$horizon" \
        --terminal_constraint \
        --terminal_penalty \
        --extend_ocp_region \
        --region_range 0.025 \
        --experiment_name extend-ocp-frame-stack-terminal-penalty-0.025 \
        --frame_stack \
        --n_stack 2 \
        --model_name trim-durian-32 \


    python experiments/rlmpc_experiment.py \
        --horizon "$horizon" \
        --terminal_constraint \
        --terminal_penalty \
        --region_range 0.025 \
        --experiment_name full-rollout-frame-stack-terminal-penalty-0.025 \
        --frame_stack \
        --n_stack 2 \
        --model_name trim-durian-32

    # python experiments/rlmpc_experiment.py \
    #     --horizon "$horizon" \
    #     --terminal_constraint \
    #     --terminal_penalty \
    #     --region_range 0.025 \
    #     --experiment_name test \
    #     --frame_stack \
    #     --n_stack 2 \
    #     --model_name stilted-dust-30

done
