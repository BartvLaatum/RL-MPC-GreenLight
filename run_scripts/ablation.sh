#!/bin/bash
export PYTHONPATH=$(pwd)

model_name="daily-glade-107"
region_range=0.05
n_stack=2
horizon=1
growth_year=2023

experiment_name="rlmpc-switch-daily-glade-107-0.05-no-terminal-constraint"

python experiments/rlmpc_experiment.py \
    --project GL-MPC-RL \
    --env_id TomatoEnv \
    --algorithm ppo \
    --model_name "$model_name" \
    --horizon "$horizon" \
    --region_range "$region_range" \
    --terminal_penalty \
    --normalize_x \
    --frame_stack \
    --n_stack "$n_stack" \
    --normalize_x \
    --experiment_name "$experiment_name" \
    --growth_year "$growth_year" \
    --start_day 151 \
    --plant_state_env \
    --selector_mechanism \
    # --terminal_constraint \

experiment_name="rlmpc-switch-daily-glade-107-0.05-no-terminal-pen"

python experiments/rlmpc_experiment.py \
    --project GL-MPC-RL \
    --env_id TomatoEnv \
    --algorithm ppo \
    --model_name "$model_name" \
    --horizon "$horizon" \
    --region_range "$region_range" \
    --terminal_constraint \
    --normalize_x \
    --frame_stack \
    --n_stack "$n_stack" \
    --normalize_x \
    --experiment_name "$experiment_name" \
    --growth_year "$growth_year" \
    --start_day 151 \
    --plant_state_env \
    --selector_mechanism \
    # --terminal_penalty \

experiment_name="rlmpc-switch-daily-glade-107-0.05-no-selector-mechanism"

python experiments/rlmpc_experiment.py \
    --project GL-MPC-RL \
    --env_id TomatoEnv \
    --algorithm ppo \
    --model_name "$model_name" \
    --horizon "$horizon" \
    --region_range "$region_range" \
    --terminal_penalty \
    --terminal_constraint \
    --normalize_x \
    --frame_stack \
    --n_stack "$n_stack" \
    --normalize_x \
    --experiment_name "$experiment_name" \
    --growth_year "$growth_year" \
    --start_day 151 \
    --plant_state_env \
    # --selector_mechanism \




