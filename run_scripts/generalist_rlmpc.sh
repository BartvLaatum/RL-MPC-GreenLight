#!/bin/bash
export PYTHONPATH=$(pwd)

# model_name="legendary-salad-82"
# experiment_name="rlmpc-switch-legendary-salad-82-0.05"
model_name="fresh-microwave-111"
experiment_name="rlmpc-switch-fresh-microwave-111-0.05"
region_range=0.05
n_stack=2
horizon=1
growth_years=(2011 2012 2019 2020 2023)
# start_days=(90 105 120 135 151)
start_days=(90 151)

# Function to run experiments for a single growth year AND start day
run_year() {
    local growth_year=$1
    local start_day=$2  # Add this - parallel passes it as second argument

    for horizon in 1; do
        echo "Running Generalist RL-MPC evaluation for year $growth_year and start day $start_day with horizon $horizon"
        python experiments/rlmpc_experiment.py \
            --project GL-MPC-RL \
            --env_id TomatoEnv \
            --algorithm ppo \
            --model_name "$model_name" \
            --horizon "$horizon" \
            --region_range "$region_range" \
            --terminal_constraint \
            --terminal_penalty \
            --normalize_x \
            --frame_stack \
            --n_stack "$n_stack" \
            --normalize_x \
            --experiment_name "$experiment_name" \
            --growth_year "$growth_year" \
            --start_day "$start_day" \
            --plant_state_env

        echo -e "Subject: Finished Generalist RL-MPC evaluation for year $growth_year and start day $start_day\n
        \nFinished Generalist RL-MPC evaluation for year $growth_year and start day $start_day with horizon $horizon at $(date).
        \nModel name: $model_name" | msmtp hbpvanlaatum123@gmail.com
    done
}

# Export function and variables for subshells
export -f run_year
export experiment_name model_name region_range n_stack

# Run all combinations in parallel (parallel creates Cartesian product)
parallel --line-buffer -j 10 run_year ::: "${growth_years[@]}" ::: "${start_days[@]}"
# Wait for background job to complete
wait
echo -e "Subject: Finished Generalist RL-MPC evaluation for all years\n
\nFinished Generalist RL-MPC evaluation for all years and start days at $(date).
\nModel name: $model_name" | msmtp hbpvanlaatum123@gmail.com
