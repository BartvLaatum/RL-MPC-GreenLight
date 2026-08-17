#!/bin/bash
export PYTHONPATH=$(pwd)

# model_name="trim-durian-32"
# model_name="daily-glade-107"
model_name="legendary-salad-82"
experiment_name="rlmpc-switch-legendary-salad-82-0.05-fixed-norm_x"
region_range=0.05
n_stack=2
growth_years=(2011 2012 2019 2020 2023)
horizon=1

# Function to run experiments for a single growth year
run_year() {
    local growth_year=$1
    for start_day in 90 105 120 135 151; do

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
            --load_norm_x \
            --frame_stack \
            --n_stack "$n_stack" \
            --normalize_x \
            --experiment_name "$experiment_name" \
            --growth_year "$growth_year" \
            --start_day "$start_day" \
            --plant_state_env

        echo -e "Subject: Finished RL-MPC evaluation for year $growth_year\n
        \nFinished RL-MPC evaluation for year $growth_year with horizon $horizon at $(date).
        \nModel name: $model_name" | msmtp hbpvanlaatum123@gmail.com
    done
}

# Export function and variables for subshells
export -f run_year
export experiment_name model_name region_range n_stack horizon

# Run growth year in parallel
parallel --line-buffer -j 9 run_year ::: "${growth_years[@]}"

# Wait for background job to complete
wait
echo -e "Subject: Finished RL-MPC evaluation for all years\n
\nFinished RL-MPC evaluation for all years at $(date).
\nExperiment name: $experiment_name" | msmtp hbpvanlaatum123@gmail.com
