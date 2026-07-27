#!/bin/bash
export PYTHONPATH=$(pwd)

# model_name="legendary-salad-82"
# model_name="daily-glade-107"
model_name="fresh-microwave-111"
start_days=(90 151)
# test_years=(2020 2023 2011 2012 2013 2014 2015 2016 2017 2018 2019)
test_years=(2020 2023 2011 2012 2019)

for year in "${test_years[@]}"; do
    for start_day in "${start_days[@]}"; do
        echo "Running RL evaluation for year $year and start day $start_day"
        python experiments/evaluate_rl.py \
            --project GL-MPC-RL \
            --env_id TomatoEnv \
            --model_name "$model_name" \
            --algorithm ppo \
            --uncertainty_scale 0 \
            --mode deterministic \
            --frame_stack \
            --n_stack 2 \
            --growth_year "$year" \
            --start_day "$start_day" \
            --plant_state_env
    done
done
