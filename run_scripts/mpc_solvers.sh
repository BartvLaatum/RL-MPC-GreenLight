#!/bin/bash
export PYTHONPATH=$(pwd)

experiment_name="mpc_linear_solver_ma57_lower_humidity"
horizon=1
location="Netherlands"
# start_days=(90 105 120 135 151)
start_days=(90 151)
# growth_years=(2020 2023 2011 2012 2013 2014 2015 2016 2017 2018 2019)
growth_years=(2011 2012 2019 2020 2023)

run_year() {
    local growth_year=$1
    local start_day=$2
    python experiments/mpc_experiment.py \
        --horizon "$horizon" \
        --experiment_name "$experiment_name" \
        --location "$location" \
        --growth_year "$growth_year" \
        --start_day "$start_day" \
        --plant_state_env

    echo -e "Subject: Finished MPC evaluation for year $growth_year and start day $start_day\n
    \nFinished MPC evaluation for year $growth_year and start day $start_day with horizon $horizon at $(date).
    \nExperiment name: $experiment_name" | msmtp hbpvanlaatum123@gmail.com
}

# Export function and variables for subshells
export -f run_year
export experiment_name location horizon
# Run growth year in parallel
parallel --line-buffer -j 10 run_year ::: "${growth_years[@]}" ::: "${start_days[@]}"

# Wait for background job to complete
wait

echo -e "Subject: Finished MPC evaluation for all years and start days\n
\nFinished MPC evaluation for all years and start days at $(date).
\nExperiment name: $experiment_name" | msmtp hbpvanlaatum123@gmail.com
