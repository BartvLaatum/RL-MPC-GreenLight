#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    # python experiments/rlmpc_experiment.py --horizon "$horizon" --terminal_constraint --region_range 0.05 --extend_ocp_region --experiment_name linear_solver_mumps --linear_solver mumps
    python experiments/rlmpc_experiment.py --horizon "$horizon" --terminal_constraint --region_range 0.05 --extend_ocp_region --experiment_name linear_solver_ma57 --linear_solver ma57
    # python experiments/rlmpc_experiment.py --horizon "$horizon" --terminal_constraint --region_range 0.025 --offline_rl --experiment_name offline-rl-trajectory-tight-region
    # python experiments/rlmpc_experiment.py --horizon "$horizon" --terminal_constraint --region_range 0.025 --experiment_name rollout-current-state-tight-region
done