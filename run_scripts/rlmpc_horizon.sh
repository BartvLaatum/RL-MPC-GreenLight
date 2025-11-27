#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    python experiments/rlmpc_experiments.py --horizon "$horizon" --method exact --terminal_constraint --region_range 0.025 --extend_ocp_region --experiment_name extend_ocp_region_tight_regions
    # python experiments/rlmpc_experiments.py --horizon "$horizon" --method exact --terminal_constraint --region_range 0.025 --offline_rl --experiment_name offline-rl-trajectory-tight-region
    # python experiments/rlmpc_experiments.py --horizon "$horizon" --method exact --terminal_constraint --region_range 0.025 --experiment_name rollout-current-state-tight-region
done
