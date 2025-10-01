#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    python experiments/rlmpc_experiments.py --horizon "$horizon" --method exact --terminal_constraint
    # python experiments/rlmpc_experiments.py --horizon "$horizon" --method finite-difference #--terminal_constraint
done
