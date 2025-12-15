#!/bin/bash
export PYTHONPATH=$(pwd)

for horizon in 1
do
    # python experiments/mpc_experiment.py --horizon "$horizon" --linear_solver mumps --experiment_name mpc_linear_solver_mumps
    python experiments/mpc_experiment.py --horizon "$horizon" --experiment_name mpc_linear_solver_ma57 
done
