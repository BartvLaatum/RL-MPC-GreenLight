#!/bin/bash
export PYTHONPATH=$(pwd)
group="n-stacked"
python experiments/train_rl.py --group limited-obs --n_eval_episodes 1 --n_evals 20 --save_model --save_env --frame_stack --n_stack 2