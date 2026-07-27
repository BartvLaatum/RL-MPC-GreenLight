#!/bin/bash
export PYTHONPATH=$(pwd)
group="n-stacked"
python experiments/train_rl.py \
    --group limited-obs \
    --n_eval_episodes 1 \
    --n_evals 20 \
    --save_model \
    --save_env \
    --frame_stack \
    --n_stack 2 \
    --plant_state_env \
    --hp_updates '{"batch_size": 64, "learning_rate": 5e-5, "n_steps": 512}'
