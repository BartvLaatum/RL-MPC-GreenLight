#!/bin/bash
export PYTHONPATH=$(pwd)
# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 256, "learning_rate": 8.75e-5, "n_steps": 1024}'

# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 256, "learning_rate": 8.75e-5, "n_steps": 2048}'


# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 64, "learning_rate": 5e-5, "n_steps": 512}'



# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 32, "learning_rate": 5e-5, "n_steps": 288}'



# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 128, "learning_rate": 8.75e-5, "n_steps": 2048}'


# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 128, "learning_rate": 3e-4, "n_steps": 2048}'

# python experiments/train_rl.py \
#     --group limited-obs \
#     --n_eval_episodes 1 \
#     --n_evals 20 \
#     --save_model \
#     --save_env \
#     --frame_stack \
#     --n_stack 2 \
#     --plant_state_env \
#     --hp_updates '{"batch_size": 256, "learning_rate": 3e-4, "n_steps": 2048}'


python experiments/train_rl.py \
    --group limited-obs \
    --n_eval_episodes 1 \
    --n_evals 20 \
    --save_model \
    --save_env \
    --frame_stack \
    --n_stack 2 \
    --plant_state_env \
    # --hp_updates '{"batch_size": 64, "learning_rate": 5e-5, "n_steps": 864}'
