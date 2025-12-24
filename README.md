# GLGymMPC
Model Predictive Control applied to GreenLight-Gym


```bash
python visualizations/performance.py
    --load_rl
    --load_rlmpc
    --model_name trim-durian-32
    --rlmpc_folders full-rollout-fixed-region-norm-x-0.01 full-rollout-fixed-region-norm-x-0.02
    --plot_all 
    --save
    --output_folder full-rollout-fixed-region-norm-x-0.01vs0.02
    --labels RL RL-MPC-full-0.01 RL-MPC-full-0.02 
```

```bash
python experiments/rlmpc_experiment.py
    --project GL-MPC-RL
    --env_id TomatoEnv
    --algorithm ppo
    --model_name trim-durian-32
    --horizon 1
    --region_range 0.01
    --terminal_constraint
    --terminal_penalty
    --normalize_x
    --frame_stack
    --n_stack 2
    --normalize_x
    --experiment_name full-rollout-fixed-region-norm-x-0.01
```


```bash
python experiments/evaluate_rl.py
    --project GL-MPC-RL
    --env_id TomatoEnv
    --model_name trim-durian-32
    --algorithm ppo
    --uncertainty_scale 0
    --mode deterministic
    --frame_stack
    --n_stack 2
```