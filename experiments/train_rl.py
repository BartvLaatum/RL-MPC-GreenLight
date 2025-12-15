import argparse
from common.utils import load_env_params, load_model_hyperparams
from experiments.rl_experiment_manager import RLExperimentManager

def main(args: argparse.Namespace):
    env_config_path = f"configs/envs/"
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    hyperparameters = load_model_hyperparams(args.algorithm, args.env_id)
    # Initialize the experiment manager
    experiment_manager = RLExperimentManager(
        env_id=args.env_id,
        project=args.project,
        env_base_params=env_base_params,
        env_specific_params=env_specific_params,
        hyperparameters=hyperparameters,
        group=args.group,
        n_eval_episodes=args.n_eval_episodes,
        n_evals=args.n_evals,
        algorithm=args.algorithm,
        env_seed=args.env_seed,
        model_seed=args.model_seed,
        stochastic=args.stochastic,
        save_model=args.save_model,
        save_env=args.save_env,
        hp_tuning=args.hyperparameter_tuning,
        device=args.device,
        frame_stack=args.frame_stack,
        n_stack=args.n_stack,
    )

    if args.hyperparameter_tuning:
        # Perform hyperparameter tuning
        experiment_manager.hyperparameter_tuning()
    else:
    # Run the experiment
        experiment_manager.run_experiment()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL", help="Wandb project name")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--algorithm", type=str, default="ppo", help="RL algorithm to use")
    parser.add_argument("--group", type=str, default="group1", help="Wandb group name")
    parser.add_argument("--n_eval_episodes", type=int, default=1, help="Number of episodes to evaluate the agent for")
    parser.add_argument("--n_evals", type=int, default=10, help="Number times we evaluate algorithm during training")
    parser.add_argument("--env_seed", type=int, default=666, help="Random seed for the environment for reproducibility")
    parser.add_argument("--model_seed", type=int, default=666, help="Random seed for the RL-model for reproducibility")
    parser.add_argument("--stochastic", action="store_true", help="Whether to run the experiment in stochastic mode")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the experiment on")
    parser.add_argument("--frame_stack", action="store_true", help="Whether to use frame stacking")
    parser.add_argument("--n_stack", type=int, default=1, help="Number of frames to stack")
    parser.add_argument("--save_model", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the model")
    parser.add_argument("--save_env", default=True, action=argparse.BooleanOptionalAction, help="Whether to save the environment")
    parser.add_argument("--hyperparameter_tuning", default=False, action=argparse.BooleanOptionalAction, help="Perform hyperparameter tuning")
    args = parser.parse_args()

    main(args)
