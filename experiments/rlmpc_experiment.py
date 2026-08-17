import os
import argparse

from stable_baselines3 import PPO

from controllers.rlmpc import RLMPC
from experiments.mpc_experiment_managers import RLMPCExperimentManager
from common.utils import load_model_hyperparams, load_env_params, load_env
from common.results import Results
from environments.utils import convert_rh_ppm

ALGS = {
    "ppo": PPO
    }

def main(args: argparse.Namespace):
    load_path = os.path.join("train_data", args.project, args.algorithm, "deterministic")
    save_dir = os.path.join("results", args.project, "deterministic", "rlmpc", args.experiment_name)
    rl_model_path = os.path.join(load_path, "models", args.model_name, "best_model.zip")

    # load in the environment parameters
    base_env_params, specific_env_params = load_env_params(args.env_id, "configs/envs/")
    env_params = base_env_params.copy()
    # set the environment to test
    base_env_params["training"] = False
    # add prediction horizon to the season length to prevent env from resetting
    base_env_params["season_length"] += args.horizon/24
    specific_env_params["eval_options"]["eval_years"] = [args.growth_year]
    specific_env_params["eval_options"]["eval_days"] = [args.start_day]
    # load the hyperparameter for MPC and RL
    mpc_params = load_model_hyperparams("mpc", args.env_id)

    eval_env = load_env(
        args.env_id,
        args.model_name,
        base_env_params,
        specific_env_params,
        load_path,
        args.frame_stack,
        args.n_stack,
        args.plant_state_env,
    )

    eval_env.reset()
    model = ALGS[args.algorithm].load(rl_model_path, env=eval_env, device="cpu")
    print(f"Running RL-MPC method...")

    rlmpc = RLMPC(
        nx=env_params["nx"],
        nu=env_params["nu"],
        ns=mpc_params["ns"],
        n_params=env_params["num_params"],
        nd=env_params["nd"],
        dt=env_params["dt"],
        horizon=args.horizon,
        u_min=env_params["u_min"],
        u_max=env_params["u_max"],
        delta_u_max=env_params["delta_u_max"],
        constraints=specific_env_params["constraints"],
        reward_params=specific_env_params["reward_params"],
        region_range=args.region_range,
        nlp_opts=mpc_params["nlp_opts"],
        terminal_constraint=args.terminal_constraint,
        normalize_x=args.normalize_x,
        eval_env=eval_env,
        model=model,
        terminal_penalty=args.terminal_penalty,
        load_norm_x=args.load_norm_x,
        norm_x_path=os.path.join(load_path, "envs", args.model_name, "x_norm.npy"),
    )

    exp = RLMPCExperimentManager(
        rlmpc,
        n_days=env_params["season_length"],
        # n_days=1/288,
        location=env_params["location"],
        growth_year=args.growth_year,
        start_day=args.start_day,
        offline_rl=args.offline_rl,
        extend_ocp_region=args.extend_ocp_region,
        plant_state_env=args.plant_state_env,
        selector_mechanism=args.selector_mechanism,
    )

    rlmpc.define_nlp_multi()
    exp.solve_nmpc_multi()

    exp.X = convert_rh_ppm(exp.X)
    exp.save_data(save_dir)
    exp.save_costs(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--model_name", type=str, default="graceful-planet-22")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    parser.add_argument("--region_range", type=float, default=0.05, help="Region range")
    parser.add_argument("--offline_rl", action=argparse.BooleanOptionalAction, help="Use offline RL trajectory")
    parser.add_argument("--terminal_constraint", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--terminal_penalty", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--extend_ocp_region", action=argparse.BooleanOptionalAction, help="Extend OCP region")
    parser.add_argument("--normalize_x", action=argparse.BooleanOptionalAction, help="Normalize states")
    parser.add_argument("--load_norm_x", action="store_true", help="Load rather than compute the state normalization vector")
    parser.add_argument("--frame_stack", action="store_true", help="Whether to use frame stacking")
    parser.add_argument("--n_stack", type=int, default=1, help="Number of frames to stack")
    parser.add_argument("--growth_year", type=int, default=2023, help="Growth year")
    parser.add_argument("--start_day", type=int, default=151, help="Start day")
    parser.add_argument("--plant_state_env", action="store_true", help="Whether to use plant state environment")
    parser.add_argument("--selector_mechanism", action="store_true", help="Whether to use selector mechanism")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    args = parser.parse_args()
    main(args)
