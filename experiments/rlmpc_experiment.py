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
    # set the environment to test
    base_env_params["training"] = False
    # add prediction horizon to the season length to prevent env from resetting
    base_env_params["season_length"] += args.horizon/24

    # load the hyperparameter for MPC and RL
    mpc_params = load_model_hyperparams("mpc", args.env_id)

    eval_env = load_env(
        args.env_id,
        args.model_name,
        base_env_params,
        specific_env_params,
        load_path,
        args.frame_stack,
        args.n_stack
    )

    eval_env.reset()
    model = ALGS[args.algorithm].load(rl_model_path, env=eval_env, device="cpu")

    n_days = 1
    month = "june"

    print(f"Running RL-MPC method...")

    rlmpc = RLMPC(
        nx=base_env_params["nx"],
        nu=base_env_params["nu"],
        ns=mpc_params["ns"],
        n_params=base_env_params["num_params"],
        nd=base_env_params["nd"],
        dt=base_env_params["dt"],
        horizon=args.horizon,
        u_min=base_env_params["u_min"],
        u_max=base_env_params["u_max"],
        delta_u_max=base_env_params["delta_u_max"],
        constraints=specific_env_params["constraints"],
        reward_params=specific_env_params["reward_params"],
        region_range=args.region_range,
        nlp_opts=mpc_params["nlp_opts"],
        terminal_constraint=args.terminal_constraint,
        eval_env=eval_env,
        model=model,
        terminal_penalty=args.terminal_penalty,
    )

    exp = RLMPCExperimentManager(
        rlmpc,
        # n_days=base_env_params["season_length"],
        n_days=1/12,
        location=base_env_params["location"],
        growth_year=base_env_params["start_train_year"],
        start_day=base_env_params["start_train_day"],
        offline_rl=args.offline_rl,
        extend_ocp_region=args.extend_ocp_region
    )

    rlmpc.define_nlp_multi()
    exp.solve_nmpc_multi()

    result_columns = eval_env.env_method("get_obs_names")[0][:20]
    result_columns.extend(["Rewards",  "EPI", "Penalty"])
    # result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    # result_columns.extend(["episode"])
    result = Results(result_columns)

    exp.X = convert_rh_ppm(exp.X)
    exp.save_data(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--model_name", type=str, default="graceful-planet-22")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    parser.add_argument("--region_range", type=float, default=0.05, help="Region range")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--offline_rl", action=argparse.BooleanOptionalAction, help="Use offline RL trajectory")
    parser.add_argument("--terminal_constraint", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--terminal_penalty", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--extend_ocp_region", action=argparse.BooleanOptionalAction, help="Extend OCP region")
    parser.add_argument("--frame_stack", action="store_true", help="Whether to use frame stacking")
    parser.add_argument("--n_stack", type=int, default=1, help="Number of frames to stack")
    args = parser.parse_args()

    main(args)
