import os
import argparse

from stable_baselines3 import PPO

from controllers.rl_mpc import RLMPC
from experiments.mpc_experiment_managers import RLMPCExperimentManager
from common.utils import load_rl_env_params, load_mpc_params, load_rl_hyperparams, load_env
from common.results import Results
from model.utils import convert_rh_ppm

ALGS = {
    "ppo": PPO
    }

def main(args: argparse.Namespace):

    load_path = os.path.join("train_data", args.project, args.algorithm, "deterministic")
    save_dir = os.path.join("results", args.project, "deterministic", "rlmpc", args.experiment_name)

    rl_model_path = os.path.join(load_path, "models", args.model_name, "best_model.zip")
    # vf_path = f"{load_path}/models/{args.model_name}/vf.zip"

    # load in the environment parameters
    base_env_params, specific_env_params = load_rl_env_params(args.env_id, "configs/envs/")
    # set the environment to test
    base_env_params["training"] = False
    # add prediction horizon to the season length to prevent env from resetting
    base_env_params["season_length"] += args.horizon/24

    # load the hyperparameter for MPC and RL
    mpc_params = load_mpc_params(args.env_id)
    hyperparameters = load_rl_hyperparams(args.env_id, args.algorithm)
    eval_env = load_env(args.env_id, args.model_name, base_env_params, specific_env_params, load_path)
    eval_env.reset()
    model = ALGS[args.algorithm].load(rl_model_path, env=eval_env, device="cpu")

    n_params = 208
    nx = 28
    nu = 6
    ns = 6
    nd = 10
    dt = 300.
    n_days = 1
    month = "june"
    Np = int(args.horizon * 3600 / dt)  # Convert hours to steps

    print(f"Running {args.method} method")

    if args.method == "finite-difference":
        mpc_params["nlp_opts"]["ipopt"]["jacobian_approximation"] = "finite-difference-values"

    rl_mpc = RLMPC(
        nx,
        nu,
        ns,
        n_params,
        nd,
        dt,
        Np,
        args.region_range,
        mpc_params["nlp_opts"],
        args.terminal_constraint,
        eval_env,
        model,
        args.terminal_penalty,
    )

    exp = RLMPCExperimentManager(
        rl_mpc, 
        month,
        n_days,
        args.method,
        offline_rl=args.offline_rl,
        extend_ocp_region=args.extend_ocp_region
    )

    if args.method == "exact":
        print("multi")
        rl_mpc.define_nlp_multi()
        exp.solve_nmpc_multi()
    elif args.method == "finite-difference":
        # ["nlp_opts"]["ipopt"]["jacobian_approximation"] = "finite-difference-values"
        rl_mpc.define_nlp()
        exp.solve_nmpc()

    result_columns = eval_env.env_method("get_obs_names")[0][:23]
    result_columns.extend(["Rewards",  "EPI", "Penalty"])
    # result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    # result_columns.extend(["episode"])
    result = Results(result_columns)

    exp.X = convert_rh_ppm(exp.X)
    exp.save_data(save_dir, "", args.horizon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="weather-300dt.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--model_name", type=str, default="graceful-planet-22")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    parser.add_argument("--method", type=str, default="exact", choices=["exact", "finite-difference"])
    parser.add_argument("--region_range", type=float, default=0.05, help="Region range")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--offline_rl", action=argparse.BooleanOptionalAction, help="Use offline RL trajectory")
    parser.add_argument("--terminal_constraint", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--terminal_penalty", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")
    parser.add_argument("--extend_ocp_region", action=argparse.BooleanOptionalAction, help="Extend OCP region")
    # parser.add_argument("--use_trained_vf", action="store_true")
    args = parser.parse_args()

    main(args)

