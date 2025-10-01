import os
import argparse

from stable_baselines3 import PPO

from agents.rl_mpc import RLMPC
from experiments.mpc_experiment_managers import RLMPCExperimentManager
from common.utils import load_rl_env_params, load_mpc_params, load_rl_hyperparams, load_env
from common.results import Results
from model.utils import convert_rh_ppm

ALGS = {
    "ppo": PPO
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--save_name", type=str)
    parser.add_argument("--weather_filename", default="weather-300dt.csv", type=str)
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--model_name", type=str, default="graceful-planet-22")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    parser.add_argument("--method", type=str, default="exact", choices=["exact", "finite-difference"])
    parser.add_argument("--terminal_constraint", action=argparse.BooleanOptionalAction, help="Enable terminal constraint in MPC")

    # parser.add_argument("--approach", type=str, default="multi", choices=["single", "multi"], required=True)

    # parser.add_argument("--use_trained_vf", action="store_true")
    args = parser.parse_args()

    load_path = f"train_data/{args.project}/{args.algorithm}/deterministic"
    save_dir = f"results/{args.project}/deterministic/rlmpc/"

    rl_model_path = f"{load_path}/models/{args.model_name}/best_model.zip"
    # vf_path = f"{load_path}/models/{args.model_name}/vf.zip"
    env_path = f"{load_path}/envs/{args.model_name}/best_vecnormalize.pkl"

    base_env_params, specific_env_params = load_rl_env_params(args.env_id, "configs/envs/")
    
    # load the hyperparameter for MPC and RL
    mpc_params = load_mpc_params(args.env_id)
    hyperparameters = load_rl_hyperparams(args.env_id, args.algorithm)
    eval_env = load_env(args.env_id, args.model_name, base_env_params, specific_env_params, load_path)
            #    load_env(args.env_id, args.model_name, env_base_params, env_specific_params, load_path)
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

    rl_mpc = RLMPC(nx, nu, ns, n_params, nd, dt, Np, mpc_params["nlp_opts"], args.terminal_constraint, eval_env, model)
    exp = RLMPCExperimentManager(rl_mpc, month, n_days, args.method)

    if args.method == "exact":
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
    main()
