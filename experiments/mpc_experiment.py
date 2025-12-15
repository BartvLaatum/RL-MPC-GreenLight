import argparse
import os

from controllers.mpc import MPC
from experiments.mpc_experiment_managers import MPCExperimentManager
from common.utils import load_model_hyperparams, load_env_params
from environments.utils import convert_rh_ppm


def main(args: argparse.Namespace):
    save_dir = os.path.join("results", args.project, "deterministic", "mpc", args.experiment_name)
    print(f"Running MPC...")
    mpc_params = load_model_hyperparams("mpc", "TomatoEnv")
    env_params, env_specific_params = load_env_params("TomatoEnv", "configs/envs/")
    nlp_opts = mpc_params["nlp_opts"]

    mpc = MPC(
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
        constraints=env_specific_params["constraints"],
        reward_params=env_specific_params["reward_params"],
        nlp_opts=nlp_opts,
    )

    exp = MPCExperimentManager(
        mpc,
        n_days=env_params["season_length"],
        location=env_params["location"],
        growth_year=env_params["start_train_year"],
        start_day=env_params["start_train_day"]
    )

    mpc.define_nlp_multi()
    exp.solve_nmpc_multi()

    exp.X = convert_rh_ppm(exp.X)
    exp.save_data(save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL", help="Name of the project (in wandb)")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    args = parser.parse_args()

    main(args)
