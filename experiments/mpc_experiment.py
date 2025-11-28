import argparse
import os

from controllers.mpc import MPC
from experiments.mpc_experiment_managers import MPCExperimentManager
from common.utils import load_model_hyperparams
from environments.utils import convert_rh_ppm


def main(args: argparse.Namespace):
    save_dir = os.path.join("results", args.project, "deterministic", "mpc", args.experiment_name)

    n_params = 208
    nx = 28
    nu = 6
    ns = 6
    nd = 10
    dt = 300.
    n_days = 1
    month = "june"
    Np = int(args.horizon * 3600 / dt)  # Convert horizon in hours to number of steps

    print(f"Running MPC...")
    print(f"Using {args.linear_solver} linear solver")
    mpc_params = load_model_hyperparams("mpc", "TomatoEnv")
    nlp_opts = mpc_params["nlp_opts"]

    nlp_opts["ipopt"]["linear_solver"] = args.linear_solver
    mpc = MPC(nx, nu, ns, n_params, nd, dt, Np, nlp_opts)
    exp = MPCExperimentManager(mpc, month, n_days, args.method)

    mpc.define_nlp_multi()
    exp.solve_nmpc_multi()

    exp.X = convert_rh_ppm(exp.X)
    exp.save_data(save_dir, args.horizon)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL", help="Name of the project (in wandb)")
    parser.add_argument("--experiment_name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--linear_solver", type=str, default="ma57", help="Linear solver to use")
    parser.add_argument("--horizon", type=int, default=1, help="Prediction horizon in hours")
    args = parser.parse_args()

    main(args)
