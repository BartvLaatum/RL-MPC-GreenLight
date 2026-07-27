import os
import argparse

import numpy as np
from stable_baselines3 import PPO

from controllers.rlmpc import RLMPC
from common.utils import load_model_hyperparams, load_env_params, load_env
from common.results import Results
from environments.utils import convert_rh_ppm

ALGS = {
    "ppo": PPO
    }

def main(args: argparse.Namespace):
    load_path = os.path.join("train_data", args.project, args.algorithm, "deterministic")
    # save_dir = os.path.join("results", args.project, "deterministic", "rlmpc", args.experiment_name)
    rl_model_path = os.path.join(load_path, "models", args.model_name, "best_model.zip")

    # load in the environment parameters
    base_env_params, specific_env_params = load_env_params(args.env_id, "configs/envs/")
    env_params = base_env_params.copy()
    # set the environment to test
    base_env_params["training"] = False
    # add prediction horizon to the season length to prevent env from resetting
    # base_env_params["season_length"] += args.horizon/24

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
        eval_env=eval_env,
        model=model,
        terminal_penalty=args.terminal_penalty,
    )
    L = base_env_params["season_length"]*86400
    t = np.arange(0, L, rlmpc.dt)
    N = len(t)

    logs = rlmpc.unroll_actor(horizon=N, freeze=False)
    # logs = rlmpc.unroll_actor(horizon=1, freeze=False)
    print(logs["x"].mean(axis=1).shape)
    print(logs["x"].mean(axis=1))
    import matplotlib.pyplot as plt

    # Each row in logs["x"] is a state variable. Shape: (nx, T)
    states = logs["x"]
    num_states, T = states.shape

    fig, axs = plt.subplots(7, 4, figsize=(24, 20), sharex=True)
    axs = axs.flatten()
    for i in range(num_states):
        axs[i].plot(t[:T-1], states[i, :T-1], label=f"State {i+1}")
        axs[i].set_title(f"State {i+1}")
        axs[i].grid(True)
        axs[i].legend(fontsize="x-small")
    # Hide any unused subplots (if num_states < 28)
    for j in range(num_states, len(axs)):
        axs[j].axis('off')
    # Set common labels
    for ax in axs[-4:]:
        ax.set_xlabel("Time [s]")
    for i in range(0, len(axs), 4):
        axs[i].set_ylabel("State Value")

    ax.set_title("State Trajectories for All 27 States")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("State Value")
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize='small', ncol=2)
    plt.tight_layout()
    fig.savefig("all_states.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL")
    parser.add_argument("--algorithm", type=str, default="ppo")
    parser.add_argument("--model_name", type=str, default="trim-durian-32")
    parser.add_argument("--env_id", type=str, default="TomatoEnv")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--region_range", type=float, default=0.1)
    parser.add_argument("--terminal_constraint", type=bool, default=True)
    parser.add_argument("--terminal_penalty", type=bool, default=True)
    parser.add_argument("--frame_stack", action="store_true", default=False)
    parser.add_argument("--n_stack", type=int, default=1)
    
    args = parser.parse_args()
    main(args)