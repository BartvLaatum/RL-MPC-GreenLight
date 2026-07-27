import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_config
import matplotlib.cm as cm
# Fixed subplot margins (figure-fraction) so all metric figures have identical
# axes widths regardless of y-tick / y-label content.
SUBPLOT_KW = dict[str, float](left=0.25, right=0.97, top=0.97, bottom=0.2)

dt = 300

def load_rl_mpc_data(project, ablations, location, growth_year, start_day, horizons):
    data = {"rlmpc": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "rlmpc")

    data["rlmpc"] = {ablation: {} for ablation in ablations}

    for ablation in ablations:
        for H in horizons:
            U = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"control-inputs-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            X = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"states-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            times = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"times-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            rewards = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"rewards-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            EPI = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"EPI-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            penalties = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"penalties-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
            data["rlmpc"][ablation][H] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }
            if ablation == "offline_rl_trajectory":
                offline_rl_x = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"x-offline-rl.csv"), delimiter=",").T
                data[ablation][H]["rollout_x"] = offline_rl_x
    return data

def load_mpc_data(project, mpc_folder, location, growth_year, start_day):
    data = {"mpc": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "mpc", mpc_folder)
    horizons = [1]

    for H in horizons:
        U = np.loadtxt(os.path.join(BASE_DIR, f"control-inputs-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        X = np.loadtxt(os.path.join(BASE_DIR, f"states-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        times = np.loadtxt(os.path.join(BASE_DIR, f"times-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        rewards = np.loadtxt(os.path.join(BASE_DIR, f"rewards-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        EPI = np.loadtxt(os.path.join(BASE_DIR, f"EPI-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        penalties = np.loadtxt(os.path.join(BASE_DIR, f"penalties-300dt-{H}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        data["mpc"][H] = {
            "U": U,
            "X": X,
            "times": times,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data

def load_rl_data(project, model_names, location, growth_year, start_day):
    data = {
        "rl": {model_name: {} for model_name in model_names}
    }
    BASE_DIR = os.path.join("results", project, "deterministic", "ppo")

    for model_name in model_names:
        rl_data = pd.read_csv(os.path.join(BASE_DIR, f"{model_name}-{location}-{growth_year}-{start_day}.csv"))

        U = rl_data[["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]].values.T
        X = rl_data[["co2_air", "temp_air", "rh_air", "pipe_temp", "cFruit"]].values.T
        EPI = rl_data["EPI"].values.T
        rewards = rl_data["Rewards"].values.T
        penalties = rl_data["Penalty"].values.T
        data["rl"][model_name] = {
            "U": U,
            "X": X,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data

def plot_control_trajectories(data, labels, colors, linestyles, horizon=1,output_path=None, show=False, N_to_plot=None):
    """Plot the optimized control trajectories."""
    control_labels_with_units = {
        0: r"$u_{heat}$",
        1: r"$u_{CO_2}$",
        2: r"$u_{ThScr}$",
        3: r"$u_{vent}$",
        4: r"$u_{light}$",
        5: r"$u_{BlScr}$",
    }
    WIDTH = 175 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True, sharey=True)
    color_idx = 0
    for i, method in enumerate[Any](data.keys()):
        if method == "rl":
            for model_name in data[method].keys():
                U = data[method][model_name]["U"]
                t = np.arange(0, U.shape[-1]*dt, dt)/86400

                if N_to_plot:
                    t = t[:N_to_plot]
                    U = U[:, :N_to_plot]

                axes[0,0].step(t[:-1], U[0, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[0,1].step(t[:-1], U[1, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,0].step(t[:-1], U[2, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,1].step(t[:-1], U[3, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[2,0].step(t[:-1], U[4, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[2,1].step(t[:-1], U[5, 1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                color_idx += 1
        elif method == "mpc":
            for horizon in data[method].keys():
                U = data[method][horizon]["U"]
                t = np.arange(0, U.shape[-1]*dt, dt)/86400
                if N_to_plot:
                    t = t[:N_to_plot]
                    U = U[:, :N_to_plot]

                axes[0,0].step(t[:-1], U[0,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                axes[0,1].step(t[:-1], U[1,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                axes[1,0].step(t[:-1], U[2,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                axes[1,1].step(t[:-1], U[3,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                axes[2,0].step(t[:-1], U[4,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                axes[2,1].step(t[:-1], U[5,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                color_idx += 1
        elif method == "rlmpc":
            for ablation in data[method].keys():
                for horizon in data[method][ablation].keys():
                    U = data[method][ablation][horizon]["U"]
                    t = np.arange(0, U.shape[-1]*dt, dt)/86400
                    if N_to_plot:
                        t = t[:N_to_plot]
                        U = U[:, :N_to_plot]
                    axes[0,0].step(t[:-1], U[0,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    axes[0,1].step(t[:-1], U[1,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    axes[1,0].step(t[:-1], U[2,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    axes[1,1].step(t[:-1], U[3,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    axes[2,0].step(t[:-1], U[4,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    axes[2,1].step(t[:-1], U[5,1:], color=colors[color_idx], linestyle=linestyles[color_idx], label=f"{labels[color_idx]} ({horizon}H)", where='post')
                    color_idx += 1
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel(control_labels_with_units[i])
    # axes[0,0].legend()
    fig.supxlabel('Time (days)')
    fig.supylabel('Control Input')
    fig.suptitle('Closed-loop Control Trajectories')
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Saved control trajectories plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


def plot_states(data, labels, colors, linestyles, output_path=None, show=False, N_to_plot=None):
    """Plot the optimized state trajectories."""
    state_labels_with_units = {
        0: r"Air Temperature ($^\circ$C)",
        1: r"CO$_2$ (ppm)",
        2: r"Relative Humidty (%)",
        3: r"Fruit Weight (DM mg/m$^2$)"
    }
    bounds = {
        0: [15, 25],
        1: [400, 1600],
        2: [0, 90],
        3: None,
    }

    WIDTH = 175 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True)
    color_idx = 0
    for i, method in enumerate(data.keys()):
        if method == "rl":
            for i, model_name in enumerate(data[method].keys()):
                X = data[method][model_name]["X"]
                t = np.arange(0, X.shape[-1]*dt, dt)/86400
                if N_to_plot:
                    t = t[:N_to_plot]
                    X = X[:, :N_to_plot]
                axes[0,0].step(t[:], X[1, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[0,1].step(t[:], X[0, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,0].step(t[:], X[2, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,1].step(t[:], X[4, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                color_idx += 1
        elif method == "mpc":
            for horizon in data[method].keys():
                X = data[method][horizon]["X"]
                t = np.arange(0, X.shape[-1]*dt, dt)/86400
                if N_to_plot:
                    t = t[:N_to_plot]
                    X = X[:, :N_to_plot]
                axes[0,0].step(t, X[2, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[0,1].step(t, X[0, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,0].step(t, X[15, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                axes[1,1].step(t, X[25, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                color_idx += 1
        elif method == "rlmpc":
            for ablation in data[method].keys():
                for horizon in data[method][ablation].keys():
                    X = data[method][ablation][horizon]["X"]
                    t = np.arange(0, X.shape[-1]*dt, dt)/86400
                    if N_to_plot:
                        t = t[:N_to_plot]
                        X = X[:, :N_to_plot]
                    axes[0,0].step(t, X[2, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                    axes[0,1].step(t, X[0, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                    axes[1,0].step(t, X[15, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                    axes[1,1].step(t, X[25, :], color=colors[color_idx], linestyle=linestyles[color_idx], label=labels[color_idx], where='post')
                    color_idx += 1

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(state_labels_with_units[i])
    for i, ax in enumerate(axes.flat):
        if bounds[i]:
            ax.hlines(bounds[i], t[0], t[-1], color='grey', linestyle='--', label='Boundaries')

    if "offline_rl_trajectory" in data:
        offline_rl_x = data["offline_rl_trajectory"][1]["rollout_x"]
        t = np.arange(0, offline_rl_x.shape[-1]*dt, dt)/86400
        axes[0,0].step(t, offline_rl_x[2,:], color="C5", linestyle='--', label="Offline RL Trajectory", where='post')
        axes[0,1].step(t, offline_rl_x[0, :], color="C5", linestyle='--', label="Offline RL Trajectory", where='post')
        axes[1,0].step(t, offline_rl_x[15, :], color="C5", linestyle='--', label="Offline RL Trajectory", where='post')
        axes[1,1].step(t, offline_rl_x[25, :], color="C5", linestyle='--', label="Offline RL Trajectory", where='post')

    # axes[0,1].legend(loc='upper left')
    fig.supxlabel('Time (days)')
    fig.supylabel('State variable')
    fig.suptitle('Closed-loop State Trajectories')
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Saved state trajectories plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

def plot_runtime(data, labels, output_path=None, show=False):
    """Plot a barplot of the average solver times for both methods."""
    avg_times = []
    for method in data.keys():
        if method == "rl":
            continue
        elif method == "rlmpc":
            for ablation in data[method].keys():
                for horizon in data[method][ablation].keys():
                    times = data[method][ablation][horizon]["times"]
                    avg_times.append(np.mean(times))
        elif method == "mpc":
            for horizon in data[method].keys():
                times = data[method][horizon]["times"]
                avg_times.append(np.mean(times))
    from tabulate import tabulate

    # Print average times per label using tabulate
    table = [[label, f"{avg_time:.5f}"] for label, avg_time in zip(labels, avg_times)]
    print("Average Solver Times:")
    print(tabulate(table, headers=["Label", "Avg Time (s)"], tablefmt="github"))

    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)
    ax.bar(labels, avg_times, color=[cm.YlGn((i+1) / len(avg_times)) for i in range(len(avg_times))])
    ax.tick_params(axis='x', rotation=45)


    ax.set_ylabel("Average Solver Time (s)")
    ax.set_title("Average Solver Time")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Saved runtime plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)

def plot_rewards(data, labels, colors, linestyles, alpha=1., metric="rewards", output_path=None, show=False, N_to_plot=None):
    """Plot rewards, EPI, and penalties over time and print final summaries using tabulate."""
    from tabulate import tabulate

    WIDTH = 130/3* 0.03937
    HEIGHT = WIDTH * 0.6

    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, HEIGHT), dpi=300)
    ax.set_xlabel('Time (days)')
    if metric == "EPI":
        ax.set_ylabel('$\mathcal{E}_{j}$ (EUR/m$^2$)')
    elif metric == "penalties":
        ax.set_ylabel('$\mathcal{P}_{j}$')
    else:
        ax.set_ylabel('$\mathcal{J}_{j}$')
    color_idx = 0

    summary_table = []
    header = ["Label", f"Final Cumulative {metric.capitalize()}"]

    # Plot RL
    if "rl" in data:
        for i, model_name in enumerate(data["rl"].keys()):
            RL = data["rl"][model_name]
            if metric == "rewards":
                rewards = RL["rewards"]
            elif metric == "EPI":
                rewards = RL["EPI"]
            elif metric == "penalties":
                rewards = RL["penalties"]
            if N_to_plot:
                rewards = rewards[:N_to_plot]
            t = np.arange(0, rewards.shape[-1] * dt, dt) / 86400

            ax.step(t, rewards.cumsum(), label=f"{labels[color_idx]}", color=colors[color_idx], linestyle=linestyles[color_idx], alpha=alpha)
            # ax[1].step(t, EPI.cumsum(), color=colors[color_idx], linestyle=linestyles[color_idx])
            # ax[2].step(t, penalties.cumsum(), color=colors[color_idx], linestyle=linestyles[color_idx])

            summary_table.append([
                f"{labels[color_idx]} (RL)",
                f"{rewards.cumsum()[-1]:.3f}",
            ])
            color_idx += 1

    # Plot MPC
    if "mpc" in data:
        for horizon in data["mpc"].keys():
            MPC = data["mpc"][horizon]
            if metric == "rewards":
                rewards = MPC["rewards"]
            elif metric == "EPI":
                rewards = MPC["EPI"]
            elif metric == "penalties":
                rewards = MPC["penalties"]
            EPI = MPC["EPI"]
            penalties = MPC["penalties"]
            t = np.arange(0, rewards.shape[-1] * dt, dt) / 86400
            ax.step(t, rewards.cumsum(), label=f"MPC", color=colors[color_idx], linestyle=linestyles[color_idx], alpha=alpha)
            # ax[1].step(t, EPI.cumsum(), color=colors[color_idx], linestyle=linestyles[color_idx])
            # ax[2].step(t, penalties.cumsum(), color=colors[color_idx], linestyle=linestyles[color_idx])

            summary_table.append([
                f"MPC (1H)",
                f"{rewards.cumsum()[-1]:.3f}",
            ])
            color_idx += 1

    # Plot RL-MPC variants
    if "rlmpc" in data:
        for key in data["rlmpc"].keys():
            RL_MPC = data["rlmpc"][key][1]
            if metric == "rewards":
                rewards = RL_MPC["rewards"]
            elif metric == "EPI":
                rewards = RL_MPC["EPI"]
            elif metric == "penalties":
                rewards = RL_MPC["penalties"]
            t = np.arange(0, rewards.shape[-1] * dt, dt) / 86400
            ax.step(t, rewards.cumsum(), label=f"{labels[color_idx]}", color=colors[color_idx], linestyle=linestyles[color_idx], alpha=alpha)
            # ax.step(t, rewards.cumsum(), label=f"RL-MPC", color=colors[color_idx], linestyle=linestyles[color_idx], alpha=alpha)
            summary_table.append([
                f"{labels[color_idx]}",
                f"{rewards.cumsum()[-1]:.3f}",
            ])
            color_idx += 1
    print("\nFinal Results Summary:")
    print(tabulate(summary_table, headers=header, tablefmt="github"))
    
    ax.set_xlim(0, 1)
    # ax.set_ylim(0)
    ax.xaxis.set_major_locator(plt.LinearLocator(3))
    if metric == "penalties":
        ax.set_ylim(0)
        ax.set_yticks(np.linspace(0, 0.04, 3))
    elif metric == "EPI":
        ax.set_yticks(np.linspace(0, 0.2, 3))
    elif metric == "rewards":
        ax.set_yticks(np.linspace(0, 0.2, 3))

        ax.legend(loc='upper right', frameon=False, ncol=6)
    # fig.tight_layout()
    fig.subplots_adjust(**SUBPLOT_KW)

    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        # fig.savefig(output_path.replace(".png", ".eps"), format="eps", dpi=300)
        print(f"Saved rewards plot to {output_path}")
    if show:
        plt.show()
    return fig, ax

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate performance plots for RL, MPC, and RL-MPC methods."
    )
    # Data loading arguments

    parser.add_argument("--load_mpc", action="store_true", default=False,
                        help="Load MPC data.")
    parser.add_argument("--load_rl", action="store_true", default=True,
                        help="Load RL data.")
    parser.add_argument("--load_rlmpc", action="store_true", default=False,
                        help="Load RL-MPC data.")

    parser.add_argument("--colors", type=str, nargs="+", default=["#1f77b4", "#00a693", "#d62728"],
                        help="Colors for each method in the plots.")
    parser.add_argument("--linestyles", type=str, nargs="+", default=["-", "-", "-", "--", "-.", ":"],
                        help="Linestyles for each method in the plots.")

    parser.add_argument("--model_names", type=str, nargs="+", default=["daily-glade-107"], 
                        help="Names of the RL models to load.")
    parser.add_argument("--mpc_folder", type=str, default="mpc_linear_solver_ma57",
                        help="Folder to load MPC data from.")
    parser.add_argument("--rlmpc_folders", type=str, nargs="+", default=["rlmpc-switch-daily-glade-107-0.05"],
                        help="Folder to load RL-MPC data from.")
    parser.add_argument("--project", type=str, default="GL-MPC-RL",
                        help="Project name for data loading.")
    parser.add_argument("--location", type=str, default="Netherlands",
                        help="Location for data loading.")
    parser.add_argument("--growth_year", type=int, default=2023,
                        help="Growth year for data loading.")
    parser.add_argument("--start_day", type=int, default=151,
                        help="Start day for data loading.")
    parser.add_argument("--horizons", type=int, nargs="+", default=[1],
                        help="Horizons for data loading.")
    # Plot control arguments
    parser.add_argument("--plot_rewards", action="store_true", default=False,
                        help="Generate rewards plot.")
    parser.add_argument("--plot_states", action="store_true", default=False,
                        help="Generate states plot.")
    parser.add_argument("--plot_controls", action="store_true", default=False,
                        help="Generate control inputs plot.")
    parser.add_argument("--plot_runtime", action="store_true", default=False,
                        help="Generate runtime plot.")
    parser.add_argument("--plot_all", action="store_true", default=False,
                        help="Generate all plots.")

    # Output control arguments
    parser.add_argument("--save", action="store_true", default=False,
                        help="Save plots to files instead of displaying.")
    parser.add_argument("--show", action="store_true", default=False,
                        help="Show plots interactively.")
    parser.add_argument("--output_folder", type=str, default=None,
                        help="Directory to save plots. If not specified, defaults to figures/<project>/.")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Prefix for output filenames.")

    # Label customization
    parser.add_argument("--labels", type=str, nargs="+", 
                        default=["RL", "MPC", "RL-MPC", "No RL selection", r"No $X_N$", r"No $V_N$"],
                        
                        help="Labels for each method in the plots.")
    args = parser.parse_args()
    args.labels[-2:] = ["No $\mathbb{X}_f$", "No $\hat{V}_f$"]
    # Set up output directory
    output_dir = os.path.join("figures", args.project, args.output_folder)
    print(args.labels)
    if args.save:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    # Default behavior: if no specific plots are requested and neither save nor show is specified, show all
    if not (args.plot_rewards or args.plot_states or args.plot_controls or args.plot_runtime or args.plot_all):
        if not args.save and not args.show:
            args.plot_all = True
            args.show = True

    # Load data
    print("Loading data...")
    rlmpc_data = {}
    mpc_data = {}
    rl_data = {}
    if args.load_rlmpc:
        rlmpc_data = load_rl_mpc_data(args.project, args.rlmpc_folders, args.location, args.growth_year, args.start_day, args.horizons)
    if args.load_mpc:
        mpc_data = load_mpc_data(args.project, args.mpc_folder, args.location, args.growth_year, args.start_day)
    if args.load_rl:
        rl_data = load_rl_data(args.project, args.model_names, args.location, args.growth_year, args.start_day)
    data = {**rl_data, **mpc_data, **rlmpc_data}
    print("Data loaded successfully.")

    # Generate plots
    if args.plot_all or args.plot_rewards:
        output_path = os.path.join(output_dir, f"{args.output_prefix}.png") if args.save else None
        plot_rewards(data, args.labels, args.colors, args.linestyles, metric="rewards", output_path=output_path.replace(".png", "Rewards.png"), show=args.show, alpha=0.8)
        plot_rewards(data, args.labels, args.colors, args.linestyles, metric="EPI", output_path=output_path.replace(".png", "EPI.png"), show=args.show, alpha=0.8)
        plot_rewards(data, args.labels, args.colors, args.linestyles, metric="penalties", output_path=output_path.replace(".png", "Penalties.png"), show=args.show, alpha=0.8)

    if args.plot_all or args.plot_states:
        output_path = os.path.join(output_dir, f"{args.output_prefix}states.png") if args.save else None
        plot_states(data, args.labels, args.colors, args.linestyles, output_path=output_path, show=args.show)

    if args.plot_all or args.plot_runtime:
        output_path = os.path.join(output_dir, f"{args.output_prefix}runtime.png") if args.save else None

        # Exclude RL from runtime plot labels
        runtime_labels = args.labels[len(data["rl"].keys()):]

        plot_runtime(data, runtime_labels, output_path=output_path, show=args.show)

    if args.plot_all or args.plot_controls:
        output_path = os.path.join(output_dir, f"{args.output_prefix}control-inputs.png") if args.save else None
        plot_control_trajectories(data, args.labels, args.colors, args.linestyles, output_path=output_path, show=args.show)
