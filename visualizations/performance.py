import argparse
import os
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_config
import matplotlib.cm as cm

def load_rl_mpc_data(project, ablations):
    data = {}
    BASE_DIR = os.path.join("results", project, "deterministic", "rlmpc")
    horizons = [1]

    data = {ablation: {} for ablation in ablations}

    for ablation in ablations:
        for H in horizons:
            U = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"control-inputs-cs-300dt-{H}H.csv"), delimiter=",").T
            X = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"states-cs-300dt-{H}H.csv"), delimiter=",").T
            times = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"times-cs-300dt-{H}H.csv"), delimiter=",").T
            rewards = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"rewards-cs-300dt-{H}H.csv"), delimiter=",").T
            EPI = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"EPI-cs-300dt-{H}H.csv"), delimiter=",").T
            penalties = np.loadtxt(os.path.join(BASE_DIR, ablation,  f"penalties-cs-300dt-{H}H.csv"), delimiter=",").T
            data[ablation][H] = {
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

def load_mpc_data(project):
    data = {"mpc": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "mpc")
    horizons = [1]

    for H in horizons:
        U = np.loadtxt(os.path.join(BASE_DIR, f"control-inputs-cs-300dt-{H}H.csv"), delimiter=",").T
        X = np.loadtxt(os.path.join(BASE_DIR, f"states-cs-300dt-{H}H.csv"), delimiter=",").T
        times = np.loadtxt(os.path.join(BASE_DIR, f"times-cs-300dt-{H}H.csv"), delimiter=",").T
        rewards = np.loadtxt(os.path.join(BASE_DIR, f"rewards-cs-300dt-{H}H.csv"), delimiter=",").T
        EPI = np.loadtxt(os.path.join(BASE_DIR, f"EPI-cs-300dt-{H}H.csv"), delimiter=",").T
        penalties = np.loadtxt(os.path.join(BASE_DIR, f"penalties-cs-300dt-{H}H.csv"), delimiter=",").T
        data["mpc"][H] = {
            "U": U,
            "X": X,
            "times": times,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data

def load_rl_data(project, model_name):
    data = {"rl": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "ppo")

    rl_data = pd.read_csv(os.path.join(BASE_DIR, model_name + ".csv"))
    U = rl_data.iloc[:, 7:13].values.T
    X = rl_data.iloc[:, 0:6].values.T
    EPI = rl_data["EPI"].values.T
    rewards = rl_data["Rewards"].values.T
    penalties = rl_data["Penalty"].values.T
    data["rl"] = {
        "U": U,
        "X": X,
        "rewards": rewards,
        "EPI": EPI,
        "penalties": penalties
    }
    return data

def plot_control_trajectories(data, dt, labels, output_path=None, show=False):
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

    for i, method in enumerate[Any](data.keys()):
        if method == "rl":
            U = data[method]["U"]
            t = np.arange(0, U.shape[-1]*dt, dt)/86400

            axes[0,0].step(t[:-1], U[0, 1:], color=COLORS[0], label=labels[i], where='post')
            axes[0,1].step(t[:-1], U[1, 1:], color=COLORS[0], label=labels[i], where='post')
            axes[1,0].step(t[:-1], U[2, 1:], color=COLORS[0], label=labels[i], where='post')
            axes[1,1].step(t[:-1], U[3, 1:], color=COLORS[0], label=labels[i], where='post')
            axes[2,0].step(t[:-1], U[4, 1:], color=COLORS[0], label=labels[i], where='post')
            axes[2,1].step(t[:-1], U[5, 1:], color=COLORS[0], label=labels[i], where='post')
        else:
            for H in data[method].keys():
                U = data[method][H]["U"]
                t = np.arange(0, U.shape[-1]*dt, dt)/86400
                all_Hs = sorted(data[method].keys())
                max_H = all_Hs[-1]

                axes[0,0].step(t[:-1], U[0,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')
                axes[0,1].step(t[:-1], U[1,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')
                axes[1,0].step(t[:-1], U[2,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')
                axes[1,1].step(t[:-1], U[3,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')
                axes[2,0].step(t[:-1], U[4,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')
                axes[2,1].step(t[:-1], U[5,1:], color=COLORS[i], label=f"{labels[i]} ({H}H)", where='post')

        # elif method == "rlmpc":
        #     for H in data[method].keys():
        #         U = data[method][H]["U"]
        #         t = np.arange(0, U.shape[-1]*dt, dt)/86400
        #         all_Hs = sorted(data[method].keys())
        #         max_H = all_Hs[-1]
        #         current_color = cm.OrRd((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)

        #         axes[0,0].step(t[:-1], U[0,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
        #         axes[0,1].step(t[:-1], U[1,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
        #         axes[1,0].step(t[:-1], U[2,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
        #         axes[1,1].step(t[:-1], U[3,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
        #         axes[2,0].step(t[:-1], U[4,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
        #         axes[2,1].step(t[:-1], U[5,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')

    for i, ax in enumerate(axes.flat):
        ax.set_ylabel(control_labels_with_units[i])
    axes[0,0].legend()
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


def plot_states(data, dt, labels, output_path=None, show=False):
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

    for i, method in enumerate(data.keys()):
        if method == "rl":
            X = data[method]["X"]
            t = np.arange(0, X.shape[-1]*dt, dt)/86400
            axes[0,0].step(t[:], X[1, :], color=COLORS[i], label=labels[i], where='post')
            axes[0,1].step(t[:], X[0, :], color=COLORS[i], label=labels[i], where='post')
            axes[1,0].step(t[:], X[2, :], color=COLORS[i], label=labels[i], where='post')
            axes[1,1].step(t[:], X[5, :], color=COLORS[i], label=labels[i], where='post')
        else:
            for j, H in enumerate(data[method].keys()):
                X = data[method][H]["X"]
                t = np.arange(0, X.shape[-1]*dt, dt)/86400
                all_Hs = sorted(data[method].keys())
                max_H = all_Hs[-1]

                axes[0,0].step(t, X[2,:], color=COLORS[i], label=labels[i], where='post')
                axes[0,1].step(t, X[0, :], color=COLORS[i], label=labels[i], where='post')
                axes[1,0].step(t, X[15, :], color=COLORS[i], label=labels[i], where='post')
                axes[1,1].step(t, X[25, :], color=COLORS[i], label=labels[i], where='post')


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

    axes[0,1].legend(loc='upper left')
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

def plot_runtime(data, dt, labels, output_path=None, show=False):
    """Plot a barplot of the average solver times for both methods."""
    avg_times = []
    for method in data.keys():
        if method == "rl":
            continue
        if data[method]:
            H = 1
            times = data[method][H]["times"]
            avg_times.append(np.mean(times))
            
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

def plot_rewards(data, dt, labels, output_path=None, show=False):
    """Plot rewards, EPI, and penalties over time."""
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), dpi=180)
    ax[0].set_xlabel('Time (days)')
    ax[1].set_xlabel('Time (days)')
    ax[2].set_xlabel('Time (days)')
    ax[0].set_ylabel('Cumulative Reward')
    ax[1].set_ylabel('EPI')
    ax[2].set_ylabel('Cumulative Penalty')
    fig.suptitle(f'Rewards over time')
    
    # Plot RL
    if "rl" in data:
        RL = data["rl"]
        rewards = RL["rewards"]
        EPI = RL["EPI"]
        penalties = RL["penalties"]
        t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
        ax[0].step(t, rewards.cumsum(), label=f"RL", color=COLORS[0])
        ax[1].step(t, EPI.cumsum(), color=COLORS[0])
        ax[2].step(t, penalties.cumsum(), color=COLORS[0])

    # Plot MPC
    if "mpc" in data:
        MPC = data["mpc"][1]
        rewards = MPC["rewards"]
        EPI = MPC["EPI"]
        penalties = MPC["penalties"]
        t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
        ax[0].plot(t, rewards.cumsum(), label=f"MPC (1H)", color=COLORS[1])
        ax[1].plot(t, EPI.cumsum(), color=COLORS[1])
        ax[2].plot(t, penalties.cumsum(), color=COLORS[1])

    # Plot RL-MPC variants
    color_idx = 2
    for i, method in enumerate(data.keys()):
        if method not in ["rl", "mpc"]:
            if method in data and 1 in data[method]:
                RL_MPC = data[method][1]
                rewards = RL_MPC["rewards"]
                EPI = RL_MPC["EPI"]
                penalties = RL_MPC["penalties"]
                t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
                
                # Find corresponding label
                label_idx = [j for j, key in enumerate(data.keys()) if key == method][0]
                label = labels[label_idx] if label_idx < len(labels) else method
                
                ax[0].plot(t, rewards.cumsum(), label=label, color=COLORS[color_idx % len(COLORS)])
                ax[1].plot(t, EPI.cumsum(), color=COLORS[color_idx % len(COLORS)])
                ax[2].plot(t, penalties.cumsum(), color=COLORS[color_idx % len(COLORS)])
                color_idx += 1

    ax[0].legend(loc='upper left')
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        print(f"Saved rewards plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


COLORS = ['C0', 'C1', 'C2', 'C3', 'C4']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate performance plots for RL, MPC, and RL-MPC methods."
    )
    # Data loading arguments
    parser.add_argument("--model_name", type=str, default="graceful-planet-22", 
                        help="Name of the RL model to load.")
    parser.add_argument("--rlmpc_folders", type=str, nargs="+", 
                        default=["terminal_constr", "terminal_constr_all", "terminal_constr_pen_all"], 
                        help="Folders to load RL-MPC data from.")
    parser.add_argument("--project", type=str, default="GL-MPC-RL",
                        help="Project name for data loading.")
    parser.add_argument("--dt", type=int, default=300,
                        help="Time step in seconds.")
    
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
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save plots. If not specified, defaults to figures/<project>/.")
    parser.add_argument("--output_prefix", type=str, default="",
                        help="Prefix for output filenames.")
    
    # Label customization
    parser.add_argument("--labels", type=str, nargs="+", 
                        default=["RL", "MPC", "Terminal state", "Terminal state (all)", "Terminal state penalty (all)"],
                        help="Labels for each method in the plots.")
    
    args = parser.parse_args()
    
    # Set up output directory
    if args.output_dir is None:
        output_dir = os.path.join("figures", args.project)
    else:
        output_dir = args.output_dir
    
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
    rlmpc_data = load_rl_mpc_data(args.project, args.rlmpc_folders)
    mpc_data = load_mpc_data(args.project)
    rl_data = load_rl_data(args.project, args.model_name)
    data = {**rl_data, **mpc_data, **rlmpc_data}
    print("Data loaded successfully.")
    
    # Generate plots
    if args.plot_all or args.plot_rewards:
        output_path = os.path.join(output_dir, f"{args.output_prefix}rewards-{args.dt}dt.png") if args.save else None
        plot_rewards(data, args.dt, args.labels, output_path=output_path, show=args.show)
    
    if args.plot_all or args.plot_states:
        output_path = os.path.join(output_dir, f"{args.output_prefix}states-{args.dt}dt.png") if args.save else None
        plot_states(data, args.dt, args.labels, output_path=output_path, show=args.show)
    
    if args.plot_all or args.plot_runtime:
        output_path = os.path.join(output_dir, f"{args.output_prefix}runtime-{args.dt}dt.png") if args.save else None
        # Exclude RL from runtime plot labels
        runtime_labels = [label for i, label in enumerate(args.labels) if list(data.keys())[i] != "rl"]
        plot_runtime(data, args.dt, runtime_labels, output_path=output_path, show=args.show)
    
    if args.plot_all or args.plot_controls:
        output_path = os.path.join(output_dir, f"{args.output_prefix}control-inputs-{args.dt}dt.png") if args.save else None
        plot_control_trajectories(data, args.dt, args.labels, output_path=output_path, show=args.show)
    

