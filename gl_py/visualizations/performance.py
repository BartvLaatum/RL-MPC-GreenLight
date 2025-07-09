import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_config
import matplotlib.cm as cm

def load_data(methods, month, cs_suffix):
    """Load the data from the CSV files."""

    data = {
        "exact": {},
        "finite-difference": {},
        "RL": {},
    }

    for method in methods:
        if method == "exact":
            approach = "multi"
        elif method == "finite-difference":
            approach = "single"
        elif method == "rbc":
            approach = "single"
        elif method == "RL":
            rl_data = pd.read_csv(f"results/GL-MPC-RL/deterministic/graceful-planet-22.csv")
            U = rl_data.iloc[:, 7:13].values.T
            X = rl_data.iloc[:, 0:6].values.T
            EPI = rl_data["EPI"].values.T
            rewards = rl_data["Rewards"].values.T
            penalties = rl_data["Penalty"].values.T
            data[method] = {
                "U": U,
                "X": X,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }
            continue
        for H in [1, 2, 3]:
            U = np.loadtxt(f"results/test/{method}/{approach}/{month}/control-inputs{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            X = np.loadtxt(f"results/test/{method}/{approach}/{month}/states{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            times = np.loadtxt(f"results/test/{method}/{approach}/{month}/times{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            rewards = np.loadtxt(f"results/test/{method}/{approach}/{month}/rewards{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            EPI = np.loadtxt(f"results/test/{method}/{approach}/{month}/EPI{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            penalties = np.loadtxt(f"results/test/{method}/{approach}/{month}/penalties{cs_suffix}-300dt-{H}H.csv", delimiter=",").T
            data[method][H] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }

    return data

def plot_control_trajectories(data, dt):
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

    for i, method in enumerate(data.keys()):
        if data[method]:
            if method == "RL":
                U = data[method]["U"]
                t = np.arange(0, U.shape[-1]*dt, dt)/86400

                axes[0,0].step(t[:-1], U[0, 1:], color=f"C0", label=method.capitalize(), where='post')
                axes[0,1].step(t[:-1], U[1, 1:], color=f"C0", label=method.capitalize(), where='post')
                axes[1,0].step(t[:-1], U[2, 1:], color=f"C0", label=method.capitalize(), where='post')
                axes[1,1].step(t[:-1], U[3, 1:], color=f"C0", label=method.capitalize(), where='post')
                axes[2,0].step(t[:-1], U[4, 1:], color=f"C0", label=method.capitalize(), where='post')
                axes[2,1].step(t[:-1], U[5, 1:], color=f"C0", label=method.capitalize(), where='post')
            else:
                for H in data[method].keys():
                    U = data[method][H]["U"]
                    t = np.arange(0, U.shape[-1]*dt, dt)/86400
                    all_Hs = sorted(data[method].keys())
                    max_H = all_Hs[-1]
                    current_color = cm.YlGn((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)

                    axes[0,0].step(t[:-1], U[0,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
                    axes[0,1].step(t[:-1], U[1,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
                    axes[1,0].step(t[:-1], U[2,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
                    axes[1,1].step(t[:-1], U[3,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
                    axes[2,0].step(t[:-1], U[4,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')
                    axes[2,1].step(t[:-1], U[5,1:], color=current_color, label=f"{method.capitalize()} ({H}H)", where='post')

    for i, ax in enumerate(axes.flat):
        ax.set_ylabel(control_labels_with_units[i])
    axes[0,0].legend()
    fig.supxlabel('Time (days)')
    fig.supylabel('Control Input')
    fig.suptitle('Closed-loop Control Trajectories')
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{dir}/control-inputs{cs_suffix}-{int(dt)}dt.png")


def rewards_plot(data, dt, cs_suffix):
    """Plot the rewards over time."""

    fig, ax = plt.subplots(3, 1, figsize=(8, 6), dpi=180)
    colors = ["C0", "C1", "C0"]
    for i, method in enumerate(data.keys()):
        if data[method]:
            if method == "RL":
                rewards = data[method]["rewards"]
                EPI = data[method]["EPI"]
                penalties = data[method]["penalties"]

                t = np.arange(0, rewards.shape[-1]*dt, dt)/86400

                ax[0].plot(t, rewards.cumsum(), label=f"{method.capitalize()}", color=colors[i])
                ax[1].plot(t, EPI.cumsum(), label=f"{method.capitalize()}", color=colors[i])
                ax[2].plot(t, penalties.cumsum(), label=f"{method.capitalize()}", color=colors[i])
            else:
                for H in data[method].keys():
                    rewards = data[method][H]["rewards"]
                    EPI = data[method][H]["EPI"]
                    penalties = data[method][H]["penalties"]

                    t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
                    all_Hs = sorted(data[method].keys())
                    max_H = all_Hs[-1]
                    current_color = cm.YlGn((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)
                    ax[0].plot(t, rewards.cumsum(), label=f"{method.capitalize()} ({H}H)", color=current_color)
                    ax[1].plot(t, EPI.cumsum(), label=f"{method.capitalize()} ({H}H)", color=current_color)
                    ax[2].plot(t, penalties.cumsum(), label=f"{method.capitalize()} ({H}H)", color=current_color)


    ax[0].set_xlabel('Time (days)')
    ax[1].set_xlabel('Time (days)')
    ax[2].set_xlabel('Time (days)')
    ax[0].set_ylabel('Reward')
    ax[1].set_ylabel('EPI')
    ax[2].set_ylabel('Penalties')
    ax[0].legend()
    fig.suptitle(f'Rewards over time')
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{dir}/rewards{cs_suffix}-{int(dt)}dt.png")

def plot_states(data, dt, cs_suffix):
    """Plot the optimized control trajectories."""
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
        if data[method]:
            if method == "RL":
                X = data[method]["X"]
                axes[0,0].step(t[:-1], X[1, :], color=f"C0", label=method.capitalize(), where='post')
                axes[0,1].step(t[:-1], X[0, :], color=f"C0", label=method.capitalize(), where='post')
                axes[1,0].step(t[:-1], X[2, :], color=f"C0", label=method.capitalize(), where='post')
                axes[1,1].step(t[:-1], X[5, :], color=f"C0", label=method.capitalize(), where='post')
            else:
                for j, H in enumerate(data[method].keys()):
                    X = data[method][H]["X"]
                    t = np.arange(0, X.shape[-1]*dt, dt)/86400
                    all_Hs = sorted(data[method].keys())
                    max_H = all_Hs[-1]
                    current_color = cm.YlGn((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)

                    axes[0,0].step(t, X[2,:], color=current_color, label=method.capitalize() + f"{H}H", where='post')
                    axes[0,1].step(t, X[0, :], color=current_color, label=method.capitalize() + f"{H}H", where='post')
                    axes[1,0].step(t, X[15, :], color=current_color, label=method.capitalize() + f"{H}H", where='post')
                    axes[1,1].step(t, X[25, :], color=current_color, label=method.capitalize() + f"{H}H", where='post')


            for i, ax in enumerate(axes.flat):
                ax.set_ylabel(state_labels_with_units[i])
    for i, ax in enumerate(axes.flat):
        if bounds[i]:
            ax.hlines(bounds[i], t[0], t[-1], color='grey', linestyle='--', label='Boundaries')
                
    axes[0,1].legend(loc='upper left')
    fig.supxlabel('Time (days)')
    fig.supylabel('State variable')
    fig.suptitle('Closed-loop State Trajectories')
    fig.tight_layout()
    # plt.show()
    fig.savefig(f"{dir}/states{cs_suffix}-{int(dt)}dt.png")

def plot_runtime(data, dt, cs_suffix):
    """Plot a barplot of the average solver times for both methods."""
    methods = list(data.keys())
    avg_times = []
    labels = []
    for method in methods:
        if method == "RL":
            continue
        if data[method]:
            print(method)
            for H in data[method].keys():
                times = data[method][H]["times"]
                avg_times.append(np.mean(times))
                labels.append(f"{H}H")
        # else:
            # avg_times.append(0)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)
    ax.bar(labels, avg_times, color=[cm.YlGn((i+1) / len(avg_times)) for i in range(len(avg_times))])
    
    ax.set_ylabel("Average Solver Time (s)")
    ax.set_title("Average Solver Time")
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{dir}/avg_solver_times{cs_suffix}-{int(dt)}dt.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", required=True, help="Optimization methods used.")
    parser.add_argument("--constraints", action="store_true", help="OCP with constraints.")
    args = parser.parse_args()
    month = "june"
    method = "-".join(args.methods)
    dir = f"figures/test/{method}/{month}"
    os.makedirs(dir, exist_ok=True)
    if args.constraints:
        cs_suffix = "-cs"
    else:
        cs_suffix = ""
    dt = 300
    data = load_data(args.methods, month, cs_suffix)

    rewards_plot(data, dt, cs_suffix)
    plot_states(data, dt, cs_suffix)
    plot_control_trajectories(data, dt)
    plot_runtime(data, dt, cs_suffix)