import argparse
import os

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

                axes[0,0].step(t[:-1], U[0, 1:], color=COLORS[0], label=method.capitalize(), where='post')
                axes[0,1].step(t[:-1], U[1, 1:], color=COLORS[0], label=method.capitalize(), where='post')
                axes[1,0].step(t[:-1], U[2, 1:], color=COLORS[0], label=method.capitalize(), where='post')
                axes[1,1].step(t[:-1], U[3, 1:], color=COLORS[0], label=method.capitalize(), where='post')
                axes[2,0].step(t[:-1], U[4, 1:], color=COLORS[0], label=method.capitalize(), where='post')
                axes[2,1].step(t[:-1], U[5, 1:], color=COLORS[0], label=method.capitalize(), where='post')
            elif method == "finite-difference" or method == "exact":
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

            elif method == "rlmpc":
                for H in data[method].keys():
                    U = data[method][H]["U"]
                    t = np.arange(0, U.shape[-1]*dt, dt)/86400
                    all_Hs = sorted(data[method].keys())
                    max_H = all_Hs[-1]
                    current_color = cm.OrRd((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)

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
    fig.savefig(f"{dir}/control-inputs-{int(dt)}dt.png")


def plot_states(data, dt, labels):
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
        if method == "rl":
            X = data[method]["X"]
            t = np.arange(0, X.shape[-1]*dt, dt)/86400
            axes[0,0].step(t[:], X[1, :], color=COLORS[i], label=labels[i], where='post')
            axes[0,1].step(t[:], X[0, :], color=COLORS[i], label=labels[i], where='post')
            axes[1,0].step(t[:], X[2, :], color=COLORS[i], label=labels[i], where='post')
            axes[1,1].step(t[:], X[5, :], color=COLORS[i], label=labels[i], where='post')
        else:
            for j, H in enumerate(data[method].keys()):
                print(f"Plotting {method} with horizon {H}")
                X = data[method][H]["X"]
                t = np.arange(0, X.shape[-1]*dt, dt)/86400
                all_Hs = sorted(data[method].keys())
                max_H = all_Hs[-1]
                # if method == "rlmpc":
                #     current_color = cm.OrRd((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)
                # else:
                #     current_color = cm.YlGn((H+1 - all_Hs[0]) / (max_H+1 - all_Hs[0]) if max_H != all_Hs[0] else 0.5)

                axes[0,0].step(t, X[2,:], color=COLORS[i], label=labels[i], where='post')
                axes[0,1].step(t, X[0, :], color=COLORS[i], label=labels[i], where='post')
                axes[1,0].step(t, X[15, :], color=COLORS[i], label=labels[i], where='post')
                axes[1,1].step(t, X[25, :], color=COLORS[i], label=labels[i], where='post')


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
    fig.savefig(f"{dir}/states-{int(dt)}dt.png")

def plot_runtime(data, dt, labels):
    """Plot a barplot of the average solver times for both methods."""
    avg_times = []
    for method in data.keys():
        if method == "rl":
            continue
        if data[method]:
            print(method)
            # for H in data[method].keys():
            H = 1
            times = data[method][H]["times"]
            avg_times.append(np.mean(times))
        # else:
            # avg_times.append(0)
    fig, ax = plt.subplots(figsize=(4, 3), dpi=180)
    ax.bar(labels, avg_times, color=[cm.YlGn((i+1) / len(avg_times)) for i in range(len(avg_times))])
    ax.tick_params(axis='x', rotation=45)
    
    ax.set_ylabel("Average Solver Time (s)")
    ax.set_title("Average Solver Time")
    fig.tight_layout()
    plt.show()
    fig.savefig(f"{dir}/avg_solver_times-{int(dt)}dt.png")

def create_rewards_plot():
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), dpi=180)
    ax[0].set_xlabel('Time (days)')
    ax[1].set_xlabel('Time (days)')
    ax[2].set_xlabel('Time (days)')
    ax[0].set_ylabel('Cumulative Reward')
    ax[1].set_ylabel('EPI')
    ax[2].set_ylabel('Cumulative Penalty')
    # ax[0].legend()
    fig.suptitle(f'Rewards over time')
    return fig, ax

def plot_rewards_data(fig, ax, data, dt):
    ### Plot RL
    RL = data["rl"]
    rewards = RL["rewards"]
    EPI = RL["EPI"]
    penalties = RL["penalties"]

    t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
    ax[0].step(t, rewards.cumsum(), label=f"RL", color=COLORS[0])
    ax[1].step(t, EPI.cumsum(), color=COLORS[0])
    ax[2].step(t, penalties.cumsum(), color=COLORS[0])

    ### Plot MPC
    MPC = data["mpc"][1]
    rewards = MPC["rewards"]
    EPI = MPC["EPI"]
    penalties = MPC["penalties"]
    t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
    ax[0].plot(t, rewards.cumsum(), label=f"MPC (1H)", color=COLORS[1])
    ax[1].plot(t, EPI.cumsum(), color=COLORS[1])
    ax[2].plot(t, penalties.cumsum(), color=COLORS[1])

    ### Plot RL-MPC
    RL_MPC = data["terminal_constr"][1]
    rewards = RL_MPC["rewards"]
    EPI = RL_MPC["EPI"]
    penalties = RL_MPC["penalties"]
    t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
    ax[0].plot(t, rewards.cumsum(), label=f"RL-MPC (1H) Terminal State", color=COLORS[2])
    ax[1].plot(t, EPI.cumsum(), color=COLORS[2])    
    ax[2].plot(t, penalties.cumsum(), color=COLORS[2])

    RL_MPC = data["terminal_constr_pen"][1]
    rewards = RL_MPC["rewards"]
    EPI = RL_MPC["EPI"]
    penalties = RL_MPC["penalties"]
    t = np.arange(0, rewards.shape[-1]*dt, dt)/86400
    ax[0].plot(t, rewards.cumsum(), label=f"RL-MPC (1H) Terminal State + Penalty", color=COLORS[3])
    ax[1].plot(t, EPI.cumsum(), color=COLORS[3])
    ax[2].plot(t, penalties.cumsum(), color=COLORS[3])


    fig.legend(loc='upper left')
    fig.savefig(f"{dir}/rewards-{int(dt)}dt.png")


COLORS = ['C0', 'C1', 'C2', 'C3']
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--methods", type=str, nargs="+", required=True, help="Optimization methods used.")
    parser.add_argument("--constraints", action="store_true", help="OCP with constraints.")
    args = parser.parse_args()
    month = "june"
    project = "GL-MPC-RL"
    method = "-".join(args.methods)
    dir = os.path.join("figures", project)
    os.makedirs(dir, exist_ok=True)
    dt = 300

    rlmpc_data = load_rl_mpc_data("GL-MPC-RL", ["terminal_constr", "terminal_constr_pen"])
    mpc_data = load_mpc_data("GL-MPC-RL")
    rl_data = load_rl_data(project, "graceful-planet-22")
    data = {**rl_data, **mpc_data, **rlmpc_data}
    fig, ax = create_rewards_plot()
    plot_rewards_data(fig, ax, data, dt)
    plot_states(data, dt, labels=["RL", "MPC", "RL-MPC Terminal state", "RL-MPC Terminal state + Penalty"])
    plot_runtime(data, dt, labels=["MPC", "Terminal state", "Terminal S + Pen"])
    # data = load_data(args.methods, month)

    # rewards_plot(data, dt, cs_suffix)
    # plot_states(data, dt, cs_suffix)
    # plot_control_trajectories(data, dt)
    # plot_runtime(data, dt, cs_suffix)
