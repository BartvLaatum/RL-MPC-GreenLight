import argparse

import numpy as np
import matplotlib.pyplot as plt

import plot_config

def plot_control_trajectories(data, approach, dt, cs_suffix):
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
            U = data[method]["U"]
            t = np.arange(0, U.shape[-1]*dt, dt)/86400

            axes[0,0].step(t[:-1], U[0, 1:], color=f"C{i}", label=method, where='post')
            axes[0,1].step(t[:-1], U[1, 1:], color=f"C{i}", label=method, where='post')
            axes[1,0].step(t[:-1], U[2, 1:], color=f"C{i}", label=method, where='post')
            axes[1,1].step(t[:-1], U[3, 1:], color=f"C{i}", label=method, where='post')
            axes[2,0].step(t[:-1], U[4, 1:], color=f"C{i}", label=method, where='post')
            axes[2,1].step(t[:-1], U[5, 1:], color=f"C{i}", label=method, where='post')

    for i, ax in enumerate(axes.flat):
        ax.set_ylabel(control_labels_with_units[i])
    axes[0,0].legend()
    fig.supxlabel('Time (days)')
    fig.supylabel('Control Input')
    fig.suptitle('Closed-loop Control Trajectories')
    fig.tight_layout()
    plt.show()
    # fig.savefig(f"figures/exact-finite/{month}/control-inputs{cs_suffix}-{int(dt)}dt.png")

def plot_states(data, approach, dt, cs_suffix):
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
            X = data[method]["X"]
            t = np.arange(0, X.shape[-1]*dt, dt)/86400

            axes[0,0].step(t, X[2,:], color=f"C{i}", label=method, where='post')
            axes[0,1].step(t, X[0, :], color=f"C{i}", label=method, where='post')
            axes[1,0].step(t, X[15, :], color=f"C{i}", label=method, where='post')
            axes[1,1].step(t, X[25, :], color=f"C{i}", label=method, where='post')

            for i, ax in enumerate(axes.flat):
                ax.set_ylabel(state_labels_with_units[i])
                if bounds[i]:
                    ax.hlines(bounds[i], t[0], t[-1], color='grey', linestyle='--', label='Boundaries')
                

    axes[0,0].legend(loc='upper left')
    fig.supxlabel('Time (days)')
    fig.supylabel('State variable')
    fig.suptitle('Closed-loop State Trajectories')
    fig.tight_layout()
    plt.show()
    # fig.savefig(f"figures/exact-finite/{month}/states{cs_suffix}-{int(dt)}dt.png")

def load_data(methods, approach, month, cs_suffix):
    """Load the data from the CSV files."""

    data = {
        "exact": {},
        "finite-difference": {}
    }

    for method in methods:
        if method == "exact":
            approach = "multi"
        elif method == "finite-difference":
            approach = "single"
        U = np.loadtxt(f"results/{method}/{approach}/{month}/control-inputs{cs_suffix}-300dt.csv", delimiter=",").T
        X = np.loadtxt(f"results/{method}/{approach}/{month}/states{cs_suffix}-300dt.csv", delimiter=",").T
        data[method] = {
            "U": U,
            "X": X
        }

    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--approach", type=str, default="single", help="Approach used for optimization.")
    parser.add_argument("--method", type=str, nargs="+", required=True, help="Optimization method used.")
    parser.add_argument("--constraints", action="store_true", help="OCP with constraints.")
    args = parser.parse_args()
    month = "june"
    if args.constraints:
        cs_suffix = "-cs"
    else:
        cs_suffix = ""
    dt = 300
    data = load_data(args.method, args.approach, month, cs_suffix)
    
    print(data.keys())
    plot_control_trajectories(data, args.approach, dt, cs_suffix)
    plot_states(data, args.approach, dt, cs_suffix)