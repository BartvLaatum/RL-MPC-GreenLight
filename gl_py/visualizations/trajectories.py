import numpy as np
import matplotlib.pyplot as plt

import visualizations.plot_config

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
    # fig.savefig(f"figures/exact-finite/{approach}/{month}/control-inputs-{int(dt)}dt.png")

def plot_states(data, dt):
    """Plot the optimized control trajectories."""
    state_labels_with_units = {
        0: r"Air Temperature ($^\circ$C)",
        1: r"CO$_2$ (mg/m$^3$)",
        2: r"Vapor Pressure (Pa)",
        3: r"Fruit Weight (DM mg/m$^2$)"
    }
    WIDTH = 175 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True)

    for i, method in enumerate(data.keys()):
        X = data[method]["X"]
        t = np.arange(0, X.shape[-1]*dt, dt)/86400

        axes[0,0].step(t, X[2,:], color=f"C{i}", label=method, where='post')
        axes[0,1].step(t, X[0, :], color=f"C{i}", label=method, where='post')
        axes[1,0].step(t, X[15, :], color=f"C{i}", label=method, where='post')
        axes[1,1].step(t, X[25, :], color=f"C{i}", label=method, where='post')

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(state_labels_with_units[i])

    axes[0,0].legend()
    fig.supxlabel('Time (days)')
    fig.supylabel('State variable')
    fig.suptitle('Closed-loop State Trajectories')
    fig.tight_layout()
    plt.show()
    # fig.savefig(f"figures/exact-finite/{approach}/{month}/states-{int(dt)}dt.png")

def load_data(methods, month):
    """Load the data from the CSV files."""
    data = {
        "exact": {},
        "finite-difference": {}
    }
    for method in methods:
        
        U = np.loadtxt(f"results/{method}/{approach}/{month}/control-inputs-300dt.csv", delimiter=",").T
        X = np.loadtxt(f"results/{method}/{approach}/{month}/states-300dt.csv", delimiter=",").T
        data[method] = {
            "U": U,
            "X": X
        }
    return data

if __name__ == "__main__":
    approach = "single"
    method = ["exact", "finite-difference"]
    month = "june"
    data = load_data(method, month)
    plot_control_trajectories(data, 300)
    plot_states(data, 300)