import os
import numpy as np
import matplotlib.pyplot as plt
from solver_costs import load_rlmpc_results
import plot_config

def plot_horizon_vs_mpc_fraction(train_data_summary, test_data_summary, horizons, labels, output_path=None, show=False):
    """
    Plot the fraction of time MPC controller was chosen vs horizon length.
    
    Parameters
    ----------
    data : dict
        Data dictionary from load_rlmpc_results
    train_suffixes : list
        List of experiment train suffixes
    test_suffixes : list
        List of experiment test suffixes
    horizons : list
        List of horizon lengths
    output_path : str, optional
        Path to save the figure
    show : bool
        Whether to display the plot
    """
    WIDTH = 75 * 0.0393700787  # Convert mm to inches
    HEIGHT = WIDTH * 0.8
    
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # Colors for different suffixes
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 2))

    train_fractions = [train_data_summary[H]["fraction"] for H in horizons]
    test_fractions = [test_data_summary[H]["fraction"] for H in horizons]
    train_std_fractions = [train_data_summary[H]["std_fraction"] for H in horizons]
    test_std_fractions = [test_data_summary[H]["std_fraction"] for H in horizons]
    ax.plot(horizons, train_fractions, 'o-', color=colors[0], markersize=8, linewidth=2, label=labels[0], alpha=0.9)
    ax.plot(horizons, test_fractions, 'o-', color=colors[1], markersize=8, linewidth=2, label=labels[1], alpha=0.9)
    ax.errorbar(horizons, train_fractions, yerr=train_std_fractions, color=colors[0], markersize=8, linewidth=2, label=labels[0], alpha=0.9)
    ax.errorbar(horizons, test_fractions, yerr=test_std_fractions, color=colors[1], markersize=8, linewidth=2, label=labels[1], alpha=0.9)
    
    ax.set_xlabel('Horizon Length ($H$)')
    ax.set_ylabel('Fraction MPC Chosen')
    ax.set_xticks(horizons)
    ax.set_ylim(0, 1)
    # ax.set_xlim(min(horizons) - 0.5, max(horizons) + 0.5)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, linewidth=1)
    
    # Add legend if multiple suffixes
    # if len(suffixes) > 1:
    #     ax.legend(loc='best', framealpha=0.9)
    
    fig.tight_layout()
    
    if output_path is not None:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        fig.savefig(output_path)
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved horizon vs MPC fraction plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, ax

def compute_statistics(data, horizons):
    data_summary = {}
    for H in horizons:
        data_summary[H] = {}
        fractions = []
        for suffix in data.keys():
            chosen_is_mpc = data[suffix][H]["chosen_is_mpc"]
            fraction = np.mean(chosen_is_mpc)
            fractions.append(fraction)
        data_summary[H]["fraction"] = np.mean(fractions)
        data_summary[H]["std_fraction"] = np.std(fractions)
    return data_summary

if __name__ == "__main__":
    project = "GL-MPC-RL"
    train_suffixes = ["Netherlands-2023-151"]
    test_suffixes = [
        "Netherlands-2011-151",
        "Netherlands-2012-151",
        "Netherlands-2013-151",
        # "Netherlands-2014-151",
        "Netherlands-2015-151",
        "Netherlands-2016-151",
        # "Netherlands-2017-151",
        "Netherlands-2018-151",
        "Netherlands-2019-151",
        "Netherlands-2020-151",
    ]

    experiment_name = "rlmpc-switch-daily-glade-107-0.05"
    horizons = [1, 2]
    labels = ["Train", "Test"]
    train_data = load_rlmpc_results(project, train_suffixes, experiment_name, horizons)
    test_data = load_rlmpc_results(project, test_suffixes, experiment_name, horizons)
    train_data_summary = compute_statistics(train_data, horizons)
    test_data_summary = compute_statistics(test_data, horizons)

    # Print summary statistics
    print("\nSummary: Fraction of time MPC was chosen")
    for H in horizons:
        print(f"Horizon {H}:")
        print(f"  Train: {train_data_summary[H]['fraction']:.2f} ± {train_data_summary[H]['std_fraction']:.2f}")
        print(f"  Test: {test_data_summary[H]['fraction']:.2f} ± {test_data_summary[H]['std_fraction']:.2f}")
        print("-" * 45)
    # Create output directory and save plot
    output_dir = os.path.join("figures", project, f"{experiment_name}-analysis")
    output_path = os.path.join(output_dir, "horizon_vs_mpc_fraction.png")
    
    plot_horizon_vs_mpc_fraction(train_data_summary, test_data_summary, horizons, labels, output_path=output_path)
