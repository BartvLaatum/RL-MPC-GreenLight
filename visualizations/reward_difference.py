import argparse
import os
from typing import List, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_config





def load_rl_mpc_data(project, experiment_folder, location, growth_years, start_day, horizon):
    data = {"rlmpc": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "rlmpc", experiment_folder)

    for growth_year in growth_years:
        U = np.loadtxt(os.path.join(BASE_DIR, f"control-inputs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        X = np.loadtxt(os.path.join(BASE_DIR, f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        times = np.loadtxt(os.path.join(BASE_DIR, f"times-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        rewards = np.loadtxt(os.path.join(BASE_DIR, f"rewards-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        EPI = np.loadtxt(os.path.join(BASE_DIR, f"EPI-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        penalties = np.loadtxt(os.path.join(BASE_DIR, f"penalties-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        data["rlmpc"][growth_year] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }
    return data

def load_rl_data(project, model_name, location, growth_years, start_day):
    data = {"rl": {}}
    BASE_DIR = os.path.join("results", project, "deterministic", "ppo")

    for growth_year in growth_years:
        rl_data = pd.read_csv(os.path.join(BASE_DIR, f"{model_name}-{location}-{growth_year}-{start_day}.csv"))
        
        U = rl_data[["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]].values.T
        X = rl_data[["co2_air", "temp_air", "rh_air", "pipe_temp", "cFruit"]].values.T
        EPI = rl_data["EPI"].values.T
        rewards = rl_data["Rewards"].values.T
        penalties = rl_data["Penalty"].values.T
        data["rl"][growth_year] = {
            "U": U,
            "X": X,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data

def load_mpc_data(project, mpc_folder, location, growth_years, start_day, horizon=1):
    data = {"mpc": {}}
    data["mpc"] = {growth_year: {} for growth_year in growth_years}
    BASE_DIR = os.path.join("results", project, "deterministic", "mpc", mpc_folder)

    for growth_year in growth_years:
        U = np.loadtxt(os.path.join(BASE_DIR, f"control-inputs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        X = np.loadtxt(os.path.join(BASE_DIR, f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        times = np.loadtxt(os.path.join(BASE_DIR, f"times-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        rewards = np.loadtxt(os.path.join(BASE_DIR, f"rewards-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        EPI = np.loadtxt(os.path.join(BASE_DIR, f"EPI-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        penalties = np.loadtxt(os.path.join(BASE_DIR, f"penalties-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"), delimiter=",").T
        data["mpc"][growth_year] = {
            "U": U,
            "X": X,
            "times": times,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data


def plot_reward_difference(
    diff_train: np.ndarray,
    diff_test_mean: np.ndarray,
    diff_test_std: np.ndarray,
    dt: int,
    train_label: str = "Training",
    test_label: str = "Test",
    show_std: bool = True,
    output_path: str = None,
    show: bool = False,
    metric: str = "rewards",
):
    """
    Plot the difference between cumulative rewards for RLMPC and RL
    for both training and test environments in the same subplot.
    
    The difference is calculated as:
        cumsum_rlmpc - cumsum_rl
    
    Parameters
    ----------
    diff_train : np.ndarray
        Difference array for training environment.
    diff_test_mean : np.ndarray
        Mean difference array across test environments.
    diff_test_std : np.ndarray
        Standard deviation of differences across test environments.
    dt : int
        Time step in seconds.
    train_label : str
        Label for training environment line.
    test_label : str
        Label for test environment line.
    show_std : bool
        Whether to show std deviation as shaded region.
    output_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot interactively.
    """
    # Convert time to days (each timestamp is dt seconds, which is 5 minutes = 300s)
    n_steps_train = len(diff_train)
    n_steps_test = len(diff_test_mean)
    time_days_train = np.arange(n_steps_train) * dt / 86400
    time_days_test = np.arange(n_steps_test) * dt / 86400
    
    # Create figure
    WIDTH = 173.8 / 3 * 0.03937
    HEIGHT = WIDTH * 0.75

    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, HEIGHT), dpi=300)
    colors = plt.cm.plasma(np.linspace(0.2, 0.8, 2))
    # Plot training line
    ax.step(time_days_train, diff_train, color=colors[0], label=train_label)

    # Plot test line with optional std shading
    ax.step(time_days_test, diff_test_mean, color=colors[1], label=test_label)
    if show_std and diff_test_std is not None:
        ax.fill_between(
            time_days_test,
            diff_test_mean - diff_test_std,
            diff_test_mean + diff_test_std,
            color=colors[1],
            alpha=0.2,
        )

    # ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    ax.set_xlabel("Time (days)")
    if metric == "rewards":
        ax.set_ylabel(r"$\Delta\mathcal{J}(t)$")
        ax.set_yticks(np.linspace(-0.25, 0.25, 3))
        ax.set_ylim(-0.25, 0.25)
    elif metric == "EPI":
        ax.set_ylabel(r"Cumulative $\Delta$ EPI")
        ax.set_yticks(np.linspace(0., 0.3, 3))
    elif metric == "penalties":
        ax.set_ylabel(r"Cumulative $\Delta$ Penalty")
        ax.set_yticks(np.linspace(-0.4, 0.4, 5))

    ax.legend()
    fig.tight_layout()
    # Show three x and y-ticks on the axes
    ax.set_xticks(np.linspace(0, 1, 3))
    
    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved reward difference plot to {output_path}")

    if show:
        plt.show()

    plt.close(fig)


def plot_reward_difference_grid(
    diff_rl_data: Dict[str, Dict[str, np.ndarray]],
    diff_mpc_data: Dict[str, Dict[str, np.ndarray]],
    dt: int,
    train_label: str = "Training",
    test_label: str = "Test",
    show_std: bool = True,
    output_path: str = None,
    show: bool = False,
):
    """
    Plot the difference between cumulative metrics for RL-MPC vs RL and RL-MPC vs MPC
    in a 3x2 subplot grid.
    
    Parameters
    ----------
    diff_rl_data : dict
        Dictionary with structure:
        {
            "rewards": {"train_mean": array, "train_std": array, "test_mean": array, "test_std": array},
            "EPI": {...},
            "penalties": {...}
        }
        Difference data for RL-MPC vs RL comparison.
    diff_mpc_data : dict
        Same structure as diff_rl_data for RL-MPC vs MPC comparison.
    dt : int
        Time step in seconds.
    train_label : str
        Label for training environment line.
    test_label : str
        Label for test environment line.
    show_std : bool
        Whether to show std deviation as shaded region.
    output_path : str, optional
        Path to save the figure.
    show : bool
        Whether to display the plot interactively.
    """
    metrics = ["rewards", "EPI", "penalties"]
    y_labels = {
        "rewards": r"$\Delta\mathcal{J}(x)$",
        "EPI": r"$\Delta$ EPI (EUR/m$^2$)",
        "penalties": r"$\Delta$ Penalty",
    }
    # column_titles = ["RL-MPC vs RL", "RL-MPC vs MPC"]
    
    # Create figure with 3 rows x 2 columns
    WIDTH = 173.8 / 2 * 0.03937
    HEIGHT = WIDTH * 1.5
    
    fig, axes = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), dpi=300, 
                              sharex=True, sharey='row')
    
    # colors = plt.cm.plasma(np.linspace(0.2, 0.8, 2))
    colors = ["#1f77b4", "#00a693"]
    
    # Data sources for each column
    data_sources = [diff_rl_data, diff_mpc_data]
    
    for row, metric in enumerate(metrics):
        for col, data in enumerate(data_sources):
            ax = axes[row, col]
            
            # Get data for this metric
            metric_data = data[metric]
            diff_train_mean = metric_data["train_mean"]
            diff_train_std = metric_data["train_std"]
            diff_test_mean = metric_data["test_mean"]
            diff_test_std = metric_data["test_std"]
            
            # Convert time to days
            n_steps_train = len(diff_train_mean)
            n_steps_test = len(diff_test_mean)
            time_days_train = np.arange(n_steps_train) * dt / 86400
            time_days_test = np.arange(n_steps_test) * dt / 86400
            
            # Plot training line
            ax.step(time_days_train, diff_train_mean, color=colors[col], linestyle="-",
                   label=train_label if row == 0 and col == 0 else None)
            
            # Plot test line with optional std shading
            ax.step(time_days_test, diff_test_mean, color=colors[col], linestyle="--",
                   label=test_label if row == 0 and col == 0 else None)
            
            if show_std and diff_test_std is not None:
                ax.fill_between(
                    time_days_test,
                    diff_test_mean - diff_test_std,
                    diff_test_mean + diff_test_std,
                    color=colors[col],
                    alpha=0.2,
                )
            
            # Set column titles on top row
            # if row == 0:
                # ax.set_title(title)
            
            # Set y-label on left column only
            if col == 0:
                ax.set_ylabel(y_labels[metric])
            
            # Set x-label on bottom row only
            if row == 2:
                ax.set_xlabel("Time (days)")
            
            # Set x-ticks
            ax.set_xticks(np.linspace(0, 1, 3))
    
    # Add legend to first subplot
    legend = axes[0, 0].legend(loc='upper left')
    for handle in legend.legend_handles:
        if hasattr(handle, "set_color"):
            handle.set_color('dimgray')
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, bbox_inches="tight")
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved reward difference grid plot to {output_path}")

    if show:
        plt.show()

    plt.close(fig)

def compute_reward_difference(rl_rewards: np.ndarray, rlmpc_rewards: np.ndarray):
    """Compute the difference between cumulative rewards for RLMPC and RL."""
    return np.cumsum(rlmpc_rewards) - np.cumsum(rl_rewards)

def load_reward_data(
    project: str, 
    rl_model: str, 
    rlmpc_folder: str, 
    location: str, 
    growth_years: List[int], 
    start_day: int, 
    horizon: int,
    mpc_folder: str = "mpc_linear_solver_ma57"
):
    """Load the reward data for the given project, location, and year."""
    rl_data = load_rl_data(project, rl_model, location, growth_years, start_day)
    rlmpc_data = load_rl_mpc_data(project, rlmpc_folder, location, growth_years, start_day, horizon)
    mpc_data = load_mpc_data(project, mpc_folder, location, growth_years, start_day, horizon)
    return rl_data, rlmpc_data, mpc_data

def compute_reward_difference_per_year(
    rewards_dict: Dict[int, Dict[str, np.ndarray]], 
    reference_rewards_dict: Dict[int, Dict[str, np.ndarray]],
    method: str,
    reference_method: str,
    growth_years: List[int],
    metric: str = "rewards",
):
    """Compute the difference between metrics of a method and a reference method for each growth year."""
    diff_rewards = {
        year: 
            np.cumsum(reference_rewards_dict[reference_method][year][metric]) - np.cumsum(rewards_dict[method][year][metric]) for year in growth_years
    }
    return diff_rewards

def compute_statistics(rewards_dict: Dict[int, np.ndarray], metric: str = "rewards"):
    """Compute the statistics for the rewards."""
    rewards_mean = np.mean([rewards_dict[year] for year in rewards_dict.keys()], axis=0)
    rewards_std = np.std([rewards_dict[year] for year in rewards_dict.keys()], axis=0)
    return rewards_mean, rewards_std

def compute_cumulative_rewards(rewards_dict: Dict[int, Dict[str, np.ndarray]], method: str, growth_years: List[int], metric: str = "rewards"):
    """Compute the cumulative rewards for the given rewards dictionary."""
    print(rewards_dict.keys())
    cumulative_rewards = {year: np.cumsum(rewards_dict[method][year][metric]) for year in growth_years}
    return cumulative_rewards

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot difference between RLMPC and RL cumulative rewards."
    )
    
    # Data loading arguments
    parser.add_argument(
        "--project", type=str, default="GL-MPC-RL",
        help="Project name for data loading."
    )
    parser.add_argument(
        "--location", type=str, default="Netherlands",
        help="Location for data loading."
    )
    parser.add_argument(
        "--train_years", type=int, nargs="+", default=[2023],
        help="Training environment year."
    )
    parser.add_argument(
        "--test_years", type=int, nargs="+", default=[2020],
        help="Test environment year(s). Multiple years will be averaged."
    )
    parser.add_argument(
        "--start_day", type=int, default=151,
        help="Start day for data loading."
    )
    parser.add_argument(
        "--dt", type=int, default=300,
        help="Time step in seconds (default: 300 = 5 minutes)."
    )
    
    # RL arguments
    parser.add_argument(
        "--rl_model", type=str, required=True,
        help="Name of the RL model to use as baseline."
    )
    
    # RLMPC arguments
    parser.add_argument(
        "--rlmpc_folder", type=str, required=True,
        help="Folder containing RLMPC data."
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Horizon for RLMPC data loading."
    )
    
    # Labels
    parser.add_argument(
        "--train_label", type=str, default="Training",
        help="Label for training environment line."
    )
    parser.add_argument(
        "--test_label", type=str, default="Test",
        help="Label for test environment line."
    )

    # Output control
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="Save plots to files."
    )
    parser.add_argument(
        "--show", action="store_true", default=False,
        help="Show plots interactively."
    )
    parser.add_argument(
        "--output_folder", type=str, default="relative_difference",
        help="Directory to save plots."
    )
    parser.add_argument(
        "--output_prefix", type=str, default="",
        help="Prefix for output filenames."
    )
    parser.add_argument(
        "--metric", type=str, default="rewards",
        help="Metric to plot."
    )
    
    args = parser.parse_args()

    # Load training environment data
    print(f"Loading training environment data (year={args.train_years})...")
    rl_rewards_train, rlmpc_rewards_train, mpc_rewards_train = load_reward_data(
        args.project,
        args.rl_model,
        args.rlmpc_folder,
        args.location,
        args.train_years,
        args.start_day,
        args.horizon
    )

    # Load test environment data for each test year
    print(f"Loading test environment data (years={args.test_years})...")
    rl_data_test, rlmpc_data_test, mpc_data_test = load_reward_data(
        args.project,
        args.rl_model,
        args.rlmpc_folder,
        args.location,
        args.test_years,
        args.start_day,
        args.horizon
    )

    # Compute differences for all three metrics
    metrics = ["rewards", "EPI", "penalties"]
    diff_rl_data = {}
    diff_mpc_data = {}
    
    for metric in metrics:
        # RL-MPC vs RL differences
        diff_train = compute_reward_difference_per_year(
            rl_rewards_train, rlmpc_rewards_train, "rl", "rlmpc", args.train_years, metric
        )
        diff_train_mean, diff_train_std = compute_statistics(diff_train, args.train_years)
        
        diff_test = compute_reward_difference_per_year(
            rl_data_test, rlmpc_data_test, "rl", "rlmpc", args.test_years, metric
        )
        diff_test_mean, diff_test_std = compute_statistics(diff_test, args.test_years)
        
        diff_rl_data[metric] = {
            "train_mean": diff_train_mean,
            "train_std": diff_train_std,
            "test_mean": diff_test_mean,
            "test_std": diff_test_std,
        }
        
        # RL-MPC vs MPC differences
        diff_mpc_train = compute_reward_difference_per_year(
            mpc_rewards_train, rlmpc_rewards_train, "mpc", "rlmpc", args.train_years, metric
        )
        diff_mpc_train_mean, diff_mpc_train_std = compute_statistics(diff_mpc_train, args.train_years)
        
        diff_mpc_test = compute_reward_difference_per_year(
            mpc_data_test, rlmpc_data_test, "mpc", "rlmpc", args.test_years, metric
        )
        diff_mpc_test_mean, diff_mpc_test_std = compute_statistics(diff_mpc_test, args.test_years)
        
        diff_mpc_data[metric] = {
            "train_mean": diff_mpc_train_mean,
            "train_std": diff_mpc_train_std,
            "test_mean": diff_mpc_test_mean,
            "test_std": diff_mpc_test_std,
        }

    # Compute cumulative rewards for summary (using the specified metric)
    rl_cumsum_train = compute_cumulative_rewards(rl_rewards_train, "rl", args.train_years, args.metric)
    rlmpc_cumsum_train = compute_cumulative_rewards(rlmpc_rewards_train, "rlmpc", args.train_years, args.metric)
    rl_cumsum_train_mean, rl_cumsum_train_std = compute_statistics(rl_cumsum_train, args.train_years)
    rlmpc_cumsum_train_mean, rlmpc_cumsum_train_std = compute_statistics(rlmpc_cumsum_train, args.train_years)
    mpc_cumsum_train = compute_cumulative_rewards(mpc_rewards_train, "mpc", args.train_years, args.metric)
    mpc_cumsum_train_mean, mpc_cumsum_train_std = compute_statistics(mpc_cumsum_train, args.train_years)

    rl_cumsum_test = compute_cumulative_rewards(rl_data_test, "rl", args.test_years, args.metric)
    rlmpc_cumsum_test = compute_cumulative_rewards(rlmpc_data_test, "rlmpc", args.test_years, args.metric)
    rl_cumsum_test_mean, rl_cumsum_test_std = compute_statistics(rl_cumsum_test, args.test_years)
    rlmpc_cumsum_test_mean, rlmpc_cumsum_test_std = compute_statistics(rlmpc_cumsum_test, args.test_years)
    mpc_cumsum_test = compute_cumulative_rewards(mpc_data_test, "mpc", args.test_years, args.metric)
    mpc_cumsum_test_mean, mpc_cumsum_test_std = compute_statistics(mpc_cumsum_test, args.test_years)

    # Set up output directory
    output_dir = os.path.join("figures", args.project, args.output_folder)
    if args.save:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Output directory: {output_dir}")

    output_path = None
    if args.save:
        output_path = os.path.join(
            output_dir,
            f"{args.metric}_{args.rlmpc_folder}_vs_{args.rl_model}.png"
        )

    # # Generate single metric plots for difference between RL and RL-MPC
    # plot_reward_difference(
    #     diff_train=diff_rl_data[args.metric]["train_mean"],
    #     diff_test_mean=diff_rl_data[args.metric]["test_mean"],
    #     diff_test_std=diff_rl_data[args.metric]["test_std"],
    #     dt=args.dt,
    #     train_label=args.train_label,
    #     test_label=args.test_label,
    #     show_std=len(args.test_years) > 1,
    #     output_path=output_path,
    #     show=args.show,
    #     metric=args.metric,
    # )
    # # Generate single metric plot for difference between MPC and RL-MPC
    # plot_reward_difference(
    #     diff_train=diff_mpc_data[args.metric]["train_mean"],
    #     diff_test_mean=diff_mpc_data[args.metric]["test_mean"],
    #     diff_test_std=diff_mpc_data[args.metric]["test_std"],
    #     dt=args.dt,
    #     train_label=args.train_label,
    #     test_label=args.test_label,
    #     show_std=len(args.test_years) > 1,
    #     output_path=output_path.replace(f"{args.rl_model}.png", "mpc.png") if output_path else None,
    #     show=args.show,
    #     metric=args.metric,
    # )

    # Generate the 3x2 grid plot with all metrics
    grid_output_path = None
    if args.save:
        grid_output_path = os.path.join(
            output_dir,
            f"all_metrics_{args.rlmpc_folder}_comparison.png"
        )

    plot_reward_difference_grid(
        diff_rl_data=diff_rl_data,
        diff_mpc_data=diff_mpc_data,
        dt=args.dt,
        train_label=args.train_label,
        test_label=args.test_label,
        show_std=len(args.test_years) > 1,
        output_path=grid_output_path,
        show=args.show,
    )
    
    # Print summary
    diff_train_mean = diff_rl_data[args.metric]["train_mean"]
    diff_train_std = diff_rl_data[args.metric]["train_std"]
    diff_test_mean = diff_rl_data[args.metric]["test_mean"]
    diff_test_std = diff_rl_data[args.metric]["test_std"]
    diff_mpc_train_mean = diff_mpc_data[args.metric]["train_mean"]
    diff_mpc_train_std = diff_mpc_data[args.metric]["train_std"]
    diff_mpc_test_mean = diff_mpc_data[args.metric]["test_mean"]
    diff_mpc_test_std = diff_mpc_data[args.metric]["test_std"]
    
    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'='*60}")
    print(f"Training Environment (year={args.train_years}):")
    print(f"  Final cumulative RL reward: {rl_cumsum_train_mean[-1]:.3f} ± {rl_cumsum_train_std[-1]:.3f}")
    print(f"  Final cumulative RLMPC reward: {rlmpc_cumsum_train_mean[-1]:.3f} ± {rlmpc_cumsum_train_std[-1]:.3f}")
    print(f"  Final difference: {diff_train_mean[-1]:.3f} ± {diff_train_std[-1]:.3f}")
    print(f"  Final cumulative MPC reward: {mpc_cumsum_train_mean[-1]:.3f} ± {mpc_cumsum_train_std[-1]:.3f}")
    print(f"  Final difference MPC: {diff_mpc_train_mean[-1]:.3f} ± {diff_mpc_train_std[-1]:.3f}")

    print(f"\nTest Environments (years={args.test_years}):")
    print(f"  Mean final cumulative RL reward: {rl_cumsum_test_mean[-1]:.3f} ± {rl_cumsum_test_std[-1]:.3f}")
    print(f"  Mean final cumulative RLMPC reward: {rlmpc_cumsum_test_mean[-1]:.3f} ± {rlmpc_cumsum_test_std[-1]:.3f}")
    print(f"  Mean final difference: {diff_test_mean[-1]:.3f} ± {diff_test_std[-1]:.3f}")
    
    print(f"  Mean final cumulative MPC reward: {mpc_cumsum_test_mean[-1]:.3f} ± {mpc_cumsum_test_std[-1]:.3f}")
    print(f"  Mean final difference MPC: {diff_mpc_test_mean[-1]:.3f} ± {diff_mpc_test_std[-1]:.3f}")

    print(f"\n  Per-year final differences for {args.metric}:")
    for year in args.test_years:
        diff_test = compute_reward_difference_per_year(
            rl_data_test, rlmpc_data_test, "rl", "rlmpc", [year], args.metric
        )
        diff_mpc_test = compute_reward_difference_per_year(
            mpc_data_test, rlmpc_data_test, "mpc", "rlmpc", [year], args.metric
        )
        print(f"    {year} (RL-MPC vs RL): {diff_test[year][-1]:.3f}")
        print(f"    {year} (RL-MPC vs MPC): {diff_mpc_test[year][-1]:.3f}")
    print(f"{'='*60}")
