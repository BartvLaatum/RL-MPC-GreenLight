import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plot_config
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from performance import plot_rewards, plot_states, plot_control_trajectories
from scipy import stats


@dataclass
class RunResult:
    """Container for a single experiment run result."""
    controller_type: str
    model_name: str
    growth_year: int
    start_day: int
    location: str
    total_reward: float
    total_epi: float
    total_penalty: float

def load_rl_performance(run_info: Dict) -> Tuple[float, float, float]:
    """Load performance metrics from an RL result file."""
    df = pd.read_csv(run_info["path"])
    
    reward_col = "Rewards" if "Rewards" in df.columns else "rewards"
    epi_col = "EPI" if "EPI" in df.columns else "epi"
    penalty_col = "Penalty" if "Penalty" in df.columns else "penalty"
    
    total_reward = df[reward_col].sum()
    total_epi = df[epi_col].sum() if epi_col in df.columns else 0.0
    total_penalty = df[penalty_col].sum() if penalty_col in df.columns else 0.0
    
    return total_reward, total_epi, total_penalty


def load_mpc_performance(
    results_dir: Path,
    mpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1
) -> Tuple[float, float, float]:
    """Load performance metrics from MPC result files."""
    mpc_dir = results_dir / "mpc" / mpc_folder
    
    rewards_file = mpc_dir / f"rewards-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    epi_file = mpc_dir / f"EPI-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    penalties_file = mpc_dir / f"penalties-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    
    rewards = np.loadtxt(rewards_file, delimiter=",")
    total_reward = np.sum(rewards)
    
    total_epi = 0.0
    if epi_file.exists():
        epi = np.loadtxt(epi_file, delimiter=",")
        total_epi = np.sum(epi)
    
    total_penalty = 0.0
    if penalties_file.exists():
        penalties = np.loadtxt(penalties_file, delimiter=",")
        total_penalty = np.sum(penalties)
    
    return total_reward, total_epi, total_penalty


def load_rlmpc_performance(
    results_dir: Path,
    rlmpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1
) -> Tuple[float, float, float]:
    """Load performance metrics from RL-MPC result files."""
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    
    rewards_file = rlmpc_dir / f"rewards-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    epi_file = rlmpc_dir / f"EPI-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    penalties_file = rlmpc_dir / f"penalties-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    
    rewards = np.loadtxt(rewards_file, delimiter=",")
    total_reward = np.sum(rewards)
    
    total_epi = 0.0
    if epi_file.exists():
        epi = np.loadtxt(epi_file, delimiter=",")
        total_epi = np.sum(epi)
    
    total_penalty = 0.0
    if penalties_file.exists():
        penalties = np.loadtxt(penalties_file, delimiter=",")
        total_penalty = np.sum(penalties)
    
    return total_reward, total_epi, total_penalty


def load_runs(
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    results_dir: Path,
    mpc_folder: str = "mpc_linear_solver_ma57",
    rlmpc_folder: Optional[str] = None,
    location: str = "Netherlands",
    horizon: int = 1
) -> Dict[str, List[RunResult]]:
    """
    Load all experiment runs for the specified parameters.
    
    Returns:
        Dictionary mapping controller type to list of RunResult objects.
    """
    if rlmpc_folder is None:
        rlmpc_folder = f"rlmpc-switch-{model_name}-0.05"
    
    results = {
        "rl": [],
        "mpc": [],
        "rlmpc": []
    }
    
    # Load RL runs
    for start_day in start_days:
        for year in growth_years:
            rl_file = results_dir / "ppo" / f"{model_name}-{location}-{year}-{start_day}.csv"
            if rl_file.exists():
                try:
                    reward, epi, penalty = load_rl_performance({"path": rl_file})
                    results["rl"].append(RunResult(
                        controller_type="rl",
                        model_name=model_name,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL run for year {year}, day {start_day}: {e}")
    
    # Load MPC runs
    mpc_dir = results_dir / "mpc" / mpc_folder
    if mpc_dir.exists():
        for start_day in start_days:
            for year in growth_years:
                try:
                    reward, epi, penalty = load_mpc_performance(
                        results_dir, mpc_folder, location, year, start_day, horizon
                    )
                    results["mpc"].append(RunResult(
                        controller_type="mpc",
                        model_name=mpc_folder,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty
                    ))
                except Exception as e:
                    pass  # Silently skip missing MPC runs
    
    # Load RL-MPC runs
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    if rlmpc_dir.exists():
        for start_day in start_days:
            for year in growth_years:
                try:
                    reward, epi, penalty = load_rlmpc_performance(
                        results_dir, rlmpc_folder, location, year, start_day, horizon
                    )
                    results["rlmpc"].append(RunResult(
                        controller_type="rlmpc",
                        model_name=rlmpc_folder,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty
                    ))
                except Exception as e:
                    pass  # Silently skip missing RL-MPC runs
    
    return results

def plot_average_rewards_over_time(data, dt=300, metric="rewards", method_labels=None, 
                                    method_colors=None, show_individual_years=True, 
                                    output_path=None, show=False, y_label=None):
    """
    Plot average reward metrics over time across multiple evaluation years.
    
    Parameters
    ----------
    data : dict
        Dictionary with structure {method_name: {year: {metric: array}}}
        Each method holds keys with evaluation years.
    dt : int
        Time step in seconds (default: 300)
    metric : str
        Metric to plot: "rewards", "EPI", or "penalties" (default: "rewards")
    method_labels : dict, optional
        Dictionary mapping method names to display labels
    method_colors : dict, optional
        Dictionary mapping method names to colors
    show_individual_years : bool
        If True, show individual year traces with low alpha (default: True)
    output_path : str, optional
        Path to save the figure
    show : bool
        If True, display the plot (default: False)
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    # Default labels and colors if not provided
    if method_labels is None:
        method_labels = {method: method.upper() for method in data.keys()}
    if method_colors is None:
        default_colors = {"rl": "C0", "mpc": "C1", "rlmpc": "C3", "rbc": "C2"}
        method_colors = {method: default_colors.get(method, f"C{i}") 
                        for i, method in enumerate(data.keys())}
    
    # Set up figure
    WIDTH = 173.8/2 * 0.03937
    HEIGHT = WIDTH * 0.75
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, HEIGHT), dpi=300)

    ax.set_xlabel('Time (days)')
    ax.set_ylabel(y_label)
    
    # Plot each method
    for method_name, year_data in data.items():
        if not year_data:
            continue

        # Collect data for all years
        all_trajectories = []
        max_length = 0

        for year, year_dict in year_data.items():
            if metric in year_dict:
                trajectory = year_dict[metric]
                cumulative = np.cumsum(trajectory)
                all_trajectories.append(cumulative)
                max_length = max(max_length, len(cumulative))

        if not all_trajectories:
            continue

        # Pad trajectories to same length (use last value for padding)
        padded_trajectories = []
        for traj in all_trajectories:
            if len(traj) < max_length:
                padding = np.full(max_length - len(traj), traj[-1])
                traj = np.concatenate([traj, padding])
            padded_trajectories.append(traj)

        trajectories_array = np.array(padded_trajectories)

        # Compute mean and std
        mean_trajectory = np.mean(trajectories_array, axis=0)
        std_trajectory = np.std(trajectories_array, axis=0)
        var_trajectory = np.var(trajectories_array, axis=0)

        # Time axis
        t = np.arange(0, len(mean_trajectory) * dt, dt) / 86400
        
        # Plot individual years if requested
        if show_individual_years:
            for traj in trajectories_array:
                ax.step(t, traj, color=method_colors[method_name], 
                       alpha=0.15, linewidth=0.8, where='post')
        
        # Plot mean trajectory (bold)
        ax.step(t, mean_trajectory, label=method_labels[method_name], 
               color=method_colors[method_name], linewidth=2.0, where='post')

        # Optionally add std shading
        # ax.fill_between(t, mean_trajectory - var_trajectory, 
        #                  mean_trajectory + var_trajectory,
        #                  color=method_colors[method_name], alpha=0.2, step='post')
    
    ax.legend(loc='upper left')
    ax.set_xlim(left=0)
    
    if metric == "penalties":
        ax.set_ylim(bottom=0)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved average {metric} plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax

def compute_relative_differences(
    runs: Dict[str, List[RunResult]],
    metrics: List[str] = ["rewards", "epis", "penalties"]
) -> Dict[str, Dict[str, List[float]]]:
    """
    Compute relative differences between RL-MPC and baseline methods.
    
    For each matching (year, start_day) combination, computes:
        relative_diff = (rlmpc_value - baseline_value) / |baseline_value| * 100
    
    Args:
        runs: Dictionary mapping controller type to list of RunResult objects.
        metrics: List of metrics to compute differences for.
    
    Returns:
        Dictionary with structure:
        {
            "rl": {"rewards": [...], "epis": [...], "penalties": [...]},
            "mpc": {"rewards": [...], "epis": [...], "penalties": [...]}
        }
        Each list contains the relative differences for all matching environments.
    """
    # Index RL-MPC results by (year, start_day)
    rlmpc_by_key = {}
    for run in runs.get("rlmpc", []):
        key = (run.growth_year, run.start_day)
        rlmpc_by_key[key] = {
            "rewards": run.total_reward,
            "epis": run.total_epi,
            "penalties": run.total_penalty
        }
    
    differences = {
        "rl": {m: [] for m in metrics},
        "mpc": {m: [] for m in metrics}
    }
    differences_rel = {
        "rl": {m: [] for m in metrics},
        "mpc": {m: [] for m in metrics}
    }
    
    # Compute differences for each baseline controller
    for controller_type in ["rl", "mpc"]:
        for run in runs.get(controller_type, []):
            key = (run.growth_year, run.start_day)
            
            if key not in rlmpc_by_key:
                continue
            
            rlmpc_val = rlmpc_by_key[key]
            
            for metric in metrics:
                if metric == "rewards":
                    baseline_val = run.total_reward
                    rlmpc_metric = rlmpc_val["rewards"]
                elif metric == "epis":
                    baseline_val = run.total_epi
                    rlmpc_metric = rlmpc_val["epis"]
                else:
                    baseline_val = run.total_penalty
                    rlmpc_metric = rlmpc_val["penalties"]
                
                # Compute relative difference as percentage
                # if abs(baseline_val) > 1e-10:
                diff = (rlmpc_metric - baseline_val)
                rel_diff = 100 * diff / abs(baseline_val)
                # else:
                    # rel_diff = 0.0
                
                differences[controller_type][metric].append(diff)
                differences_rel[controller_type][metric].append(rel_diff)
    
    return differences, differences_rel

def compute_statistics(
    differences: Dict[str, Dict[str, List[float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute mean and 95% confidence interval for each comparison.
    
    Returns:
        Dictionary with structure:
        {
            "rl": {
                "rewards": {"mean": ..., "ci_lower": ..., "ci_upper": ..., "std": ..., "n": ...},
                ...
            },
            "mpc": {...}
        }
    """
    statistics = {}
    
    for controller_type, metric_diffs in differences.items():
        statistics[controller_type] = {}

        for metric, values in metric_diffs.items():
            if not values:
                statistics[controller_type][metric] = {
                    "mean": 0.0,
                    "ci_lower": 0.0,
                    "ci_upper": 0.0,
                    "std": 0.0,
                    "n": 0
                }
                continue
            
            values = np.array(values)
            n = len(values)
            mean = np.mean(values)
            std = np.std(values, ddof=1) if n > 1 else 0.0
            
            # 95% confidence interval
            if n > 1:
                sem = std / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=n-1)
                ci_half = t_crit * sem
            else:
                ci_half = 0.0
            
            statistics[controller_type][metric] = {
                "mean": mean,
                "ci_lower": mean - ci_half,
                "ci_upper": mean + ci_half,
                "std": std,
                "n": n
            }

    return statistics


def plot_relative_performance_bars(rlmpc_data, baseline_data, train_years, test_years,
                                    baseline_name="RL", metrics=None, metric_labels=None,
                                    output_path=None, show=False, color=None):
    """
    Create a grouped bar plot showing the relative performance of RL-MPC against a baseline method.
    
    Parameters
    ----------
    rlmpc_data : dict
        Dictionary with structure {year: {metric: array}} for RL-MPC results
    baseline_data : dict
        Dictionary with structure {year: {metric: array}} for baseline (RL or MPC) results
    train_years : list
        List of years used for training environment evaluation
    test_years : list
        List of years used for test environment evaluation
    baseline_name : str
        Name of the baseline method for labeling (default: "RL")
    metrics : list, optional
        List of metrics to plot (default: ["rewards", "EPI", "penalties"])
    metric_labels : dict, optional
        Dictionary mapping metric names to display labels
    output_path : str, optional
        Path to save the figure
    show : bool
        If True, display the plot (default: False)
    
    Returns
    -------
    fig, ax : matplotlib figure and axis objects
    """
    if metrics is None:
        metrics = ["rewards", "EPI", "penalties"]
    
    if metric_labels is None:
        metric_labels = {
            "rewards": r"$\mathcal{J}$",
            "EPI": "EPI",
            "penalties": "Penalty"
        }

    def compute_mean_final_value(data, years, metric):
        """Compute mean of final cumulative values across years."""
        final_values = []
        for year in years:
            if year in data and metric in data[year]:
                cumulative_final = np.sum(data[year][metric])
                final_values.append(cumulative_final)
        return np.mean(final_values) if final_values else 0

    def compute_std_final_value(data, years, metric):
        """Compute std of final cumulative values across years."""
        final_values = []
        for year in years:
            if year in data and metric in data[year]:
                cumulative_final = np.std(data[year][metric])
                final_values.append(cumulative_final)
        return np.std(final_values) if final_values else 0

    def compute_relative_improvement(rlmpc_val, baseline_val, metric):
        """
        Compute relative improvement of RL-MPC over baseline.
        For rewards: higher is better, so improvement = (rlmpc - baseline) / |baseline|
        For EPI and penalties: lower is better, so improvement = (baseline - rlmpc) / |baseline|
        Returns percentage improvement.
        """
        if baseline_val == 0:
            return 0
        if metric == "rewards":
            # Higher reward is better
            return 100 * (rlmpc_val - baseline_val) / abs(baseline_val)
        elif metric == "EPI":
            return 100 * (rlmpc_val - baseline_val) / abs(baseline_val)
        else:
            # Lower penalty is better
            return 100 * (baseline_val - rlmpc_val) / abs(baseline_val)
    
    # Compute improvements for each metric and environment
    train_improvements = []
    test_improvements = []
    
    for metric in metrics:
        # Training environment
        rlmpc_train = compute_mean_final_value(rlmpc_data, train_years, metric)
        baseline_train = compute_mean_final_value(baseline_data, train_years, metric)
        train_improvements.append(compute_relative_improvement(rlmpc_train, baseline_train, metric))
        
        # Test environment
        rlmpc_test = compute_mean_final_value(rlmpc_data, test_years, metric)
        baseline_test = compute_mean_final_value(baseline_data, test_years, metric)
        test_improvements.append(compute_relative_improvement(rlmpc_test, baseline_test, metric))
    
    # Set up figure
    WIDTH = 173.8/2 * 0.03937
    HEIGHT = WIDTH * 0.75
    fig, ax = plt.subplots(1, 1, figsize=(WIDTH, HEIGHT), dpi=300)
    
    # Bar plot setup
    x = np.arange(len(metrics))
    bar_width = 0.35
    
    # Colors for train/test
    # train_color, test_color = plt.cm.plasma(np.linspace(0.2, 0.8, 2))
    # Create bars
    bars_train = ax.bar(x - bar_width/2, train_improvements, bar_width, 
                        label="Train", color=color, edgecolor="white", linewidth=0.5)
    # Note: edgecolor controls hatch color in SVG - use white for visibility on colored bars
    bars_test = ax.bar(x + bar_width/2, test_improvements, bar_width, 
                       label="Test", facecolor=color, edgecolor="white", linewidth=0.5, hatch="//")
    
    # Add value labels on bars
    def add_bar_labels(bars, values):
        for bar, val in zip(bars, values):
            height = bar.get_height()
            va = 'bottom' if height >= 0 else 'top'
            offset = 0.5 if height >= 0 else -0.5
            ax.annotate(f'{val:.1f}%',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, offset),
                       textcoords="offset points",
                       ha='center', va=va, fontsize=5)
    
    add_bar_labels(bars_train, train_improvements)
    add_bar_labels(bars_test, test_improvements)

    # Formatting
    ax.set_ylabel(f"$\Delta$ (%)")
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in metrics])
    ax.legend(loc='best')
    ax.set_yscale('symlog')
    # ax.set_yticks(np.linspace(-100, 1000, 5))
    # ax.set_ylim(-300, 7000)
    # Add horizontal line at y=0
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    fig.tight_layout()
    
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved relative performance bar plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def plot_relative_performance_bars_grid(train_statistics, test_statistics,
                                        metrics=["rewards", "epis", "penalties"],
                                        output_path=None, show=False, colors=None):
    """
    Create a 1x2 subplot showing relative performance of RL-MPC vs RL (left) and RL-MPC vs MPC (right).
    
    Parameters
    ----------
    train_statistics : dict
        Dictionary with structure {controller_type: {metric: {mean: float, ci_lower: float, ci_upper: float, std: float, n: int}}}
    test_statistics : dict
        Dictionary with structure {controller_type: {metric: {mean: float, ci_lower: float, ci_upper: float, std: float, n: int}}}
    metrics : list, optional
        List of metrics to plot (default: ["rewards", "EPI", "penalties"])
    output_path : str, optional
        Path to save the figure
    show : bool
        If True, display the plot (default: False)
    
    Returns
    -------
    fig, axes : matplotlib figure and axis objects
    """
    
    metric_labels = {
        "rewards": r"$\mathcal{J}$",
        "epis": "EPI",
        "penalties": "Penalty"
    }


    rl_train_imp = []
    rl_test_imp = []
    rl_train_errors = []
    rl_test_errors = []
    mpc_train_imp = []
    mpc_test_imp = []
    mpc_train_errors = []
    mpc_test_errors = []
    for metric in metrics:
        rl_train_imp.append(train_statistics["rl"][metric]["mean"])
        rl_test_imp.append(test_statistics["rl"][metric]["mean"])
        rl_train_errors.append(train_statistics["rl"][metric]["mean"] - train_statistics["rl"][metric]["ci_lower"])
        rl_test_errors.append(test_statistics["rl"][metric]["mean"] - test_statistics["rl"][metric]["ci_lower"])

        mpc_train_imp.append(train_statistics["mpc"][metric]["mean"])
        mpc_test_imp.append(test_statistics["mpc"][metric]["mean"])
        mpc_train_errors.append(train_statistics["mpc"][metric]["mean"] - train_statistics["mpc"][metric]["ci_lower"])
        mpc_test_errors.append(test_statistics["mpc"][metric]["mean"] - test_statistics["mpc"][metric]["ci_lower"])

    # Set up figure with 1 row x 2 columns
    WIDTH = 173.8 / 2 * 0.03937
    HEIGHT = WIDTH * 0.75
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, HEIGHT), dpi=300, sharey=True)
    
    # Colors for train/test
    # train_color, test_color = plt.cm.plasma(np.linspace(0.2, 0.8, 2))
    
    # Data for each column
    column_data = [
        (rl_train_imp, rl_test_imp, rl_train_errors, rl_test_errors, "RL-MPC vs RL"),
        (mpc_train_imp, mpc_test_imp, mpc_train_errors, mpc_test_errors, "RL-MPC vs MPC"),
    ]
    
    # Bar plot setup
    x = np.arange(len(metrics))
    bar_width = 0.35

    for col, (train_imp, test_imp, train_errors, test_errors, title) in enumerate(column_data):
        ax = axes[col]
        
        # Create bars
        bars_train = ax.bar(x - bar_width/2, train_imp, bar_width, 
                            label="Train" if col == 0 else None, 
                            color=colors[col], edgecolor="white", linewidth=0.5)
        # Note: edgecolor controls hatch color in SVG - use white for visibility on colored bars
        bars_test = ax.bar(x + bar_width/2, test_imp, bar_width, 
                           label="Test" if col == 0 else None, 
                           facecolor=colors[col], edgecolor="white", linewidth=0.5, hatch="//")
        
        # ax.errorbar(
        #     x-bar_width/2,
        #     train_imp,
        #     yerr=[train_errors, train_errors],
        #     fmt="none",
        #     color=colors[col],
        #     capsize=3,
        #     capthick=1,
        #     linewidth=0.5,
        #     alpha=0.7
        # )
        ax.errorbar(
            x+bar_width/2,
            test_imp,
            yerr=[test_errors, test_errors],
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
            linewidth=1,
            alpha=0.7
        )

        # Add value labels on bars
        def add_bar_labels(bars, values):
            for bar, val in zip(bars, values):
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                offset = 0.5 if height >= 0 else -0.5
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, offset),
                           textcoords="offset points",
                           ha='center', va=va, fontsize=8)
                bar.set_clip_on(False)

        
        add_bar_labels(bars_train, train_imp)
        add_bar_labels(bars_test, test_imp)
        
        # Formatting
        ax.set_xticks(x)
        ax.set_xticklabels([metric_labels[m] for m in metrics], rotation=25, ha='center')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        # ax.set_title(title)
        
        # Y-label only on left column
        if col == 0:
            ax.set_ylabel(r"$\Delta$ Performance metric")
        
        # ax.set_yscale('symlog')
        # ax.set_yticks([-100, -10, 10, 100, 1000])
        print(ax.get_ylim()[0], ax.get_ylim()[1])
    axes[0].set_ylim(bottom=axes[0].get_ylim()[0]*1.05)
    axes[1].set_ylim(bottom=axes[1].get_ylim()[0]*1.05)
    
    # Add legend to first subplot
    legend = axes[0].legend(loc='best')
    for handle in legend.legend_handles:
        if hasattr(handle, "set_facecolor"):
            handle.set_facecolor('gray')
    
    fig.tight_layout()
    if output_path:
        fig.savefig(output_path, bbox_inches='tight')
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        fig.savefig(output_path.replace(".png", ".eps"), format="eps", dpi=300)
        fig.savefig(output_path.replace(".png", ".pdf"), format="pdf", dpi=300)
        print(f"Saved relative performance bar grid plot to {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, axes


def print_summary(
    runs: Dict[str, List[RunResult]],
    statistics: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str]
) -> None:
    """Print a summary of the relative differences."""
    print("\n" + "=" * 90)
    print("Relative Performance Difference Summary")
    print("=" * 90)
    
    # Count runs per controller
    for controller_type, run_list in runs.items():
        print(f"  {controller_type.upper()}: {len(run_list)} runs loaded")
    
    print("\n" + "-" * 90)
    print(f"{'Comparison':<25} {'Metric':<12} {'Mean':<12} {'95% CI':<20} {'N':<6}")
    print("-" * 90)
    
    metric_labels = {
        "rewards": r"$\mathcal{J}$",
        "epis": "EPI",
        "penalties": "Penalty"
    }

    for controller_type in ["rl", "mpc"]:
        label = f"RL-MPC vs {controller_type.upper()}"
        for metric in metrics:
            stat = statistics[controller_type][metric]
            ci_str = f"[{stat['ci_lower']:+.2f}, {stat['ci_upper']:+.2f}]"
            print(f"  {label:<23} {metric_labels.get(metric, metric):<12} {stat['mean']:+.2f}      {ci_str:<20} {stat['n']:<6}")
    
    print("=" * 90)


if __name__ == "__main__":
    project = "GL-MPC-RL"

    model_name = "daily-glade-107"
    horizon = 1
    location = "Netherlands"
    growth_years = [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2023]
    start_days = [151]
    results_dir = Path("results") / project / "deterministic"
    mpc_folder = "mpc_linear_solver_ma57"
    rlmpc_folder = f"rlmpc-switch-{model_name}-0.05"

    # rl_mpc_data = load_rl_mpc_data(project, rl_mpc_folder, location, growth_years, start_day, horizon)
    # rl_data = load_rl_data(project, model_name, location, growth_years, start_day)
    # mpc_data = load_mpc_data(project, "mpc_linear_solver_ma57", location, growth_years, start_day, horizon)

    train_runs = load_runs(
        model_name=model_name,
        start_days=start_days,
        growth_years=[2023],
        results_dir=results_dir,
        mpc_folder=mpc_folder,
        rlmpc_folder=rlmpc_folder,
        location=location,
        horizon=horizon
    )
    test_runs = load_runs(
        model_name=model_name,
        start_days=start_days,
        growth_years=growth_years[:-1],
        results_dir=results_dir,
        mpc_folder=mpc_folder,
        rlmpc_folder=rlmpc_folder,
        location=location,
        horizon=horizon
    )
    train_differences, train_differences_rel = compute_relative_differences(train_runs)
    train_statistics = compute_statistics(train_differences)
    train_statistics_rel = compute_statistics(train_differences_rel)
    test_differences, test_differences_rel = compute_relative_differences(test_runs)
    test_statistics = compute_statistics(test_differences)
    test_statistics_rel = compute_statistics(test_differences_rel)

    plot_relative_performance_bars_grid(
        train_statistics=train_statistics,
        test_statistics=test_statistics,
        output_path="figures/GL-MPC-RL/performance_bars_grid.png",
        colors=["#1f77b4", "#00a693"]
    )

    print_summary(
        runs=train_runs,
        statistics=train_statistics,
        metrics=["rewards", "epis", "penalties"]
    )
    print_summary(
        runs=test_runs,
        statistics=test_statistics,
        metrics=["rewards", "epis", "penalties"]
    )

    print_summary(
        runs=train_runs,
        statistics=train_statistics_rel,
        metrics=["rewards", "epis", "penalties"]
    )
    print_summary(
        runs=test_runs,
        statistics=test_statistics_rel,
        metrics=["rewards", "epis", "penalties"]
    )