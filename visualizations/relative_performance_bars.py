"""
Visualize relative performance difference between RL-MPC and baseline methods.

This script creates a grouped bar plot showing the relative difference in
performance metrics between RL-MPC and standalone RL/MPC methods, aggregated
over multiple test environments (combinations of test_year and start_day).

Usage:
    python visualizations/relative_performance_bars.py \
        --model_name daily-glade-107 \
        --start_days 151

    python visualizations/relative_performance_bars.py \
        --model_name daily-glade-107 \
        --start_days 90 105 120 135 151 \
        --test_years 2011 2012 2019 2020 2023
"""

import argparse
import csv
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# Import plot configuration for consistent styling
try:
    from visualizations.plot_config import *
except ModuleNotFoundError:
    import plot_config


# ==============================================================================
# Constants
# ==============================================================================

LOCATION = "Netherlands"

# Colors for the two comparisons
COLORS = {
    "rl": "#1f77b4",      # Blue - RL-MPC vs RL
    "mpc": "#00a693"      # Teal - RL-MPC vs MPC
}

# Metric display labels
METRIC_LABELS = {
    "rewards": "$\mathcal{J}$",
    "epis": "$\mathcal{E}$",
    "penalties": "$\mathcal{P}$"
}


# ==============================================================================
# Data Classes
# ==============================================================================

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


# ==============================================================================
# Results Loading (simplified from generalist_performance.py)
# ==============================================================================

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
    location: str = LOCATION,
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


# ==============================================================================
# Compute Relative Differences
# ==============================================================================

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
        "mpc": {m: [] for m in metrics},
        "rl_rel": {m: [] for m in metrics},
        "mpc_rel": {m: [] for m in metrics}
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
                rel_diff = 100 * (rlmpc_metric - baseline_val) / abs(baseline_val)
                diff = (rlmpc_metric - baseline_val)
                # else:
                    # rel_diff = 0.0

                differences[controller_type][metric].append(diff)
                differences[controller_type+"_rel"][metric].append(rel_diff)

    return differences

def compute_averages(
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
  
    averages = {
        "rlmpc": {m: [] for m in metrics},
        "rl": {m: [] for m in metrics},
        "mpc": {m: [] for m in metrics},
    }

    # Compute differences for each baseline controller
    for controller_type in averages.keys():
        for metric in metrics:
            for run in runs.get(controller_type, []):
                if metric == "rewards":
                    averages[controller_type][metric].append(run.total_reward)
                elif metric == "epis":
                    averages[controller_type][metric].append(run.total_epi)
                else:
                    averages[controller_type][metric].append(run.total_penalty)

    return averages


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

# ==============================================================================
# Plotting
# ==============================================================================

def make_plot(
    statistics: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    output_path: Path,
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar plot showing relative performance differences.
    
    Args:
        statistics: Statistics from compute_statistics().
        metrics: List of metrics to plot.
        output_path: Path to save the figure.
        show: Whether to display the plot.
    
    Returns:
        Tuple of (figure, axes).
    """
    # Figure setup
    WIDTH = 120 / 3 * 0.03937  # Journal width
    HEIGHT = WIDTH * 1.0
    SUBPLOT_KW = dict[str, float](left=0.2, right=0.97, top=0.97, bottom=0.2)
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    # Bar positions
    x = np.arange(len(metrics))
    bar_width = 0.35
    
    # Plot bars for each comparison
    comparisons = [
        ("rl", r"RL-MPC vs RL", COLORS["rl"]),
        ("mpc", r"RL-MPC vs MPC", COLORS["mpc"])
    ]
    
    for i, (controller_type, label, color) in enumerate(comparisons):
        means = []
        errors_lower = []
        errors_upper = []
        
        for metric in metrics:
            stat = statistics[controller_type][metric]
            means.append(stat["mean"])
            errors_lower.append(stat["mean"] - stat["ci_lower"])
            errors_upper.append(stat["ci_upper"] - stat["mean"])
        
        # Position for this group
        offset = (i - 0.5) * bar_width
        positions = x + offset
        
        # Plot bars with error bars
        ax.bar(
            positions,
            means,
            bar_width,
            label=label,
            color=color,
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85
        )

        # Add error bars (95% CI)
        ax.errorbar(
            positions,
            means,
            yerr=[errors_lower, errors_upper],
            fmt="none",
            color="black",
            capsize=3,
            capthick=1,
            linewidth=1,
            alpha=0.7
        )

        # Annotate bars with mean values
        for i, (pos, mean_val) in enumerate(zip(positions, means)):
            # Position annotation above positive bars, below negative bars
            va = "bottom" if mean_val >= 0 else "top"
            error = errors_lower[i] if mean_val >= 0 else errors_upper[i]
            offset = 0.5 if mean_val >= 0 else -0.6
            ypos = mean_val+error if mean_val >= 0 else mean_val-error
            ax.annotate(
                f"{mean_val:.2f}",
                xy=(pos, ypos),
                xytext=(0, offset),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=7,
                color="black"
            )

    # Add horizontal line at y=0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, zorder=1)

    # Axis labels and formatting
    # ax.set_xlabel("Performance Metric")
    ax.set_ylabel("$\Delta$ Performance")
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS.get(m, m) for m in metrics])
    ax.set_ylim(bottom=ax.get_ylim()[0]*1.2)
    ax.set_yticks(np.linspace(0, 0.2, 3))

    # Legend
    ax.legend(loc="best", frameon=False)

    # Grid (horizontal only)
    # ax.yaxis.grid(False, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Tight layout
    # fig.tight_layout()
    fig.subplots_adjust(**SUBPLOT_KW)
    
    # Save figure
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved figure to {output_path.with_suffix('.png')} and .svg")
    
    if show:
        plt.show()
    else:
        plt.close(fig)
    
    return fig, ax


def print_summary_averages(
    runs: Dict[str, List[RunResult]],
    statistics: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str]
) -> None:
    """Print a summary of the averages."""
    print("\n" + "=" * 90)
    print("Average Performance Summary")
    print("=" * 90)
    
    # Count runs per controller
    for controller_type, run_list in runs.items():
        print(f"  {controller_type.upper()}: {len(run_list)} runs loaded")
    
    print("\n" + "-" * 90)
    print(f"{'Comparison':<25} {'Metric':<12} {'Mean':<12} {'95% CI':<20} {'N':<6}")
    print("-" * 90)

    for controller_type in ["rlmpc", "rl", "mpc"]:
        label = f"{controller_type.upper()}"
        for metric in metrics:
            stat = statistics[controller_type][metric]
            ci_str = f"[{stat['ci_lower']:+.3f}, {stat['ci_upper']:+.3f}]"
            print(f"  {label:<23} {METRIC_LABELS.get(metric, metric):<12} {stat['mean']:+.3f}      {ci_str:<20} {stat['n']:<6}")
    
    print("=" * 90)


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
    
    for controller_type in ["rl", "mpc"]:
        label = f"RL-MPC vs {controller_type.upper()}"
        for metric in metrics:
            stat = statistics[controller_type][metric]
            ci_str = f"[{stat['ci_lower']:+.3f}, {stat['ci_upper']:+.3f}]"
            print(f"  {label:<23} {METRIC_LABELS.get(metric, metric):<12} {stat['mean']:+.3f}      {ci_str:<20} {stat['n']:<6}")
    
    print("=" * 90)


def save_statistics(
    average_statistics: Dict[str, Dict[str, Dict[str, float]]],
    difference_statistics: Dict[str, Dict[str, Dict[str, float]]],
    metrics: List[str],
    output_path: Path,
) -> None:
    """Save the printed average and difference summaries as CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "summary",
        "controller",
        "metric",
        "mean",
        "ci_lower",
        "ci_upper",
        "std",
        "n",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for controller in ["rlmpc", "rl", "mpc"]:
            for metric in metrics:
                writer.writerow(
                    {
                        "summary": "average",
                        "controller": controller,
                        "metric": metric,
                        **average_statistics[controller][metric],
                    }
                )

        for controller in ["rl", "mpc"]:
            for metric in metrics:
                writer.writerow(
                    {
                        "summary": "difference",
                        "controller": f"rlmpc_vs_{controller}",
                        "metric": metric,
                        **difference_statistics[controller][metric],
                    }
                )

        for controller in ["rl", "mpc"]:
            for metric in metrics:
                writer.writerow(
                    {
                        "summary": "relative_difference_percent",
                        "controller": f"rlmpc_vs_{controller}",
                        "metric": metric,
                        **difference_statistics[f"{controller}_rel"][metric],
                    }
                )

    print(f"Saved statistics to {output_path}")


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function to create the relative performance bar plot."""
    parser = argparse.ArgumentParser(
        description="Create grouped bar plot of relative performance differences."
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name", 
        type=str, 
        required=True,
        help="Name of the RL model (e.g., 'daily-glade-107')"
    )
    parser.add_argument(
        "--start_days", 
        type=int,
        nargs="+",
        required=True,
        help="Start day(s) for evaluation (e.g., 151 or 90 105 120 135 151)"
    )
    
    # Optional arguments
    parser.add_argument(
        "--results_dir", 
        type=str, 
        default="results/GL-MPC-RL/deterministic",
        help="Base directory for results"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="outputs/figures",
        help="Output directory for figures"
    )
    parser.add_argument(
        "--mpc_folder", 
        type=str, 
        default="mpc_linear_solver_ma57",
        help="MPC results subfolder"
    )
    parser.add_argument(
        "--rlmpc_folder", 
        type=str, 
        default=None,
        help="RL-MPC results subfolder (default: rlmpc-switch-{model_name}-0.05)"
    )
    parser.add_argument(
        "--test_years", 
        type=int, 
        nargs="+",
        default=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        help="Years to evaluate on"
    )
    parser.add_argument(
        "--location", 
        type=str, 
        default=LOCATION,
        help="Location for results"
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["rewards", "epis", "penalties"],
        choices=["rewards", "epis", "penalties"],
        help="Metrics to include in the plot"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show plot interactively"
    )
    
    args = parser.parse_args()
    
    # Load runs
    print(f"Loading results from {args.results_dir}...")
    print(f"  Start days: {args.start_days}")
    print(f"  Test years: {args.test_years}")
    results_dir = Path(args.results_dir)
    
    runs = load_runs(
        model_name=args.model_name,
        start_days=args.start_days,
        growth_years=args.test_years,
        results_dir=results_dir,
        mpc_folder=args.mpc_folder,
        rlmpc_folder=args.rlmpc_folder,
        location=args.location
    )
    
    # Check if we have data
    total_runs = sum(len(v) for v in runs.values())
    if total_runs == 0:
        print("\nERROR: No runs found.")
        return 1
    
    if not runs.get("rlmpc"):
        print("\nERROR: No RL-MPC runs found. Cannot compute differences.")
        return 1
    
    averages = compute_averages(runs, args.metrics)
    stats = compute_statistics(averages)
    from pprint import pprint
    print("Statistics:")
    print_summary_averages(runs, stats, args.metrics)

    # Compute relative differences
    differences = compute_relative_differences(runs, args.metrics)
    
    # Compute statistics
    statistics = compute_statistics(differences)
    
    # Print summary
    print_summary(runs, statistics, args.metrics)
    # print_summary(runs, stats, args.metrics)
    
    # Create plot
    output_path = Path(args.out_dir) / "relative_performance_bars"
    save_statistics(
        average_statistics=stats,
        difference_statistics=statistics,
        metrics=args.metrics,
        output_path=output_path.with_suffix(".csv"),
    )
    
    print(f"\nGenerating plot...")
    make_plot(
        statistics=statistics,
        metrics=args.metrics,
        output_path=output_path,
        show=args.show
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
