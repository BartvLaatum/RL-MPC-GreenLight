"""
Visualize average cumulative cost per episode broken down by controller type
and cost component.

This script creates a grouped bar chart where each cost component (Revenue,
Heat, CO2, Electricity) has a cluster of bars (one per controller type: RL,
MPC, RL-MPC), showing the mean cumulative cost across episodes.

Usage:
    python visualizations/average_costs_grouped_bars.py \
        --model_name legendary-salad-82 \
        --start_days 151

    python visualizations/average_costs_grouped_bars.py \
        --model_name legendary-salad-82 \
        --start_days 90 105 120 135 151 \
        --test_years 2011 2012 2019 2020
"""
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

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

COST_COMPONENTS = ["revenue", "heat_costs", "co2_costs", "electricity_costs"]

COST_LABELS = {
    "revenue": "Revenue",
    "heat_costs": "Heat",
    "co2_costs": r"CO$_2$",
    "electricity_costs": "Electricity",
}

CONTROLLER_LABELS = {
    "rl": "RL",
    "mpc": "MPC",
    "rlmpc": "RL-MPC",
}

COLORS = {
    "rl": "#1f77b4",       # Blue
    "mpc": "#00a693",      # Teal
    "rlmpc": "#d62728",    # Red
}


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class CostResult:
    """Container for cumulative cost components of a single episode."""
    controller_type: str
    growth_year: int
    start_day: int
    revenue: float
    heat_costs: float
    co2_costs: float
    electricity_costs: float

# ==============================================================================
# Cost Computation Helpers
# ==============================================================================

def compute_costs_from_control_and_states(
    control_inputs_file: Path,
    states_file: Path,
) -> Tuple[float, float, float, float]:
    """
    Compute cumulative cost components from control input and state
    trajectory files (used for MPC and RL-MPC controllers).

    Control input columns: heating, CO2, ThScr, Ventilation, Lamp, BlScr
    State column 26 (0-indexed): fruit weight

    Returns:
        (revenue, heat_costs, co2_costs, electricity_costs)
    """
    controls = np.loadtxt(control_inputs_file, delimiter=",")
    states = np.loadtxt(states_file, delimiter=",")

    u_heat = controls[:, 0]
    u_co2 = controls[:, 1]
    u_lamp = controls[:, 4]

    heat_costs = np.sum(0.09 * 44 * 1e-3 * u_heat / 12)
    co2_costs = np.sum(0.3 * u_co2 * 5 * 1e-6 * 300)
    electricity_costs = np.sum(0.2 * 116 * 1e-3 * u_lamp / 12)

    x_fruit = states[:, 25]
    revenue = (x_fruit[-1] - x_fruit[0]) * 1e-6 / 0.06 * 1.2

    return revenue, heat_costs, co2_costs, electricity_costs


# ==============================================================================
# Results Loading
# ==============================================================================

def load_rl_costs(
    results_dir: Path,
    model_name: str,
    location: str,
    growth_year: int,
    start_day: int,
) -> Tuple[float, float, float, float]:
    """Load cost components from an RL result file."""
    rl_file = results_dir / "ppo" / f"{model_name}-{location}-{growth_year}-{start_day}.csv"
    df = pd.read_csv(rl_file)

    revenue = df["Revenue"].sum()
    heat_costs = df["Heat costs"].sum()
    co2_costs = df["CO2 costs"].sum()
    electricity_costs = df["Elec costs"].sum()

    return revenue, heat_costs, co2_costs, electricity_costs


def load_mpc_costs(
    results_dir: Path,
    mpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1,
) -> Tuple[float, float, float, float]:
    """Load cost components from MPC control-input and state files."""
    mpc_dir = results_dir / "mpc" / mpc_folder
    ctrl_file = mpc_dir / f"control-inputs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    states_file = mpc_dir / f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    return compute_costs_from_control_and_states(ctrl_file, states_file)


def load_rlmpc_costs(
    results_dir: Path,
    rlmpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1,
) -> Tuple[float, float, float, float]:
    """Load cost components from RL-MPC control-input and state files."""
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    ctrl_file = rlmpc_dir / f"control-inputs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    states_file = rlmpc_dir / f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    return compute_costs_from_control_and_states(ctrl_file, states_file)


def load_all_costs(
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    results_dir: Path,
    mpc_folder: str = "mpc_linear_solver_ma57",
    rlmpc_folder: Optional[str] = None,
    location: str = LOCATION,
    horizon: int = 1,
) -> Dict[str, List[CostResult]]:
    """
    Load cumulative cost components for all controller types and episodes.

    Returns:
        Dictionary mapping controller type to list of CostResult objects.
    """
    if rlmpc_folder is None:
        rlmpc_folder = f"rlmpc-switch-{model_name}-0.05"

    results: Dict[str, List[CostResult]] = {
        "rl": [],
        "mpc": [],
        "rlmpc": [],
    }

    for start_day in start_days:
        for year in growth_years:
            # RL
            rl_file = results_dir / "ppo" / f"{model_name}-{location}-{year}-{start_day}.csv"
            if rl_file.exists():
                try:
                    rev, heat, co2, elec = load_rl_costs(
                        results_dir, model_name, location, year, start_day
                    )
                    results["rl"].append(CostResult(
                        controller_type="rl",
                        growth_year=year,
                        start_day=start_day,
                        revenue=rev,
                        heat_costs=heat,
                        co2_costs=co2,
                        electricity_costs=elec,
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL costs for year {year}, day {start_day}: {e}")

            # MPC
            mpc_dir = results_dir / "mpc" / mpc_folder
            ctrl_file = mpc_dir / f"control-inputs-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            states_file = mpc_dir / f"states-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            if ctrl_file.exists() and states_file.exists():
                try:
                    rev, heat, co2, elec = load_mpc_costs(
                        results_dir, mpc_folder, location, year, start_day, horizon
                    )
                    results["mpc"].append(CostResult(
                        controller_type="mpc",
                        growth_year=year,
                        start_day=start_day,
                        revenue=rev,
                        heat_costs=heat,
                        co2_costs=co2,
                        electricity_costs=elec,
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading MPC costs for year {year}, day {start_day}: {e}")

            # RL-MPC
            rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
            ctrl_file = rlmpc_dir / f"control-inputs-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            states_file = rlmpc_dir / f"states-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            if ctrl_file.exists() and states_file.exists():
                try:
                    rev, heat, co2, elec = load_rlmpc_costs(
                        results_dir, rlmpc_folder, location, year, start_day, horizon
                    )
                    results["rlmpc"].append(CostResult(
                        controller_type="rlmpc",
                        growth_year=year,
                        start_day=start_day,
                        revenue=rev,
                        heat_costs=heat,
                        co2_costs=co2,
                        electricity_costs=elec,
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL-MPC costs for year {year}, day {start_day}: {e}")

    return results


# ==============================================================================
# Compute Mean Costs
# ==============================================================================

def compute_cost_statistics(
    runs: Dict[str, List[CostResult]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute mean and 95% confidence interval for each controller type and
    cost component.

    Returns:
        Dictionary:
        {controller_type: {cost_component: {"mean": ..., "ci_lower": ..., "ci_upper": ..., "std": ..., "n": ...}}}
    """
    statistics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for controller_type, run_list in runs.items():
        statistics[controller_type] = {}

        if not run_list:
            for comp in COST_COMPONENTS:
                statistics[controller_type][comp] = {
                    "mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                    "std": 0.0, "n": 0,
                }
            continue

        values = {comp: [] for comp in COST_COMPONENTS}
        for run in run_list:
            values["revenue"].append(run.revenue)
            values["heat_costs"].append(run.heat_costs)
            values["co2_costs"].append(run.co2_costs)
            values["electricity_costs"].append(run.electricity_costs)

        for comp in COST_COMPONENTS:
            arr = np.array(values[comp])
            n = len(arr)
            mean = float(np.nanmean(arr))
            std = float(np.std(arr, ddof=1)) if n > 1 else 0.0

            if n > 1:
                sem = std / np.sqrt(n)
                t_crit = stats.t.ppf(0.975, df=n - 1)
                ci_half = t_crit * sem
            else:
                ci_half = 0.0

            statistics[controller_type][comp] = {
                "mean": mean,
                "ci_lower": mean - ci_half,
                "ci_upper": mean + ci_half,
                "std": std,
                "n": n,
            }

    return statistics


# ==============================================================================
# Plotting
# ==============================================================================

def make_plot(
    cost_statistics: Dict[str, Dict[str, Dict[str, float]]],
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    output_path: Path,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar chart of average cumulative costs with 95% CIs.

    Args:
        cost_statistics: {controller_type: {cost_component: {"mean", "ci_lower", "ci_upper", ...}}}
        model_name: Model name for the title.
        start_days: Start days used (for the title).
        growth_years: Growth years used (for the title).
        output_path: Path to save the figure.
        show: Whether to display the plot interactively.

    Returns:
        Tuple of (figure, axes).
    """
    SUBPLOT_KW = dict[str, float](left=0.2, right=0.97, top=0.97, bottom=0.2)

    # Determine which controller types have data
    active_controllers = [
        ct for ct in ["rl", "mpc", "rlmpc"]
        if ct in cost_statistics and not all(
            np.isnan(cost_statistics[ct][comp]["mean"]) for comp in COST_COMPONENTS
        )
    ]

    if not active_controllers:
        warnings.warn("No controller data available for plotting.")
        return None, None

    n_components = len(COST_COMPONENTS)
    n_controllers = len(active_controllers)

    # Figure setup
    WIDTH = 120/3 * 0.03937  # Full journal column width in inches
    HEIGHT = WIDTH * 1
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    x = np.arange(n_components)
    bar_width = 0.8 / n_controllers

    for i, ct in enumerate(active_controllers):
        offset = (i - (n_controllers - 1) / 2) * bar_width
        positions = x + offset

        means = [cost_statistics[ct][comp]["mean"] for comp in COST_COMPONENTS]
        ci_lower = [cost_statistics[ct][comp]["mean"] - cost_statistics[ct][comp]["ci_lower"]
                     for comp in COST_COMPONENTS]
        ci_upper = [cost_statistics[ct][comp]["ci_upper"] - cost_statistics[ct][comp]["mean"]
                     for comp in COST_COMPONENTS]

        ax.bar(
            positions,
            means,
            bar_width,
            label=CONTROLLER_LABELS[ct],
            color=COLORS[ct],
            edgecolor="white",
            linewidth=0.5,
            alpha=0.85,
        )
        if ci_lower != ci_upper:
            ax.errorbar(
                positions,
                means,
                yerr=[ci_lower, ci_upper],
                fmt="none",
                color="black",
                capsize=3,
                capthick=1,
                linewidth=1,
                alpha=0.7,
            )
        # ax.errorbar(
        #     positions,
        #     means,
        #     yerr=[ci_lower, ci_upper],
        #     fmt="none",
        #     color="black",
        #     capsize=3,
        #     capthick=1,
        #     linewidth=1,
        #     alpha=0.7,
        # )

        # for j, (pos, mean_val) in enumerate(zip(positions, means)):
        #     if np.isnan(mean_val):
        #         continue
        #     va = "bottom" if mean_val >= 0 else "top"
        #     err = ci_upper[j] if mean_val >= 0 else ci_lower[j]
        #     y_anchor = mean_val + err if mean_val >= 0 else mean_val - err
        #     pt_offset = 0.5 if mean_val >= 0 else -0.5
        #     ax.annotate(
        #         f"{mean_val:.3f}",
        #         xy=(pos, y_anchor),
        #         xytext=(0, pt_offset),
        #         textcoords="offset points",
        #         ha="center",
        #         va=va,
        #         fontsize=7,
        #         color="black",
        #     )

    # Horizontal reference line at y = 0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, zorder=1)

    # Axis labels
    # ax.set_xlabel("Cost Component")
    ax.set_ylabel("Cost component (EUR/m$^2$)")
    ax.set_xticks(x)
    ax.set_xticklabels([COST_LABELS[c] for c in COST_COMPONENTS], rotation=0, ha="center")
    # Disable grid lines on the axes
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    # Legend
    ax.legend(loc="best", frameon=False, ncol=3)

    # Grid (horizontal only)
    ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # fig.tight_layout()
    fig.subplots_adjust(**SUBPLOT_KW)

    # Save figure
    output_path = Path(output_path) / "train"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    print(f"Saved figure to {output_path.with_suffix('.png')} and .svg")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


# ==============================================================================
# Summary Printing
# ==============================================================================

def print_summary(
    runs: Dict[str, List[CostResult]],
    cost_statistics: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Print a summary of loaded data, computed means and 95% CIs."""
    print("\n" + "=" * 100)
    print("Average Cumulative Cost per Episode — Summary")
    print("=" * 100)

    for ct in ["rl", "mpc", "rlmpc"]:
        n = len(runs.get(ct, []))
        print(f"  {CONTROLLER_LABELS.get(ct, ct):>8s}: {n} episodes loaded")

    print("\n" + "-" * 100)
    print(f"{'Controller':<12} {'Component':<14} {'Mean':>10} {'95% CI':>24} {'N':>6}")
    print("-" * 100)

    for ct in ["rl", "mpc", "rlmpc"]:
        if ct not in cost_statistics:
            continue
        for comp in COST_COMPONENTS:
            s = cost_statistics[ct][comp]
            if np.isnan(s["mean"]):
                print(f"  {CONTROLLER_LABELS[ct]:<10} {COST_LABELS[comp]:<14} {'N/A':>10}")
            else:
                ci_str = f"[{s['ci_lower']:+.4f}, {s['ci_upper']:+.4f}]"
                print(f"  {CONTROLLER_LABELS[ct]:<10} {COST_LABELS[comp]:<14} {s['mean']:>10.4f} {ci_str:>24} {int(s['n']):>6}")

    print("=" * 100)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function to create the average costs grouped bar chart."""
    parser = argparse.ArgumentParser(
        description="Create grouped bar chart of average cumulative costs per episode."
    )

    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the RL model (e.g., 'legendary-salad-82')",
    )
    parser.add_argument(
        "--start_days",
        type=int,
        nargs="+",
        required=True,
        help="Start day(s) for evaluation (e.g., 151 or 90 105 120 135 151)",
    )

    # Optional arguments
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/GL-MPC-RL/deterministic",
        help="Base directory for results",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/figures/average_costs_bars",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--mpc_folder",
        type=str,
        default="mpc_linear_solver_ma57",
        help="MPC results subfolder",
    )
    parser.add_argument(
        "--rlmpc_folder",
        type=str,
        default=None,
        help="RL-MPC results subfolder (default: rlmpc-switch-{model_name}-0.05)",
    )
    parser.add_argument(
        "--test_years",
        type=int,
        nargs="+",
        default=[2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        help="Years to evaluate on",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=LOCATION,
        help="Location for results",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plot interactively",
    )

    args = parser.parse_args()

    # Load cost data
    print(f"Loading results from {args.results_dir}...")
    print(f"  Model name : {args.model_name}")
    print(f"  Start days : {args.start_days}")
    print(f"  Test years : {args.test_years}")
    results_dir = Path(args.results_dir)

    runs = load_all_costs(
        model_name=args.model_name,
        start_days=args.start_days,
        growth_years=args.test_years,
        results_dir=results_dir,
        mpc_folder=args.mpc_folder,
        rlmpc_folder=args.rlmpc_folder,
        location=args.location,
    )

    # Check if we have data
    total_runs = sum(len(v) for v in runs.values())
    if total_runs == 0:
        print("\nERROR: No runs found. Check paths and arguments.")
        return 1

    # Compute statistics (means + 95% CIs)
    cost_statistics = compute_cost_statistics(runs)

    # Print summary
    print_summary(runs, cost_statistics)

    # Create plot
    output_path = Path(args.out_dir) / f"{args.model_name}_average_costs_bars"

    print(f"\nGenerating plot...")
    make_plot(
        cost_statistics=cost_statistics,
        model_name=args.model_name,
        start_days=args.start_days,
        growth_years=args.test_years,
        output_path=output_path,
        show=args.show,
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
