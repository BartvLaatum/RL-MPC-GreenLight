"""
Visualize average cumulative penalty per episode broken down by controller
type and violation component (temperature, CO2, RH).

For RL controllers, penalties are read directly from the PPO result CSVs.
For MPC and RL-MPC controllers, penalties are recomputed from state
trajectory CSV files using the constraint violation logic and penalty weights.

Usage:
    python visualizations/average_penalties_grouped_bars.py \
        --model_name legendary-salad-82 \
        --start_days 151

    python visualizations/average_penalties_grouped_bars.py \
        --model_name legendary-salad-82 \
        --start_days 90 105 120 135 151 \
        --test_years 2011 2012 2019 2020

"""
import argparse
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

PENALTY_COMPONENTS = ["temp_violation", "co2_violation", "rh_violation"]

PENALTY_LABELS = {
    "temp_violation": "Temperature",
    "co2_violation": r"CO$_2$",
    "rh_violation": "RH",
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

# State CSV column indices for CO2, temperature, RH
STATE_COL_CO2 = 0
STATE_COL_TEMP = 2
STATE_COL_RH = 15

# Constraint bounds: [CO2, temp, RH]
Y_MIN = np.array([400.0, 15.0, 0.0])
Y_MAX = np.array([1600.0, 25.0, 90.0])

# Penalty weights: [CO2, temp, RH]
PENALTY_WEIGHTS = np.array([5e-5, 5e-3, 7e-4])


# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class PenaltyResult:
    """Container for cumulative penalty components of a single episode."""
    controller_type: str
    growth_year: int
    start_day: int
    temp_violation: float
    co2_violation: float
    rh_violation: float


# ==============================================================================
# Penalty Computation Helpers
# ==============================================================================

def compute_penalties_from_states(states_file: Path) -> Tuple[float, float, float]:
    """
    Compute cumulative weighted penalty for each violation type from a state
    trajectory file.

    State columns used: CO2 (col 0), temp (col 2), RH (col 15).
    Bounds: CO2 in [400, 1600], temp in [15, 25], RH in [0, 90].
    Penalty weights: CO2 = 5e-5, temp = 5e-3, RH = 7e-4.

    Returns:
        (temp_violation, co2_violation, rh_violation) — cumulative weighted
        penalties summed over all timesteps.
    """
    states = np.loadtxt(states_file, delimiter=",")

    co2 = states[:, STATE_COL_CO2]
    temp = states[:, STATE_COL_TEMP]
    rh = states[:, STATE_COL_RH]

    y = np.column_stack([co2, temp, rh])  # shape (T, 3)

    lb_violation = Y_MIN - y
    lb_violation[lb_violation < 0] = 0.0

    ub_violation = y - Y_MAX
    ub_violation[ub_violation < 0] = 0.0

    total_violation = lb_violation + ub_violation  # shape (T, 3)
    weighted = total_violation * PENALTY_WEIGHTS    # broadcast (T, 3) * (3,)
    cumulative = weighted.sum(axis=0)              # shape (3,)

    # cumulative order is [CO2, temp, RH]
    co2_pen, temp_pen, rh_pen = cumulative[0], cumulative[1], cumulative[2]

    return temp_pen, co2_pen, rh_pen


# ==============================================================================
# Results Loading
# ==============================================================================

def load_rl_penalties(
    results_dir: Path,
    model_name: str,
    location: str,
    growth_year: int,
    start_day: int,
) -> Tuple[float, float, float]:
    """Load penalty components from an RL result CSV file."""
    rl_file = results_dir / "ppo" / f"{model_name}-{location}-{growth_year}-{start_day}.csv"
    df = pd.read_csv(rl_file)

    temp_pen = df["temp_violation"].sum()
    co2_pen = df["co2_violation"].sum()
    rh_pen = df["rh_violation"].sum()

    return temp_pen, co2_pen, rh_pen


def load_mpc_penalties(
    results_dir: Path,
    mpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1,
) -> Tuple[float, float, float]:
    """Compute penalty components from MPC state trajectory files."""
    mpc_dir = results_dir / "mpc" / mpc_folder
    states_file = mpc_dir / f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    return compute_penalties_from_states(states_file)


def load_rlmpc_penalties(
    results_dir: Path,
    rlmpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1,
) -> Tuple[float, float, float]:
    """Compute penalty components from RL-MPC state trajectory files."""
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    states_file = rlmpc_dir / f"states-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    return compute_penalties_from_states(states_file)


def load_all_penalties(
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    results_dir: Path,
    mpc_folder: str = "mpc_linear_solver_ma57",
    rlmpc_folder: Optional[str] = None,
    location: str = LOCATION,
    horizon: int = 1,
) -> Dict[str, List[PenaltyResult]]:
    """
    Load cumulative penalty components for all controller types and episodes.

    Returns:
        Dictionary mapping controller type to list of PenaltyResult objects.
    """
    if rlmpc_folder is None:
        rlmpc_folder = f"rlmpc-switch-{model_name}-0.05"

    results: Dict[str, List[PenaltyResult]] = {
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
                    temp_pen, co2_pen, rh_pen = load_rl_penalties(
                        results_dir, model_name, location, year, start_day
                    )
                    results["rl"].append(PenaltyResult(
                        controller_type="rl",
                        growth_year=year,
                        start_day=start_day,
                        temp_violation=temp_pen*PENALTY_WEIGHTS[1],
                        co2_violation=co2_pen*PENALTY_WEIGHTS[0],
                        rh_violation=rh_pen*PENALTY_WEIGHTS[2],
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL penalties for year {year}, day {start_day}: {e}")

            # MPC
            mpc_dir = results_dir / "mpc" / mpc_folder
            states_file = mpc_dir / f"states-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            if states_file.exists():
                try:
                    temp_pen, co2_pen, rh_pen = load_mpc_penalties(
                        results_dir, mpc_folder, location, year, start_day, horizon
                    )
                    results["mpc"].append(PenaltyResult(
                        controller_type="mpc",
                        growth_year=year,
                        start_day=start_day,
                        temp_violation=temp_pen,
                        co2_violation=co2_pen,
                        rh_violation=rh_pen,
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading MPC penalties for year {year}, day {start_day}: {e}")

            # RL-MPC
            rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
            states_file = rlmpc_dir / f"states-300dt-{horizon}H-{location}-{year}-{start_day}.csv"
            if states_file.exists():
                try:
                    temp_pen, co2_pen, rh_pen = load_rlmpc_penalties(
                        results_dir, rlmpc_folder, location, year, start_day, horizon
                    )
                    results["rlmpc"].append(PenaltyResult(
                        controller_type="rlmpc",
                        growth_year=year,
                        start_day=start_day,
                        temp_violation=temp_pen,
                        co2_violation=co2_pen,
                        rh_violation=rh_pen,
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL-MPC penalties for year {year}, day {start_day}: {e}")

    return results


# ==============================================================================
# Compute Statistics
# ==============================================================================

def compute_penalty_statistics(
    runs: Dict[str, List[PenaltyResult]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Compute mean and 95% confidence interval for each controller type and
    penalty component.

    Returns:
        {controller_type: {penalty_component: {"mean", "ci_lower", "ci_upper", "std", "n"}}}
    """
    statistics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for controller_type, run_list in runs.items():
        statistics[controller_type] = {}

        if not run_list:
            for comp in PENALTY_COMPONENTS:
                statistics[controller_type][comp] = {
                    "mean": np.nan, "ci_lower": np.nan, "ci_upper": np.nan,
                    "std": 0.0, "n": 0,
                }
            continue

        values = {comp: [] for comp in PENALTY_COMPONENTS}
        for run in run_list:
            values["temp_violation"].append(run.temp_violation)
            values["co2_violation"].append(run.co2_violation)
            values["rh_violation"].append(run.rh_violation)

        for comp in PENALTY_COMPONENTS:
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
    penalty_statistics: Dict[str, Dict[str, Dict[str, float]]],
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    output_path: Path,
    show: bool = False,
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a grouped bar chart of average cumulative penalties with 95% CIs.

    Args:
        penalty_statistics: {controller_type: {penalty_component: {"mean", "ci_lower", "ci_upper", ...}}}
        model_name: Model name for the title.
        start_days: Start days used (for the title).
        growth_years: Growth years used (for the title).
        output_path: Path to save the figure.
        show: Whether to display the plot interactively.

    Returns:
        Tuple of (figure, axes).
    """
    SUBPLOT_KW = dict[str, float](left=0.2, right=0.97, top=0.97, bottom=0.2)
    active_controllers = [
        ct for ct in ["rl", "mpc", "rlmpc"]
        if ct in penalty_statistics and not all(
            np.isnan(penalty_statistics[ct][comp]["mean"]) for comp in PENALTY_COMPONENTS
        )
    ]

    if not active_controllers:
        warnings.warn("No controller data available for plotting.")
        return None, None

    n_components = len(PENALTY_COMPONENTS)
    n_controllers = len(active_controllers)

    # Figure setup
    WIDTH = 120/3 * 0.03937  # Half journal column width in inches
    HEIGHT = WIDTH * 1.0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    x = np.arange(n_components)
    bar_width = 0.8 / n_controllers

    for i, ct in enumerate(active_controllers):
        offset = (i - (n_controllers - 1) / 2) * bar_width
        positions = x + offset

        means = [penalty_statistics[ct][comp]["mean"] for comp in PENALTY_COMPONENTS]
        ci_lower = [
            penalty_statistics[ct][comp]["mean"] - penalty_statistics[ct][comp]["ci_lower"]
            for comp in PENALTY_COMPONENTS
        ]
        ci_upper = [
            penalty_statistics[ct][comp]["ci_upper"] - penalty_statistics[ct][comp]["mean"]
            for comp in PENALTY_COMPONENTS
        ]

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

    # Horizontal reference line at y = 0
    ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, zorder=1)

    # Axis labels
    # ax.set_xlabel("Violation Type")
    ax.set_ylabel("Cumulative $g_{i}$")
    ax.set_xticks(x)
    ax.set_xticklabels([PENALTY_LABELS[c] for c in PENALTY_COMPONENTS])
    # ax.set_yticks(np.linspace(0, 0.2, 3))
    # ax.set_ylim(bottom=0)
    # Legend
    # ax.legend(loc="best", framealpha=0.9)

    # Grid (horizontal only)
    # ax.yaxis.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

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


# ==============================================================================
# Summary Printing
# ==============================================================================

def print_summary(
    runs: Dict[str, List[PenaltyResult]],
    penalty_statistics: Dict[str, Dict[str, Dict[str, float]]],
) -> None:
    """Print a summary of loaded data, computed means and 95% CIs."""
    print("\n" + "=" * 100)
    print("Average Cumulative Penalty per Episode — Summary")
    print("=" * 100)

    for ct in ["rl", "mpc", "rlmpc"]:
        n = len(runs.get(ct, []))
        print(f"  {CONTROLLER_LABELS.get(ct, ct):>8s}: {n} episodes loaded")

    print("\n" + "-" * 100)
    print(f"{'Controller':<12} {'Violation':<14} {'Mean':>12} {'95% CI':>28} {'N':>6}")
    print("-" * 100)

    for ct in ["rl", "mpc", "rlmpc"]:
        if ct not in penalty_statistics:
            continue
        for comp in PENALTY_COMPONENTS:
            s = penalty_statistics[ct][comp]
            if np.isnan(s["mean"]):
                print(f"  {CONTROLLER_LABELS[ct]:<10} {PENALTY_LABELS[comp]:<14} {'N/A':>12}")
            else:
                ci_str = f"[{s['ci_lower']:+.6f}, {s['ci_upper']:+.6f}]"
                print(
                    f"  {CONTROLLER_LABELS[ct]:<10} {PENALTY_LABELS[comp]:<14} "
                    f"{s['mean']:>12.6f} {ci_str:>28} {int(s['n']):>6}"
                )

    print("=" * 100)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function to create the average penalties grouped bar chart."""
    parser = argparse.ArgumentParser(
        description="Create grouped bar chart of average cumulative penalties per episode."
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
        default="outputs/figures/average_penalties_bars",
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

    # Load penalty data
    print(f"Loading results from {args.results_dir}...")
    print(f"  Model name : {args.model_name}")
    print(f"  Start days : {args.start_days}")
    print(f"  Test years : {args.test_years}")
    results_dir = Path(args.results_dir)

    runs = load_all_penalties(
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
    penalty_statistics = compute_penalty_statistics(runs)

    # Print summary
    print_summary(runs, penalty_statistics)

    # Create plot
    output_path = Path(args.out_dir) / f"{args.model_name}_average_penalties_bars"

    print(f"\nGenerating plot...")
    make_plot(
        penalty_statistics=penalty_statistics,
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
