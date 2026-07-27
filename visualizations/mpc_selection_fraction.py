"""
Visualize the fraction of times MPC was selected by RL-MPC vs weather deviation.

This script plots how the fraction of times the MPC input was selected by the
RL-MPC method changes as the test environment deviates from the training
environment, where deviation is measured in standard deviation units of a
weather variable (temperature, humidity, or radiation).

Usage:
    # Single start day
    python visualizations/mpc_selection_fraction.py \
        --model_name daily-glade-107 \
        --start_days 151

    # Multiple start days and years
    python visualizations/mpc_selection_fraction.py \
        --model_name daily-glade-107 \
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

# Import plot configuration for consistent styling
from plot_config import *


# ==============================================================================
# Constants
# ==============================================================================

WEATHER_DATA_DIR = Path("weather")
LOCATION = "Netherlands"
SECONDS_IN_DAY = 86400

# Default training configuration
DEFAULT_TRAINING_YEARS = list(range(2013, 2019))  # 2013-2018
DEFAULT_TRAINING_START_DAY = 90
DEFAULT_TRAINING_END_DAY = 152
DEFAULT_SEASON_LENGTH = 1  # days per episode

# Weather variable configuration
WEATHER_VARIABLES = {
    "temperature": {
        "column": "air temperature",
        "unit": "°C",
        "label": "Temperature",
        "axis_label": r"$\bar{d}_{\mathrm{T}}$  ($^\degree$C)"
    },
    "humidity": {
        "column": "RH",
        "unit": "%",
        "label": "Relative Humidity",
        "axis_label": r"$\bar{d}_{\mathrm{RH}}$ (%)"
    },
    "radiation": {
        "column": "global radiation",
        "unit": "W/m²",
        "label": "Solar Radiation",
        "axis_label": r"$\bar{d}_{\mathrm{iGlob}}$ (W/m$^2$)"
    }
}

# ==============================================================================
# Data Classes
# ==============================================================================

@dataclass
class RLMPCResult:
    """Container for a single RL-MPC experiment run result."""
    growth_year: int
    start_day: int
    location: str
    mpc_fraction: float  # Fraction of times MPC was selected
    n_timesteps: int  # Total number of timesteps
    test_mean_variable: float  # Mean value of weather variable


@dataclass
class TrainingStats:
    """Statistics for the training weather variable distribution."""
    mean: float
    std: float
    n_samples: int
    variable: str = "temperature"


# ==============================================================================
# Weather Data Loading
# ==============================================================================

def load_weather_data(year: int, location: str = LOCATION) -> pd.DataFrame:
    """Load raw weather data for a given year."""
    weather_path = WEATHER_DATA_DIR / location / f"{year}.csv"
    if not weather_path.exists():
        raise FileNotFoundError(f"Weather data not found: {weather_path}")
    return pd.read_csv(weather_path)


def compute_daily_mean_variable(
    df: pd.DataFrame, 
    start_day: int, 
    end_day: int,
    variable: str = "temperature"
) -> np.ndarray:
    """Compute daily mean values for a weather variable over a range of days."""
    if variable not in WEATHER_VARIABLES:
        raise ValueError(f"Unknown variable: {variable}")
    
    column_name = WEATHER_VARIABLES[variable]["column"]
    
    start_time = start_day * SECONDS_IN_DAY
    end_time = end_day * SECONDS_IN_DAY
    
    mask = (df["time"] >= start_time) & (df["time"] < end_time)
    filtered_df = df.loc[mask].copy()
    
    if filtered_df.empty:
        return np.array([])
    
    filtered_df["day"] = (filtered_df["time"] // SECONDS_IN_DAY).astype(int)
    daily_avg = filtered_df.groupby("day")[column_name].mean().values
    
    return daily_avg


def compute_training_stats(
    training_years: List[int] = DEFAULT_TRAINING_YEARS,
    start_day: int = DEFAULT_TRAINING_START_DAY,
    end_day: int = DEFAULT_TRAINING_END_DAY,
    location: str = LOCATION,
    variable: str = "temperature"
) -> TrainingStats:
    """Compute mean and std of a daily weather variable over the training set."""
    all_daily_values = []
    
    for year in training_years:
        try:
            df = load_weather_data(year, location)
            daily_values = compute_daily_mean_variable(df, start_day, end_day, variable)
            all_daily_values.extend(daily_values)
        except FileNotFoundError:
            warnings.warn(f"Weather data for training year {year} not found, skipping.")
    
    if not all_daily_values:
        raise ValueError(f"No training {variable} data could be loaded.")
    
    all_daily_values = np.array(all_daily_values)
    
    return TrainingStats(
        mean=float(np.mean(all_daily_values)),
        std=float(np.std(all_daily_values)),
        n_samples=len(all_daily_values),
        variable=variable
    )


def compute_test_mean_variable(
    growth_year: int,
    start_day: int,
    season_length: int = DEFAULT_SEASON_LENGTH,
    location: str = LOCATION,
    variable: str = "temperature"
) -> float:
    """Compute mean daily value of a weather variable for a test environment."""
    df = load_weather_data(growth_year, location)
    end_day = start_day + season_length
    daily_values = compute_daily_mean_variable(df, start_day, end_day, variable)
    
    if len(daily_values) == 0:
        raise ValueError(f"No {variable} data for year {growth_year}, day {start_day}")
    
    return float(np.mean(daily_values))


def compute_deviation(test_mean: float, train_stats: TrainingStats) -> float:
    """Compute weather variable deviation in standard deviation units."""
    if train_stats.std == 0:
        warnings.warn("Training std is zero, returning 0 deviation.")
        return 0.0
    return (test_mean - train_stats.mean) / train_stats.std


# ==============================================================================
# RL-MPC Results Loading
# ==============================================================================

def load_mpc_selection_data(
    results_dir: Path,
    rlmpc_folder: str,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int = 1
) -> Tuple[float, int]:
    """
    Load MPC and RL costs and compute the fraction of times MPC was selected.
    
    The cost files have shape (n_timesteps, n_cost_components). We sum across
    cost components to get total cost per timestep, then compare MPC vs RL costs.
    MPC is selected when its total cost is lower than RL's total cost.
    
    Args:
        results_dir: Base results directory.
        rlmpc_folder: RL-MPC subfolder name.
        location: Location string.
        growth_year: Growth year.
        start_day: Start day.
        horizon: MPC horizon.
    
    Returns:
        Tuple of (mpc_fraction, n_timesteps).
    """
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    
    mpc_costs_file = rlmpc_dir / f"mpc_costs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    rl_costs_file = rlmpc_dir / f"rl_costs-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
    
    if not mpc_costs_file.exists():
        raise FileNotFoundError(f"MPC costs file not found: {mpc_costs_file}")
    if not rl_costs_file.exists():
        raise FileNotFoundError(f"RL costs file not found: {rl_costs_file}")
    
    # Load cost arrays - shape is (n_timesteps, n_cost_components)
    # Transpose to get (n_cost_components, n_timesteps) to match solver_costs.py convention
    mpc_costs = np.loadtxt(mpc_costs_file, delimiter=",").T
    rl_costs = np.loadtxt(rl_costs_file, delimiter=",").T
    
    # Sum costs along axis 0 (cost components) to get total cost per timestep
    # Result shape: (n_timesteps,)
    if mpc_costs.ndim > 1:
        mpc_costs_sum = np.sum(mpc_costs, axis=0)
    else:
        mpc_costs_sum = mpc_costs
        
    if rl_costs.ndim > 1:
        rl_costs_sum = np.sum(rl_costs, axis=0)
    else:
        rl_costs_sum = rl_costs
    
    # Compute fraction of times MPC was selected (MPC selected when its cost is lower)
    chosen_is_mpc = mpc_costs_sum < rl_costs_sum
    mpc_fraction = np.mean(chosen_is_mpc)
    n_timesteps = len(chosen_is_mpc)
    
    return mpc_fraction, n_timesteps


def load_runs(
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    results_dir: Path,
    rlmpc_folder: Optional[str] = None,
    location: str = LOCATION,
    horizon: int = 1,
    variable: str = "temperature"
) -> List[RLMPCResult]:
    """
    Load all RL-MPC experiment runs for the specified parameters.
    
    Args:
        model_name: Name of the RL model.
        start_days: List of start days for experiments.
        growth_years: List of growth years to load.
        results_dir: Base results directory.
        rlmpc_folder: RL-MPC subfolder name (derived from model_name if None).
        location: Location string.
        horizon: MPC horizon.
        variable: Weather variable to compute deviation for.
    
    Returns:
        List of RLMPCResult objects.
    """
    if rlmpc_folder is None:
        rlmpc_folder = f"rlmpc-switch-{model_name}-0.05"
    
    results = []
    
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    if not rlmpc_dir.exists():
        warnings.warn(f"RL-MPC folder not found: {rlmpc_dir}")
        return results
    
    for start_day in start_days:
        for year in growth_years:
            try:
                mpc_fraction, n_timesteps = load_mpc_selection_data(
                    results_dir, rlmpc_folder, location, year, start_day, horizon
                )
                test_mean = compute_test_mean_variable(
                    year, start_day, location=location, variable=variable
                )
                results.append(RLMPCResult(
                    growth_year=year,
                    start_day=start_day,
                    location=location,
                    mpc_fraction=mpc_fraction,
                    n_timesteps=n_timesteps,
                    test_mean_variable=test_mean
                ))
            except Exception as e:
                # Only warn for single start_day to avoid excessive warnings
                if len(start_days) == 1:
                    warnings.warn(f"Error loading RL-MPC run for year {year}, day {start_day}: {e}")
    
    return results


def aggregate_data(
    runs: List[RLMPCResult],
    train_stats: TrainingStats
) -> Dict[str, np.ndarray]:
    """
    Aggregate run data for plotting.
    
    Returns:
        Dictionary with "weather_variable", "mpc_fractions", "years", and "start_days" arrays.
    """
    if not runs:
        return {}
    
    weather_variable = []
    mpc_fractions = []
    years = []
    start_days = []
    
    for run in runs:
        dev = compute_deviation(run.test_mean_variable, train_stats)
        # deviations.append(dev)
        weather_variable.append(run.test_mean_variable)
        mpc_fractions.append(run.mpc_fraction)
        years.append(run.growth_year)
        start_days.append(run.start_day)
    
    return {
        "weather_variable": np.array(weather_variable),
        "mpc_fractions": np.array(mpc_fractions),
        "years": np.array(years),
        "start_days": np.array(start_days)
    }


# ==============================================================================
# Plotting
# ==============================================================================

def make_plot(
    aggregated: Dict[str, np.ndarray],
    train_stats: TrainingStats,
    output_path: Path,
    variable: str = "temperature",
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create the MPC selection fraction plot.
    
    Args:
        aggregated: Aggregated data from aggregate_data().
        train_stats: Training weather variable statistics.
        output_path: Path to save the figure.
        variable: Weather variable used for x-axis.
        show: Whether to display the plot.
    
    Returns:
        Tuple of (figure, axes).
    """
    if not aggregated:
        raise ValueError("No data to plot.")
    
    # Get x-axis label from variable configuration
    x_axis_label = WEATHER_VARIABLES.get(variable, {}).get(
        "axis_label", 
        f"{variable.capitalize()} Deviation ($\\sigma$ units)"
    )
    
    # Figure setup
    WIDTH = 130 / 3 * 0.03937  # Journal width
    HEIGHT = WIDTH * 1.0
    SUBPLOT_KW = dict[str, float](left=0.3, right=0.97, top=0.97, bottom=0.3)
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)
    
    x_data = aggregated["weather_variable"]
    y_data = aggregated["mpc_fractions"]
    
    # Sort by deviation for cleaner plotting
    sort_idx = np.argsort(x_data)
    weather_variable_sorted = x_data[sort_idx]
    y_sorted = y_data[sort_idx]
    
    # Plot scatter points
    ax.scatter(
        weather_variable_sorted, 
        y_sorted, 
        color="C3",
        # marker="^",
        marker="o",
        s=50,
        alpha=0.8,
        label="RL-MPC",
        edgecolor="white",
        linewidth=0.5,
        zorder=3
    )

    # Add trend line
    if len(weather_variable_sorted) > 2:
        z = np.polyfit(weather_variable_sorted, y_sorted, 1)
        p = np.poly1d(z)
        weather_variable_line = np.linspace(weather_variable_sorted.min(), weather_variable_sorted.max(), 100)
        ax.plot(
            weather_variable_line, 
            p(weather_variable_line), 
            color="C3",
            linestyle="-",
            linewidth=2,
            alpha=0.5,
            zorder=2
        )

    # Axis labels
    ax.set_xlabel(WEATHER_VARIABLES[variable]["axis_label"])
    ax.set_ylabel(r"$\hat{J}_{\text{RL-MPC}} \geq \hat{J}_{\mathrm{RL}}$")
    ax.set_yticks(np.linspace(0, 1, 3))
    # ax.set_yticklabels([f"{x:.0%}" for x in np.linspace(0, 1, 3)])

    # Set y-axis limits (fraction is between 0 and 1)
    ax.set_ylim(-0.05, 1.05)
    
    # Add horizontal reference lines
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.7, linewidth=2, zorder=1,
               label="50% threshold")
    
    # Add vertical line at training mean (0 deviation)
    ax.axvline(x=train_stats.mean, color="#d62728", linestyle="--", alpha=0.7, linewidth=2, zorder=1)
    ax.axvline(x=train_stats.mean+train_stats.std, color="gray", linestyle=":", alpha=0.7, linewidth=2, zorder=1)
    ax.axvline(x=train_stats.mean-train_stats.std, color="gray", linestyle=":", alpha=0.7, linewidth=2, zorder=1)
    
    # Legend
    # ax.legend(loc="best", framealpha=0.9)
    
    # Grid
    # ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
    
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


def print_summary(
    runs: List[RLMPCResult],
    train_stats: TrainingStats
) -> None:
    """Print a summary of loaded runs and their MPC selection fractions."""
    var_config = WEATHER_VARIABLES.get(train_stats.variable, {})
    var_label = var_config.get("label", train_stats.variable.capitalize())
    var_unit = var_config.get("unit", "")
    
    print("\n" + "=" * 90)
    print("MPC Selection Fraction Analysis Summary")
    print("=" * 90)
    print(f"\nTraining {var_label} Stats:")
    print(f"  Mean: {train_stats.mean:.2f}{var_unit}")
    print(f"  Std:  {train_stats.std:.2f}{var_unit}")
    print(f"  N:    {train_stats.n_samples} daily samples")
    
    if not runs:
        print("\nNo RL-MPC runs found.")
        return
    
    print(f"\nRL-MPC Results ({len(runs)} runs):")
    print(f"  {'Year':<6} {'Day':<5} {var_label:<14} {'Deviation':<12} {'MPC Frac':<12} {'N Steps':<10}")
    print(f"  {'-'*6} {'-'*5} {'-'*14} {'-'*12} {'-'*12} {'-'*10}")
    
    for run in sorted(runs, key=lambda r: (r.growth_year, r.start_day)):
        dev = compute_deviation(run.test_mean_variable, train_stats)
        print(f"  {run.growth_year:<6} {run.start_day:<5} {run.test_mean_variable:>12.2f}{var_unit} {dev:>+10.2f}σ  {run.mpc_fraction:>10.2%}  {run.n_timesteps:>8}")

    # Summary statistics
    fractions = [r.mpc_fraction for r in runs]
    print(f"\n  Overall MPC Selection: {np.mean(fractions):.2%} ± {np.std(fractions):.2%}")
    print("=" * 90)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function to run the MPC selection fraction visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize MPC selection fraction vs weather variable deviation."
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
        help="Base directory for results (default: results/GL-MPC-RL/deterministic)"
    )
    parser.add_argument(
        "--out_dir", 
        type=str, 
        default="outputs/figures/mpc_selection_fraction",
        help="Output directory for figures (default: outputs/figures)"
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
        help="Years to evaluate on (default: 2011-2020)"
    )
    parser.add_argument(
        "--training_years", 
        type=int, 
        nargs="+",
        default=DEFAULT_TRAINING_YEARS,
        help=f"Training years for baseline stats (default: {DEFAULT_TRAINING_YEARS})"
    )
    parser.add_argument(
        "--training_start_day", 
        type=int,
        default=DEFAULT_TRAINING_START_DAY,
        help=f"Training period start day (default: {DEFAULT_TRAINING_START_DAY})"
    )
    parser.add_argument(
        "--training_end_day", 
        type=int,
        default=DEFAULT_TRAINING_END_DAY,
        help=f"Training period end day (default: {DEFAULT_TRAINING_END_DAY})"
    )
    parser.add_argument(
        "--location", 
        type=str, 
        default=LOCATION,
        help=f"Location for weather data (default: {LOCATION})"
    )
    parser.add_argument(
        "--variable", 
        type=str, 
        choices=["temperature", "humidity", "radiation"],
        default="temperature",
        help="Weather variable for x-axis (default: temperature)"
    )
    parser.add_argument(
        "--horizon", 
        type=int, 
        default=1,
        help="MPC horizon (default: 1)"
    )
    parser.add_argument(
        "--show", 
        action="store_true",
        help="Show plot interactively"
    )
    
    args = parser.parse_args()
    
    # Get variable label for display
    var_label = WEATHER_VARIABLES.get(args.variable, {}).get("label", args.variable)
    
    # Compute training statistics
    print(f"Computing training {var_label.lower()} statistics...")
    train_stats = compute_training_stats(
        training_years=args.training_years,
        start_day=args.training_start_day,
        end_day=args.training_end_day,
        location=args.location,
        variable=args.variable
    )
    
    # Load runs
    print(f"Loading RL-MPC results from {args.results_dir}...")
    print(f"  Start days: {args.start_days}")
    print(f"  Test years: {args.test_years}")
    results_dir = Path(args.results_dir)
    
    runs = load_runs(
        model_name=args.model_name,
        start_days=args.start_days,
        growth_years=args.test_years,
        results_dir=results_dir,
        rlmpc_folder=args.rlmpc_folder,
        location=args.location,
        horizon=args.horizon,
        variable=args.variable
    )
    
    # Print summary
    print_summary(runs, train_stats)
    
    # Check if we have any data
    if not runs:
        print("\nERROR: No RL-MPC runs found. Check your --results_dir and --model_name arguments.")
        return 1
    
    # Aggregate data
    aggregated = aggregate_data(runs, train_stats)
    
    # Create plot
    output_path = Path(args.out_dir) / f"{args.variable}"
    
    print(f"\nGenerating plot...")
    make_plot(
        aggregated=aggregated,
        train_stats=train_stats,
        output_path=output_path,
        variable=args.variable,
        show=args.show
    )
    
    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
