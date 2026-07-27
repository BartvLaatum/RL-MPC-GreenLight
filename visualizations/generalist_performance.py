"""
Visualize controller performance vs weather variable deviation from training environment.

This script plots how RL-MPC, RL, and MPC controller performance changes as the
test environment deviates from the training environment, where deviation is
measured in standard deviation units of a daily weather variable (temperature,
humidity, or radiation).

Usage:
    # Single start day
    python visualizations/generalist_performance.py \
        --model_name daily-glade-107 \
        --start_days 151

    # Multiple start days and years
    python visualizations/generalist_performance.py \
        --model_name daily-glade-107 \
        --start_days 90 105 120 135 151 \
        --test_years 2011 2012 2019 2020
"""

import argparse
import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import plot configuration for consistent styling
# Handle both direct execution and module import
try:
    from visualizations.plot_config import *
except ModuleNotFoundError:
    import plot_config


# ==============================================================================
# Constants
# ==============================================================================

WEATHER_DATA_DIR = Path("weather")
LOCATION = "Netherlands"
SECONDS_IN_DAY = 86400

# Default training configuration (from mean_daily_dist.py and config files)
DEFAULT_TRAINING_YEARS = list(range(2013, 2019))  # 2013-2018
DEFAULT_TRAINING_START_DAY = 90
DEFAULT_TRAINING_END_DAY = 152
DEFAULT_SEASON_LENGTH = 1  # days per episode

# Controller identifiers
CONTROLLER_TYPES = ["rl", "mpc", "rlmpc"]

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
    test_mean_variable: float  # Mean value of the weather variable for the test environment


@dataclass
class TrainingStats:
    """Statistics for the training weather variable distribution."""
    mean: float
    std: float
    n_samples: int
    variable: str = "temperature"  # Which variable these stats are for


# ==============================================================================
# Weather Data Loading
# ==============================================================================

def load_weather_data(year: int, location: str = LOCATION) -> pd.DataFrame:
    """
    Load raw weather data for a given year.
    
    Args:
        year: The year to load data for.
        location: Location folder name.
    
    Returns:
        DataFrame with weather data.
    
    Raises:
        FileNotFoundError: If weather file doesn't exist.
    """
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
    """
    Compute daily mean values for a weather variable over a range of days.
    
    Args:
        df: DataFrame with weather data.
        start_day: Start day of the year (inclusive).
        end_day: End day of the year (exclusive).
        variable: Weather variable to compute ("temperature", "humidity", or "radiation").
    
    Returns:
        Array of daily mean values.
    """
    # Get column name for the variable
    if variable not in WEATHER_VARIABLES:
        raise ValueError(f"Unknown variable: {variable}. Must be one of {list(WEATHER_VARIABLES.keys())}")
    
    column_name = WEATHER_VARIABLES[variable]["column"]
    
    start_time = start_day * SECONDS_IN_DAY
    end_time = end_day * SECONDS_IN_DAY
    
    mask = (df["time"] >= start_time) & (df["time"] < end_time)
    filtered_df = df.loc[mask].copy()
    
    if filtered_df.empty:
        return np.array([])
    
    # Compute day number from time
    filtered_df["day"] = (filtered_df["time"] // SECONDS_IN_DAY).astype(int)
    
    # Group by day and compute mean
    daily_avg = filtered_df.groupby("day")[column_name].mean().values
    
    return daily_avg


def compute_training_stats(
    training_years: List[int] = DEFAULT_TRAINING_YEARS,
    start_day: int = DEFAULT_TRAINING_START_DAY,
    end_day: int = DEFAULT_TRAINING_END_DAY,
    location: str = LOCATION,
    variable: str = "temperature"
) -> TrainingStats:
    """
    Compute mean and std of a daily weather variable over the training set.
    
    Args:
        training_years: List of years in the training set.
        start_day: Start day for training period.
        end_day: End day for training period.
        location: Location of weather data.
        variable: Weather variable to compute stats for.
    
    Returns:
        TrainingStats with mean, std, and sample count.
    """
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
    """
    Compute mean daily value of a weather variable for a test environment.
    
    Args:
        growth_year: Year of the test environment.
        start_day: Start day for the test period.
        season_length: Number of days in the test episode.
        location: Location of weather data.
        variable: Weather variable to compute.

    Returns:
        Mean value over the test period.
    """
    df = load_weather_data(growth_year, location)
    end_day = start_day + season_length
    daily_values = compute_daily_mean_variable(df, start_day, end_day, variable)

    if len(daily_values) == 0:
        raise ValueError(f"No {variable} data for year {growth_year}, day {start_day}")
    
    return float(np.mean(daily_values))


def compute_deviation(
    test_mean: float, 
    train_stats: TrainingStats
) -> float:
    """
    Compute temperature deviation in standard deviation units.
    
    Args:
        test_mean: Mean temperature of the test environment.
        train_stats: Training set statistics.
    
    Returns:
        Deviation z-score: (test_mean - train_mean) / train_std
    """
    if train_stats.std == 0:
        warnings.warn("Training std is zero, returning 0 deviation.")
        return 0.0
    
    return (test_mean - train_stats.mean) / train_stats.std


# ==============================================================================
# Results Loading
# ==============================================================================

# def discover_rl_runs(
#     results_dir: Path,
#     model_name: str,
#     location: str,
#     start_day: int
# ) -> List[Dict[str, Any]]:
#     """
#     Discover RL result files matching the given parameters.
    
#     Args:
#         results_dir: Base results directory.
#         model_name: Name of the RL model.
#         location: Location string.
#         start_day: Start day.
    
#     Returns:
#         List of dictionaries with file path and growth_year.
#     """
#     rl_dir = results_dir / "ppo"
#     if not rl_dir.exists():
#         return []
    
#     runs = []
#     # Pattern: {model_name}-{location}-{growth_year}-{start_day}.csv
#     for csv_file in rl_dir.glob(f"{model_name}-{location}-*-{start_day}.csv"):
#         # Extract growth_year from filename
#         parts = csv_file.stem.split("-")
#         # Find the growth year (second to last part before start_day)
#         try:
#             # Filename format: model_name-location-growth_year-start_day
#             # But model_name can contain hyphens, so we parse from the end
#             growth_year = int(parts[-2])
#             runs.append({
#                 "path": csv_file,
#                 "growth_year": growth_year
#             })
#         except (ValueError, IndexError):
#             warnings.warn(f"Could not parse growth year from {csv_file.name}")
    
#     return runs


# def discover_mpc_runs(
#     results_dir: Path,
#     mpc_folder: str,
#     location: str,
#     start_day: int,
#     horizon: int = 1
# ) -> List[Dict[str, Any]]:
#     """
#     Discover MPC result files matching the given parameters.
    
#     Args:
#         results_dir: Base results directory.
#         mpc_folder: MPC subfolder name.
#         location: Location string.
#         start_day: Start day.
#         horizon: MPC horizon.
    
#     Returns:
#         List of dictionaries with file path and growth_year.
#     """
#     mpc_dir = results_dir / "mpc" / mpc_folder
#     if not mpc_dir.exists():
#         return []
    
#     runs = []
#     # Pattern: rewards-300dt-{H}H-{location}-{growth_year}-{start_day}.csv
#     for csv_file in mpc_dir.glob(f"rewards-300dt-{horizon}H-{location}-*-{start_day}.csv"):
#         parts = csv_file.stem.split("-")
#         try:
#             # Format: rewards-300dt-{H}H-{location}-{growth_year}-{start_day}
#             growth_year = int(parts[-2])
#             runs.append({
#                 "path": csv_file,
#                 "growth_year": growth_year
#             })
#         except (ValueError, IndexError):
#             warnings.warn(f"Could not parse growth year from {csv_file.name}")
    
#     return runs


# def discover_rlmpc_runs(
#     results_dir: Path,
#     rlmpc_folder: str,
#     location: str,
#     start_day: int,
#     horizon: int = 1
# ) -> List[Dict[str, Any]]:
#     """
#     Discover RL-MPC result files matching the given parameters.
    
#     Args:
#         results_dir: Base results directory.
#         rlmpc_folder: RL-MPC subfolder name.
#         location: Location string.
#         start_day: Start day.
#         horizon: MPC horizon.
    
#     Returns:
#         List of dictionaries with file path and growth_year.
#     """
#     rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
#     if not rlmpc_dir.exists():
#         return []
    
#     runs = []
#     # Pattern: rewards-300dt-{H}H-{location}-{growth_year}-{start_day}.csv
#     for csv_file in rlmpc_dir.glob(f"rewards-300dt-{horizon}H-{location}-*-{start_day}.csv"):
#         parts = csv_file.stem.split("-")
#         try:
#             growth_year = int(parts[-2])
#             runs.append({
#                 "path": csv_file,
#                 "growth_year": growth_year
#             })
#         except (ValueError, IndexError):
#             warnings.warn(f"Could not parse growth year from {csv_file.name}")
    
#     return runs


def load_rl_performance(run_info: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Load performance metrics from an RL result file.
    
    Args:
        run_info: Dictionary with 'path' key pointing to CSV file.
    
    Returns:
        Tuple of (total_reward, total_epi, total_penalty).
    """
    df = pd.read_csv(run_info["path"])
    
    # Get reward column (could be 'Rewards' or 'rewards')
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
    """
    Load performance metrics from MPC result files.
    
    Args:
        results_dir: Base results directory.
        mpc_folder: MPC subfolder name.
        location: Location string.
        growth_year: Growth year.
        start_day: Start day.
        horizon: MPC horizon.
    
    Returns:
        Tuple of (total_reward, total_epi, total_penalty).
    """
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
    """
    Load performance metrics from RL-MPC result files.
    
    Args:
        results_dir: Base results directory.
        rlmpc_folder: RL-MPC subfolder name.
        location: Location string.
        growth_year: Growth year.
        start_day: Start day.
        horizon: MPC horizon.
    
    Returns:
        Tuple of (total_reward, total_epi, total_penalty).
    """
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


# ==============================================================================
# Main Data Loading Function
# ==============================================================================

def load_runs(
    model_name: str,
    start_days: List[int],
    growth_years: List[int],
    results_dir: Path,
    mpc_folder: str = "mpc_linear_solver_ma57",
    rlmpc_folder: Optional[str] = None,
    location: str = LOCATION,
    horizon: int = 1,
    variable: str = "temperature"
) -> Dict[str, List[RunResult]]:
    """
    Load all experiment runs for the specified parameters.
    
    Args:
        model_name: Name of the RL model.
        start_days: List of start days for experiments.
        growth_years: List of growth years to load.
        results_dir: Base results directory.
        mpc_folder: MPC subfolder name.
        rlmpc_folder: RL-MPC subfolder name (derived from model_name if None).
        location: Location string.
        horizon: MPC horizon.
        variable: Weather variable to compute deviation for.
    
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
    
    # Load RL runs - iterate over all combinations of start_day and year
    for start_day in start_days:
        for year in growth_years:
            rl_file = results_dir / "ppo" / f"{model_name}-{location}-{year}-{start_day}.csv"
            if rl_file.exists():
                try:
                    reward, epi, penalty = load_rl_performance({"path": rl_file})
                    test_mean = compute_test_mean_variable(year, start_day, location=location, variable=variable)
                    results["rl"].append(RunResult(
                        controller_type="rl",
                        model_name=model_name,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty,
                        test_mean_variable=test_mean
                    ))
                except Exception as e:
                    warnings.warn(f"Error loading RL run for year {year}, day {start_day}: {e}")
            # Only warn if we have a single start_day (to avoid excessive warnings)
            elif len(start_days) == 1:
                warnings.warn(f"RL results not found for year {year}, day {start_day}: {rl_file}")
    
    # Load MPC runs
    mpc_dir = results_dir / "mpc" / mpc_folder
    if mpc_dir.exists():
        for start_day in start_days:
            for year in growth_years:
                try:
                    reward, epi, penalty = load_mpc_performance(
                        results_dir, mpc_folder, location, year, start_day, horizon
                    )
                    test_mean = compute_test_mean_variable(year, start_day, location=location, variable=variable)
                    results["mpc"].append(RunResult(
                        controller_type="mpc",
                        model_name=mpc_folder,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty,
                        test_mean_variable=test_mean
                    ))
                except Exception as e:
                    # Only warn for single start_day to avoid excessive warnings
                    if len(start_days) == 1:
                        warnings.warn(f"Error loading MPC run for year {year}, day {start_day}: {e}")
    else:
        warnings.warn(f"MPC folder not found: {mpc_dir}")
    
    # Load RL-MPC runs
    rlmpc_dir = results_dir / "rlmpc" / rlmpc_folder
    print(f"RL-MPC directory: {rlmpc_dir}")
    if rlmpc_dir.exists():
        for start_day in start_days:
            for year in growth_years:
                try:
                    reward, epi, penalty = load_rlmpc_performance(
                        results_dir, rlmpc_folder, location, year, start_day, horizon
                    )
                    test_mean = compute_test_mean_variable(year, start_day, location=location, variable=variable)
                    results["rlmpc"].append(RunResult(
                        controller_type="rlmpc",
                        model_name=rlmpc_folder,
                        growth_year=year,
                        start_day=start_day,
                        location=location,
                        total_reward=reward,
                        total_epi=epi,
                        total_penalty=penalty,
                        test_mean_variable=test_mean
                    ))
                except Exception as e:
                    # Only warn for single start_day to avoid excessive warnings
                    if len(start_days) == 1:
                        warnings.warn(f"Error loading RL-MPC run for year {year}, day {start_day}: {e}")
    else:
        warnings.warn(f"RL-MPC folder not found: {rlmpc_dir}")
    
    return results


def aggregate_performance(
    runs: Dict[str, List[RunResult]],
    train_stats: TrainingStats
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Aggregate performance data by computing deviation and organizing for plotting.
    
    Args:
        runs: Dictionary mapping controller type to list of RunResult objects.
        train_stats: Training weather variable statistics.
    
    Returns:
        Dictionary with structure:
        {
            controller_type: {
                "weather_variable": np.ndarray,  # deviation values (in sigma units)
                "rewards": np.ndarray,
                "epis": np.ndarray,
                "penalties": np.ndarray,
                "years": np.ndarray,
                "start_days": np.ndarray
            }
        }
    """
    aggregated = {}
    
    for controller_type, run_list in runs.items():
        if not run_list:
            continue
        
        weather_variable = []
        rewards = []
        epis = []
        penalties = []
        years = []
        start_days = []

        for run in run_list:
            dev = compute_deviation(run.test_mean_variable, train_stats)
            weather_variable.append(run.test_mean_variable)
            rewards.append(run.total_reward)
            epis.append(run.total_epi)
            penalties.append(run.total_penalty)
            years.append(run.growth_year)
            start_days.append(run.start_day)

        aggregated[controller_type] = {
            "weather_variable": np.array(weather_variable),
            "rewards": np.array(rewards),
            "epis": np.array(epis),
            "penalties": np.array(penalties),
            "years": np.array(years),
            "start_days": np.array(start_days)
        }
    
    return aggregated

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
                # else:
                    # rel_diff = 0.0
                
                differences[controller_type][metric].append(diff)
    
    return differences


# ==============================================================================
# Plotting
# ==============================================================================

def make_plot(
    aggregated: Dict[str, Dict[str, np.ndarray]],
    train_stats: TrainingStats,
    output_path: Path,
    metric: str = "rewards",
    variable: str = "temperature",
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create the generalist performance plot showing difference between
    RL-MPC and other controllers (RL, MPC).
    
    Args:
        aggregated: Aggregated performance data from aggregate_performance().
        train_stats: Training weather variable statistics.
        output_path: Path to save the figure.
        metric: Which metric to plot ("rewards", "epis", or "penalties").
        variable: Weather variable used for x-axis ("temperature", "humidity", or "radiation").
        show: Whether to display the plot.
    
    Returns:
        Tuple of (figure, axes).
    """
    # Check that RL-MPC data exists (required as reference)
    if "rlmpc" not in aggregated:
        raise ValueError("RL-MPC data is required to compute differences.")

    # Styling for difference comparisons
    comparison_config = {
        "rl": {
            "label": r"$\Delta$ RL-MPC vs RL",
            "color": "#1f77b4",
            "marker": "o",
            "linestyle": "-"
        },
        "mpc": {
            "label": r"$\Delta$ RL-MPC vs MPC",
            "color": "#00a693",
            "marker": "o",
            "linestyle": "-"
        }
    }

    # Metric labels for y-axis
    metric_labels = {
        "rewards": r"$\Delta \mathcal{J}$",
        "epis": r"$\Delta \text{ EPI}$ (EUR/m$^2$)",
        "penalties": r"$\Delta \text{ Penalty}$"
    }

    # Get x-axis label from variable configuration
    # x_axis_label = WEATHER_VARIABLES.get(variable, {}).get(
    #     "axis_label", 
    #     f"{variable.capitalize()} Deviation ($\\sigma$ units)"
    # )

    # Get RL-MPC data indexed by (year, start_day) for easy lookup
    rlmpc_data = aggregated["rlmpc"]
    rlmpc_by_key = {}
    for i in range(len(rlmpc_data["years"])):
        key = (rlmpc_data["years"][i], rlmpc_data["start_days"][i])
        rlmpc_by_key[key] = {
            "weather_variable": rlmpc_data["weather_variable"][i],
            "rewards": rlmpc_data["rewards"][i],
            "epis": rlmpc_data["epis"][i],
            "penalties": rlmpc_data["penalties"][i]
        }

    # Figure setup
    WIDTH = 130 / 3 * 0.03937  # Journal width
    HEIGHT = WIDTH * 1.0
    SUBPLOT_KW = dict[str, float](left=0.3, right=0.97, top=0.97, bottom=0.3)
    fig, ax = plt.subplots(figsize=(WIDTH*2, HEIGHT), dpi=300)

    # Plot difference for each baseline controller (RL, MPC)
    for controller_type in ["rl", "mpc"]:
        if controller_type not in aggregated:
            warnings.warn(f"No data for controller: {controller_type}, skipping difference plot.")
            continue

        data = aggregated[controller_type]
        config = comparison_config[controller_type]

        # Compute differences at matching (year, start_day) combinations
        weather_variable = []
        differences = []

        for i in range(len(data["years"])):
            year = data["years"][i]
            start_day = data["start_days"][i]
            key = (year, start_day)
            
            if key not in rlmpc_by_key:
                warnings.warn(f"Year {year}, day {start_day} not found in RL-MPC data, skipping.")
                continue

            # Get RL-MPC value for this (year, start_day)
            rlmpc_val = rlmpc_by_key[key]
            
            # Get baseline controller value
            if metric == "rewards":
                baseline_val = data["rewards"][i]
                rlmpc_metric = rlmpc_val["rewards"]
            elif metric == "epis":
                baseline_val = data["epis"][i]
                rlmpc_metric = rlmpc_val["epis"]
            else:
                baseline_val = data["penalties"][i]
                rlmpc_metric = rlmpc_val["penalties"]

            # Compute difference
            diff = rlmpc_metric - baseline_val
            diff_percent = diff / abs(baseline_val) * 100
            weather_variable.append(data["weather_variable"][i])
            differences.append(diff)

        if not weather_variable:
            warnings.warn(f"No matching (year, start_day) combinations for {controller_type}")
            continue
        x_data = np.array(weather_variable)
        y_data = np.array(differences)

        # Sort by deviation for cleaner line plots
        sort_idx = np.argsort(weather_variable)
        weather_variable_sorted = x_data[sort_idx]
        y_sorted = y_data[sort_idx]

        # Plot scatter points
        ax.scatter(
            weather_variable_sorted, 
            y_sorted, 
            color=config["color"],
            marker=config["marker"],
            s=50,
            alpha=0.5,
            label=config["label"],
            edgecolor="white",
            linewidth=0.5,
            zorder=3
        )

        # Add trend line
        if len(weather_variable_sorted) > 2:
            z = np.polyfit(weather_variable_sorted, y_sorted, 1)
            # z = np.polyfit(x_sorted, y_sorted, 2)
            p = np.poly1d(z)
            weather_variable_line = np.linspace(weather_variable_sorted.min(), weather_variable_sorted.max(), 100)
            ax.plot(
                weather_variable_line, 
                p(weather_variable_line), 
                color=config["color"],
                linestyle=config["linestyle"],
                linewidth=2,
                alpha=1.,
                zorder=2
            )

    # Axis labels
    ax.set_xlabel(WEATHER_VARIABLES[variable]["axis_label"])
    ax.set_ylabel(metric_labels.get(metric, f"$\Delta$ {metric}"))

    # Add vertical line at training mean (0 deviation)
    ax.axvline(x=train_stats.mean, color="#d62728", linestyle="--", alpha=0.7, linewidth=2, zorder=1)
    ax.axvline(x=train_stats.mean+train_stats.std, color="gray", linestyle=":", alpha=0.7, linewidth=2, zorder=1)
    ax.axvline(x=train_stats.mean-train_stats.std, color="gray", linestyle=":", alpha=0.7, linewidth=2, zorder=1)

    # Add horizontal line at y=0 for reference
    # ax.axhline(y=0, color="gray", linestyle="--", alpha=0.7, linewidth=1, zorder=1)
    
    # Ensure y-axis starts at 0 (since we're plotting differences)
    # if metric == "rewards" or metric == "epis":
        # ax.set_ylim(bottom=0)
    # xmax = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]))
    # ax.set_xlim(-xmax, xmax)
    # Legend
    ax.legend(loc="upper right", frameon=False)

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

def scatter_plot_deviation(
    aggregated: Dict[str, Dict[str, np.ndarray]],
    train_stats: TrainingStats,
    output_path: Path,
    metric: str = "rewards",
    variable: str = "temperature",
    show: bool = False
) -> Tuple[plt.Figure, plt.Axes]:
    """
    Create a contour plot of the performance data."""

    x_data = aggregated["weather_variable"]["radiation"]
    y_data = aggregated["weather_variable"]["humidity"]
    z_data = aggregated["rewards"]

    # Sort by deviation for cleaner plotting
    sort_idx = np.argsort(x_data["weather_variable"])
    weather_variable_sorted = x_data["weather_variable"][sort_idx]
    y_sorted = x_data["rewards"][sort_idx]

    # Plot scatter points

def print_summary(
    runs: Dict[str, List[RunResult]],
    train_stats: TrainingStats
) -> None:
    """
    Print a summary of loaded runs and their deviations.
    
    Args:
        runs: Dictionary mapping controller type to list of RunResult objects.
        train_stats: Training weather variable statistics.
    """
    # Get variable info for display
    variable = train_stats.variable
    var_config = WEATHER_VARIABLES.get(variable, {})
    var_label = var_config.get("label", variable.capitalize())
    var_unit = var_config.get("unit", "")
    
    print("\n" + "=" * 85)
    print("Generalist Performance Analysis Summary")
    print("=" * 85)
    print(f"\nTraining {var_label} Stats:")
    print(f"  Mean: {train_stats.mean:.2f}{var_unit}")
    print(f"  Std:  {train_stats.std:.2f}{var_unit}")
    print(f"  N:    {train_stats.n_samples} daily samples")
    
    for controller_type, run_list in runs.items():
        if not run_list:
            print(f"\n{controller_type.upper()}: No runs found")
            continue
        
        print(f"\n{controller_type.upper()} ({len(run_list)} runs):")
        print(f"  {'Year':<6} {'Day':<5} {var_label:<12} {'Deviation':<12} {'Reward':<12} {'EPI':<10}")
        print(f"  {'-'*6} {'-'*5} {'-'*12} {'-'*12} {'-'*12} {'-'*10}")
        
        # Sort by year first, then by start_day
        for run in sorted(run_list, key=lambda r: (r.growth_year, r.start_day)):
            dev = compute_deviation(run.test_mean_variable, train_stats)
            print(f"  {run.growth_year:<6} {run.start_day:<5} {run.test_mean_variable:>10.2f}{var_unit} {dev:>+10.2f}σ  {run.total_reward:>10.4f}  {run.total_epi:>8.4f}")
    
    print("\n" + "=" * 85)


# ==============================================================================
# Main Entry Point
# ==============================================================================

def main():
    """Main function to run the generalist performance visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize controller performance vs temperature deviation from training."
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
        default="outputs/figures",
        help="Output directory for figures (default: outputs/figures)"
    )
    parser.add_argument(
        "--mpc_folder", 
        type=str, 
        default="mpc_linear_solver_ma57",
        help="MPC results subfolder (default: mpc_linear_solver_ma57)"
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
        help="Years to evaluate on (default: 2011-2020, 2023)"
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
        "--metric", 
        type=str, 
        choices=["rewards", "epis", "penalties"],
        default="rewards",
        help="Performance metric to plot (default: rewards)"
    )
    parser.add_argument(
        "--variable", 
        type=str, 
        choices=["temperature", "humidity", "radiation"],
        default="temperature",
        help="Variable to plot (default: temperature)"
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
        location=args.location,
        variable=args.variable
    )

    # Print summary
    print_summary(runs, train_stats)
    
    # Check if we have any data
    total_runs = sum(len(v) for v in runs.values())
    if total_runs == 0:
        print("\nERROR: No runs found. Check your --results_dir and --model_name arguments.")
        return 1

    # Aggregate data
    aggregated = aggregate_performance(runs, train_stats)

    # Create plot
    output_path = Path(args.out_dir) / f"{args.metric}_{args.variable}"

    print(f"\nGenerating plot...")
    make_plot(
        aggregated=aggregated,
        train_stats=train_stats,
        output_path=output_path,
        metric=args.metric,
        variable=args.variable,
        show=args.show
    )

    print("\nDone!")
    return 0


if __name__ == "__main__":
    exit(main())
