"""
Visualize the daily average distributions of the training set weather variables.
Training set: days 60-151 of years 2013-2018.

Variables:
- Temperature (°C)
- Relative Humidity (%)
- Solar Radiation (W/m²)

Includes normality tests (Shapiro-Wilk and D'Agostino-Pearson).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde, shapiro, normaltest, norm
from typing import Tuple, Dict

# Import plot configuration
from visualizations.plot_config import *

# Constants
WEATHER_DATA_DIR = "weather"
LOCATION = "Netherlands"
TRAINING_YEARS = list(range(2013, 2019))  # 2013-2018
START_DAY = 90
END_DAY = 151
SECONDS_IN_DAY = 86400


def load_raw_weather_data(year: int) -> pd.DataFrame:
    """
    Load raw weather data for a given year.
    
    Args:
        year: The year to load data for.
    
    Returns:
        DataFrame with weather data.
    """
    weather_path = Path(WEATHER_DATA_DIR) / LOCATION / f"{year}.csv"
    return pd.read_csv(weather_path)


def compute_daily_avg(df: pd.DataFrame, column: str, start_day: int, end_day: int) -> np.ndarray:
    """
    Compute daily averages for a specified column over a range of days.
    
    Args:
        df: DataFrame with weather data.
        column: Column name to compute daily average for.
        start_day: Start day of the year (inclusive).
        end_day: End day of the year (exclusive).
    
    Returns:
        Array of daily averages.
    """
    # Filter data for the specified day range
    start_time = start_day * SECONDS_IN_DAY
    end_time = end_day * SECONDS_IN_DAY
    
    mask = (df["time"] >= start_time) & (df["time"] < end_time)
    filtered_df = df.loc[mask].copy()
    
    # Compute day number from time
    filtered_df["day"] = (filtered_df["time"] // SECONDS_IN_DAY).astype(int)
    
    # Group by day and compute mean
    daily_avg = filtered_df.groupby("day")[column].mean().values
    
    return daily_avg


def test_normality(data: np.ndarray, name: str) -> Dict[str, Tuple[float, float]]:
    """
    Perform normality tests on the data.
    
    Args:
        data: Array of values to test.
        name: Name of the variable for printing.
    
    Returns:
        Dictionary with test results (statistic, p-value).
    """
    results = {}
    
    # Shapiro-Wilk test
    stat_sw, p_sw = shapiro(data)
    results['shapiro'] = (stat_sw, p_sw)
    
    # D'Agostino-Pearson test (requires n >= 20)
    if len(data) >= 20:
        stat_da, p_da = normaltest(data)
        results['dagostino'] = (stat_da, p_da)
    
    return results


def plot_single_distribution(
    data: np.ndarray, 
    xlabel: str, 
    color: str, 
    title: str,
    unit: str,
    normality_results: Dict[str, Tuple[float, float]],
    output_path: Path
):
    """
    Create a single figure for one variable's distribution with normality test results.
    
    Args:
        data: Array of values to plot.
        xlabel: Label for x-axis.
        color: Color for the histogram.
        title: Title for the plot.
        unit: Unit string for the variable.
        normality_results: Dictionary with normality test results.
        output_path: Path to save the figure.
    """
    WIDTH = 130 / 3 * 0.03937  # Journal width
    HEIGHT = WIDTH * 1.0
    SUBPLOT_KW = dict[str, float](left=0.3, right=0.97, top=0.97, bottom=0.3)
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    
    # Plot histogram with density
    n, bins, patches = ax.hist(
        data, 
        bins=25, 
        density=True, 
        alpha=0.7, 
        color=color,
        edgecolor='white',
        linewidth=0.5,
        label='Data'
    )
    
    # Add a kernel density estimate curve
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min() - (data.max() - data.min()) * 0.15, 
                          data.max() + (data.max() - data.min()) * 0.15, 200)
    ax.plot(x_range, kde(x_range), color='C0', linewidth=2, label='KDE')

    # Add fitted normal distribution for comparison
    mean_val = data.mean()
    std_val = data.std()
    # normal_pdf = norm.pdf(x_range, mean_val, std_val)
    # ax.plot(x_range, normal_pdf, color='red', linewidth=2, linestyle='--', 
    #         label=f'Normal fit (μ={mean_val:.1f}, σ={std_val:.1f})')
    
    # Add vertical lines for mean and std
    ax.axvline(mean_val, color="#d62728", label='$\mu$', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(mean_val - std_val, color="gray", label='$\sigma$', linestyle=':', linewidth=2, alpha=0.7)
    ax.axvline(mean_val + std_val, color="gray", linestyle=':', linewidth=2, alpha=0.7)
    
    # Labels and title
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    # ax.set_title(f"{title}\n(Days {START_DAY}-{END_DAY}, Years {TRAINING_YEARS[0]}-{TRAINING_YEARS[-1]})")

    # Create legend
    # ax.legend(loc='upper right')

    # Add normality test results as text box
    textstr = f"$\mu$    = {mean_val:.1f} {unit}\n"
    textstr += f"$\sigma$ = {std_val:.1f} {unit}"
    # textstr += f"─────────────────\n"
    # textstr += "Normality Tests:\n"
    
    # # Shapiro-Wilk
    # sw_stat, sw_p = normality_results['shapiro']
    # sw_result = "Normal" if sw_p > 0.05 else "Not Normal"
    # textstr += f"Shapiro-Wilk: p={sw_p:.4f}\n"
    # textstr += f"  → {sw_result} (α=0.05)\n"
    
    # # D'Agostino-Pearson
    # if 'dagostino' in normality_results:
    #     da_stat, da_p = normality_results['dagostino']
    #     da_result = "Normal" if da_p > 0.05 else "Not Normal"
    #     textstr += f"D'Agostino: p={da_p:.4f}\n"
    #     textstr += f"  → {da_result} (α=0.05)"
    
    # Position text box in upper left
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='top', bbox=props, family='monospace')

    # plt.tight_layout()
    fig.subplots_adjust(**SUBPLOT_KW)
    
    # Save figure
    print(f"Saving figure to {output_path.with_suffix('.png')}")
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    
    plt.close(fig)


def plot_combined_distribution(
    all_daily_temps: np.ndarray,
    all_daily_humidity: np.ndarray,
    all_daily_radiation: np.ndarray,
    output_dir: Path
):
    """
    Create a combined figure with all three distributions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    datasets = [
        (all_daily_temps, r"$\Delta\bar{d}_{\mathrm{T}}$ (°C)", "C0", "Temperature"),
        (all_daily_humidity, r"$\Delta\bar{d}_{\mathrm{RH}}$ (%)", "C0", "Relative Humidity"),
        (all_daily_radiation, r"$\Delta\bar{d}_{\mathrm{iGlob}}$ (W/m$^2$)", "C0", "Solar Radiation")
    ]
    
    for ax, (data, xlabel, color, title) in zip(axes, datasets):
        # Plot histogram with density
        ax.hist(
            data, 
            bins=20, 
            density=True, 
            alpha=0.7, 
            color=color,
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add a kernel density estimate curve
        kde = gaussian_kde(data)
        x_range = np.linspace(data.min() - (data.max() - data.min()) * 0.1, 
                              data.max() + (data.max() - data.min()) * 0.1, 200)
        ax.plot(x_range, kde(x_range), color='black', linewidth=2, label='KDE')
        
        # Add fitted normal distribution
        mean_val = data.mean()
        std_val = data.std()
        normal_pdf = norm.pdf(x_range, mean_val, std_val)
        ax.plot(x_range, normal_pdf, color='red', linewidth=2, linestyle='--', label='Normal')
        
        # Add vertical lines for mean
        ax.axvline(mean_val, color='darkred', linestyle='-', linewidth=1.5, alpha=0.7,
                   label=f'Mean: {mean_val:.1f}')
        
        # Labels and title
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Density")
        ax.set_title(title)
        ax.legend(loc='upper right', fontsize=8)
    
    # Add overall title
    fig.suptitle(
        f"Training Set Weather Distributions (Days {START_DAY}-{END_DAY}, Years {TRAINING_YEARS[0]}-{TRAINING_YEARS[-1]})",
        fontsize=12, 
        fontweight='bold',
        y=1.02
    )
    
    plt.tight_layout()
    
    # Save figure
    output_path = output_dir / "training_weather_distribution"
    plt.savefig(output_path.with_suffix('.png'), dpi=150, bbox_inches='tight')
    plt.savefig(output_path.with_suffix('.svg'), bbox_inches='tight')
    
    plt.close(fig)


def main():
    """Main function to generate the weather distribution plots."""
    # Collect daily averages from all training years
    all_daily_temps = []
    all_daily_humidity = []
    all_daily_radiation = []
    
    for year in TRAINING_YEARS:
        print(f"Processing year {year}...")
        df = load_raw_weather_data(year)
        
        # Temperature
        daily_temps = compute_daily_avg(df, "air temperature", START_DAY, END_DAY)
        all_daily_temps.extend(daily_temps)
        
        # Humidity
        daily_humidity = compute_daily_avg(df, "RH", START_DAY, END_DAY)
        all_daily_humidity.extend(daily_humidity)
        
        # Solar Radiation
        daily_radiation = compute_daily_avg(df, "global radiation", START_DAY, END_DAY)
        all_daily_radiation.extend(daily_radiation)
    
    all_daily_temps = np.array(all_daily_temps)
    all_daily_humidity = np.array(all_daily_humidity)
    all_daily_radiation = np.array(all_daily_radiation)
    
    # Perform normality tests
    print(f"\n{'='*70}")
    print(f"Training Set Statistics (Days {START_DAY}-{END_DAY}, Years {TRAINING_YEARS[0]}-{TRAINING_YEARS[-1]})")
    print(f"{'='*70}")
    print(f"Total number of daily averages: {len(all_daily_temps)}")
    
    datasets = [
        ("Temperature", all_daily_temps, "°C"),
        ("Relative Humidity", all_daily_humidity, "%"),
        ("Solar Radiation", all_daily_radiation, "W/m²")
    ]
    
    normality_results = {}
    
    for name, data, unit in datasets:
        print(f"\n{name}:")
        print(f"  Range: {data.min():.1f} to {data.max():.1f} {unit}")
        print(f"  Mean:  {data.mean():.1f} {unit} ± {data.std():.1f} {unit}")
        
        # Normality tests
        results = test_normality(data, name)
        normality_results[name] = results
        
        print(f"  Normality Tests:")
        sw_stat, sw_p = results['shapiro']
        sw_result = "Normal" if sw_p > 0.05 else "Not Normal"
        print(f"    Shapiro-Wilk: W={sw_stat:.4f}, p={sw_p:.4f} → {sw_result} (α=0.05)")
        
        if 'dagostino' in results:
            da_stat, da_p = results['dagostino']
            da_result = "Normal" if da_p > 0.05 else "Not Normal"
            print(f"    D'Agostino-Pearson: K²={da_stat:.4f}, p={da_p:.4f} → {da_result} (α=0.05)")

    print(f"\n{'='*70}")
    
    # Create output directory
    output_dir = Path("outputs/figures/dissertation/mean_daily_dist")
    output_dir.mkdir(exist_ok=True)
    
    # Create individual figures for each variable
    print("\nGenerating individual figures...")
    
    plot_single_distribution(
        all_daily_temps,
        r"$\bar{d}_{\mathrm{T}}$ (°C)",
        "C0",
        "Temperature Distribution",
        "°C",
        normality_results["Temperature"],
        output_dir / "training_temperature_distribution"
    )
    print(f"  Saved: figures/GL-MPC-RL/generalist-training-dist/temperature_distribution.png and /.svg")
    
    plot_single_distribution(
        all_daily_humidity,
        r"$\bar{d}_{\mathrm{RH}}$ (%)",
        "C0",
        "Relative Humidity Distribution",
        "%",
        normality_results["Relative Humidity"],
        output_dir / "training_humidity_distribution"
    )
    print(f"  Saved: figures/GL-MPC-RL/generalist-training-dist/humidity_distribution.png and/.svg")
    
    plot_single_distribution(
        all_daily_radiation,
        r"$\bar{d}_{\mathrm{iGlob}}$ (W/m$^2$)",
        "C0",
        "Solar Radiation Distribution",
        "W/m$^2$",
        normality_results["Solar Radiation"],
        output_dir / "training_radiation_distribution"
    )
    print(f"  Saved: figures/GL-MPC-RL/generalist-training-dist/radiation_distribution.png and /.svg")
    
    # Create combined figure
    print("\nGenerating combined figure...")
    plot_combined_distribution(
        all_daily_temps,
        all_daily_humidity,
        all_daily_radiation,
        output_dir
    )
    print(f"  Saved: figures/GL-MPC-RL/generalist-training-dist/training_weather_distribution.png and /.svg")
    
    print("\nDone!")

if __name__ == "__main__":
    main()
