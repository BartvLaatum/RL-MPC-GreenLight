"""
Visualize average state and control trajectories across multiple years.

This script plots the mean trajectories for each controller type (RL-MPC, MPC, RL)
averaged over multiple years with the same start_day. The plot includes weather
variables and control actions in a 3x4 grid layout.

Usage:
    python visualizations/trajectory_plot.py
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import pandas as pd

import plot_config
from performance import load_rl_mpc_data, load_mpc_data, load_rl_data


def create_figure(y_labels: Dict[int, str], t: np.ndarray, boundaries: Dict[int, Optional[List]]) -> Tuple:
    """Create a 3x4 figure with proper labels and boundaries."""
    WIDTH = 173.8 * 0.03937
    HEIGHT = WIDTH * 0.75
    fig, axes = plt.subplots(3, 4, figsize=(WIDTH, HEIGHT), dpi=300, sharex=True, sharey=False)
    
    for i, ax in enumerate(axes.flat):
        ax.set_ylabel(y_labels[i])
        if boundaries[i] is not None:
            ax.hlines(boundaries[i], t[0], t[-1], color='grey', linestyle='--', 
                     linewidth=1.2, alpha=0.6, zorder=0)
    
    return fig, axes


def load_weather_data_from_rl(
    project: str,
    model_name: str,
    location: str,
    growth_year: int,
    start_day: int
) -> Optional[Dict[str, np.ndarray]]:
    """
    Load weather data from RL results CSV.
    
    Returns dictionary with outdoor temperature, CO2, humidity, and solar radiation.
    """
    try:
        # Load RL CSV which contains weather data
        rl_file = f"results/{project}/deterministic/ppo/{model_name}-{location}-{growth_year}-{start_day}.csv"
        rl_df = pd.read_csv(rl_file)
        
        # Extract weather variables from RL data
        weather_data = {
            "temp_out": rl_df["temp_out"].values,
            "co2_out": rl_df["co2_out"].values,
            "rh_out": rl_df["rh_out"].values,
            "solar_rad": rl_df["glob_rad"].values
        }
        return weather_data
    except (FileNotFoundError, KeyError) as e:
        print(f"Warning: Could not load weather data from {rl_file}: {e}")
        return None


def aggregate_trajectories_multi_year(
    project: str,
    ablations: List[str],
    model_names: List[str],
    location: str,
    growth_years: List[int],
    start_day: int,
    horizon: int = 1,
    mpc_solver: str = "linear_solver_ma57"
) -> Dict[str, Dict]:
    """
    Load and aggregate trajectories across multiple years for the same start_day.
    
    Returns:
        Dictionary with averaged trajectories and standard deviations for each controller type.
    """
    # Initialize storage for trajectories
    rlmpc_X_list, rlmpc_U_list, rlmpc_W_list = [], [], []
    mpc_X_list, mpc_U_list, mpc_W_list = [], [], []
    rl_X_list, rl_U_list, rl_W_list = [], [], []
    for year in growth_years:
        try:
            # Load RL-MPC data
            rlmpc_data = load_rl_mpc_data(project, ablations, location, year, start_day, horizons=[horizon])
            # The load_rl_mpc_data returns {"rlmpc": {ablation: {horizon: data}}}
            if "rlmpc" in rlmpc_data:
                for ablation in ablations:
                    if ablation in rlmpc_data["rlmpc"] and horizon in rlmpc_data["rlmpc"][ablation]:
                        rlmpc_X_list.append(rlmpc_data["rlmpc"][ablation][horizon]["X"])
                        rlmpc_U_list.append(rlmpc_data["rlmpc"][ablation][horizon]["U"])
            
            # Load MPC data
            mpc_data = load_mpc_data(project, mpc_solver, location, year, start_day)
            if horizon in mpc_data["mpc"]:
                mpc_X_list.append(mpc_data["mpc"][horizon]["X"])
                mpc_U_list.append(mpc_data["mpc"][horizon]["U"])
            
            # Load RL data
            rl_data = load_rl_data(project, model_names, location, year, start_day)

            for model_name in model_names:
                if model_name in rl_data["rl"]:
                    rl_X_list.append(rl_data["rl"][model_name]["X"])
                    rl_U_list.append(rl_data["rl"][model_name]["U"])
            
            # Load weather data from RL results
            weather_data = load_weather_data_from_rl(project, model_names[0], location, year, start_day)
            if weather_data:
                rlmpc_W_list.append(weather_data)
                mpc_W_list.append(weather_data)
                rl_W_list.append(weather_data)
                
        except Exception as e:
            print(f"Warning: Could not load data for year {year}: {e}")
            continue
        
    # Compute averages and standard deviations
    def compute_stats(data_list):
        if not data_list:
            return None, None
        # Find minimum length to handle potential size differences
        min_len = min(d.shape[-1] for d in data_list)
        # Truncate all to minimum length
        truncated = [d[..., :min_len] for d in data_list]
        stacked = np.stack(truncated, axis=0)
        mean = np.mean(stacked, axis=0)
        std = np.std(stacked, axis=0)
        return mean, std
    
    # Aggregate all data
    aggregated = {
        "rlmpc": {
            "X_mean": compute_stats(rlmpc_X_list)[0],
            "X_std": compute_stats(rlmpc_X_list)[1],
            "U_mean": compute_stats(rlmpc_U_list)[0],
            "U_std": compute_stats(rlmpc_U_list)[1],
            "n_years": len(rlmpc_X_list)
        },
        "mpc": {
            "X_mean": compute_stats(mpc_X_list)[0],
            "X_std": compute_stats(mpc_X_list)[1],
            "U_mean": compute_stats(mpc_U_list)[0],
            "U_std": compute_stats(mpc_U_list)[1],
            "n_years": len(mpc_X_list)
        },
        "rl": {
            "X_mean": compute_stats(rl_X_list)[0],
            "X_std": compute_stats(rl_X_list)[1],
            "U_mean": compute_stats(rl_U_list)[0],
            "U_std": compute_stats(rl_U_list)[1],
            "n_years": len(rl_X_list)
        }
    }
    
    # Average weather data
    if rlmpc_W_list:
        min_len = min(len(w["temp_out"]) for w in rlmpc_W_list)
        aggregated["weather"] = {
            "temp_out_mean": np.mean([w["temp_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "temp_out_std": np.std([w["temp_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "co2_out_mean": np.mean([w["co2_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "co2_out_std": np.std([w["co2_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "rh_out_mean": np.mean([w["rh_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "rh_out_std": np.std([w["rh_out"][:min_len] for w in rlmpc_W_list], axis=0),
            "solar_rad_mean": np.mean([w["solar_rad"][:min_len] for w in rlmpc_W_list], axis=0),
            "solar_rad_std": np.std([w["solar_rad"][:min_len] for w in rlmpc_W_list], axis=0)
        }
    
    return aggregated


def plot_averaged_trajectories(
    fig,
    axes,
    aggregated_data: Dict,
    dt: int,
    controller_type: str,
    label: str,
    color: str,
    linestyle: str = "-",
    plot_std: bool = False,
    alpha: float = 0.8
):
    """
    Plot averaged state and control trajectories for a single controller type.
    
    Args:
        fig: Matplotlib figure
        axes: 3x4 array of axes
        aggregated_data: Dictionary containing mean and std for states and controls
        dt: Time step in seconds
        controller_type: "rlmpc", "mpc", or "rl"
        label: Label for legend
        color: Line color
        linestyle: Line style
        plot_std: Whether to plot standard deviation bands
        alpha: Line transparency
    """
    data = aggregated_data[controller_type]
    X_mean = data["X_mean"]
    U_mean = data["U_mean"]
    X_std = data["X_std"]
    U_std = data["U_std"]
    
    if X_mean is None or U_mean is None:
        print(f"Warning: No data available for {controller_type}")
        return fig, axes
    
    t = np.arange(0, X_mean.shape[-1] * dt, dt) / 86400
    
    # Map state indices based on controller type
    if controller_type == "rl":
        # RL states: [co2_air, temp_air, rh_air, pipe_temp, cFruit]
        state_indices = {
            "fruit": 4,      # cFruit
            "temp_air": 1,   # temp_air (placeholder for weather)
            "co2_air": 0,    # co2_air (placeholder for weather)
            "rh_air": 2      # rh_air (placeholder for weather)
        }
    else:
        # MPC/RL-MPC states: Full state vector
        state_indices = {
            "fruit": 25,     # cFruit
            "temp_air": 2,   # temp_air (placeholder for weather)
            "co2_air": 0,    # co2_air (placeholder for weather)
            "rh_air": 15     # rh_air (placeholder for weather)
        }

    # Plot Row 0: Fruit weight and weather placeholders
    axes[0, 0].step(t, X_mean[state_indices["fruit"], :], color=color, 
                   linestyle=linestyle, label=label, where='post', alpha=alpha)

    axes[0, 1].step(t, X_mean[state_indices["temp_air"], :], color=color, 
                   linestyle=linestyle, label=label, where='post', alpha=alpha)
    axes[0, 2].step(t, X_mean[state_indices["co2_air"], :], color=color, 
                   linestyle=linestyle, label=label, where='post', alpha=alpha)
    axes[0, 3].step(t, X_mean[state_indices["rh_air"], :], color=color, 
                   linestyle=linestyle, label=label, where='post', alpha=alpha)

    if plot_std:
        axes[0, 0].fill_between(t, 
                               X_mean[state_indices["fruit"], :] - X_std[state_indices["fruit"], :],
                               X_mean[state_indices["fruit"], :] + X_std[state_indices["fruit"], :],
                               color=color, alpha=0.2, step='post')
        axes[0, 1].fill_between(t, X_mean[state_indices["temp_air"], :] - X_std[state_indices["temp_air"], :],
                               X_mean[state_indices["temp_air"], :] + X_std[state_indices["temp_air"], :],
                               color=color, alpha=0.2, step='post')
        axes[0, 2].fill_between(t, X_mean[state_indices["co2_air"], :] - X_std[state_indices["co2_air"], :],
                               X_mean[state_indices["co2_air"], :] + X_std[state_indices["co2_air"], :],
                               color=color, alpha=0.2, step='post')
        axes[0, 3].fill_between(t, X_mean[state_indices["rh_air"], :] - X_std[state_indices["rh_air"], :],
                               X_mean[state_indices["rh_air"], :] + X_std[state_indices["rh_air"], :],
                               color=color, alpha=0.2, step='post')


    # Weather data will be plotted separately (axes[0,1], axes[0,2], axes[0,3])
    # These are handled in plot_weather_data function
    
    # Plot Row 1: Control actions (heating, CO2, thermal screen)
    # Row 1, Column 0 is empty (for solar radiation - weather)
    axes[1, 1].step(t, U_mean[0, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    axes[1, 2].step(t, U_mean[1, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    axes[1, 3].step(t, U_mean[2, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    
    if plot_std:
        for idx, ax in [(0, axes[1, 1]), (1, axes[1, 2]), (2, axes[1, 3])]:
            ax.fill_between(t, U_mean[idx, :] - U_std[idx, :], 
                          U_mean[idx, :] + U_std[idx, :],
                          color=color, alpha=0.2, step='post')
    
    # Plot Row 2: Control actions (ventilation, lighting, blackout screen)
    axes[2, 1].step(t, U_mean[3, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    axes[2, 2].step(t, U_mean[4, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    axes[2, 3].step(t, U_mean[5, :], color=color, linestyle=linestyle, 
                   label=label, where='post', alpha=alpha)
    
    if plot_std:
        for idx, ax in [(3, axes[2, 1]), (4, axes[2, 2]), (5, axes[2, 3])]:
            ax.fill_between(t, U_mean[idx, :] - U_std[idx, :], 
                          U_mean[idx, :] + U_std[idx, :],
                          color=color, alpha=0.2, step='post')
    
    return fig, axes


def plot_weather_data(
    fig,
    axes,
    aggregated_data: Dict,
    dt: int,
    color: str = "black",
    linestyle: str = "-",
    plot_std: bool = False,
    alpha: float = 0.6
):
    """Plot averaged weather data (outdoor temperature, CO2, humidity, solar radiation)."""
    if "weather" not in aggregated_data:
        print("Warning: No weather data available")
        return fig, axes
    
    weather = aggregated_data["weather"]
    t = np.arange(0, len(weather["temp_out_mean"]) * dt, dt) / 86400
    
    # Plot outdoor temperature at position [0, 1]
    axes[0, 1].step(t, weather["temp_out_mean"], color=color, linestyle=linestyle, 
                   label="Outdoor", where='post', alpha=alpha, linewidth=1.0, zorder=0)
    # if plot_std:
    #     axes[0, 1].fill_between(t, 
    #                            weather["temp_out_mean"] - weather["temp_out_std"],
    #                            weather["temp_out_mean"] + weather["temp_out_std"],
    #                            color=color, alpha=0.85, step='post', zorder=0)
    
    # Plot outdoor CO2 at position [0, 2]
    axes[0, 2].step(t, weather["co2_out_mean"], color=color, linestyle=linestyle, 
                   label="Outdoor", where='post', alpha=alpha, linewidth=1.0, zorder=0)
    # if plot_std:
    #     axes[0, 2].fill_between(t, 
    #                            weather["co2_out_mean"] - weather["co2_out_std"],
    #                            weather["co2_out_mean"] + weather["co2_out_std"],
    #                            color=color, alpha=0.85, step='post', zorder=0)
    
    # Plot outdoor humidity at position [0, 3]
    axes[0, 3].step(t, weather["rh_out_mean"], color=color, linestyle=linestyle, 
                   label="Outdoor", where='post', alpha=alpha, linewidth=1.0, zorder=0)
    # if plot_std:
    #     axes[0, 3].fill_between(t, 
    #                            weather["rh_out_mean"] - weather["rh_out_std"],
    #                            weather["rh_out_mean"] + weather["rh_out_std"],
    #                            color=color, alpha=0.85, step='post', zorder=0)
    
    # Plot solar radiation at position [1, 0]
    axes[1, 0].step(t, weather["solar_rad_mean"], color=color, linestyle=linestyle, 
                   label="Solar Rad.", where='post', alpha=alpha, linewidth=1.0, zorder=0)
    # if plot_std:
    #     axes[1, 0].fill_between(t, 
    #                            weather["solar_rad_mean"] - weather["solar_rad_std"],
    #                            weather["solar_rad_mean"] + weather["solar_rad_std"],
    #                            color=color, alpha=0.85, step='post', zorder=0)
    
    return fig, axes

if __name__ == "__main__":
    # ==============================================================================
    # Configuration
    # ==============================================================================
    # 
    # This script creates a 3x4 grid of subplots showing:
    #   Row 0: [Fruit Weight] [Air Temp] [CO2 Conc] [Rel Humidity]
    #   Row 1: [Solar Rad] [u_heat] [u_CO2] [u_ThScr]
    #   Row 2: [EMPTY] [u_vent] [u_light] [u_BlScr]
    #
    # Weather data (outdoor temp, CO2, humidity, solar radiation) is plotted in gray
    # Controller trajectories (RL-MPC, MPC, RL) are averaged over multiple years
    # ==============================================================================
    
    # Axis labels for the 3x4 grid
    y_labels = {
        0: r"Fruit Weight (DM mg/m$^2$)",
        1: r"Air Temperature ($^\circ$C)",
        2: r"CO$_2$ concentration (ppm)",
        3: r"Relative Humidity (%)",
        4: r"Solar Radiation (W/m$^2$)",
        5: r"$u_{\mathrm{heat}}$",
        6: r"$u_{\mathrm{CO_2}}$",
        7: r"$u_{\mathrm{ThScr}}$",
        8: "",  # Empty position (bottom-left corner)
        9: r"$u_{\mathrm{vent}}$",
        10: r"$u_{\mathrm{light}}$",
        11: r"$u_{\mathrm{BlScr}}$",
    }
    
    # Constraint boundaries for each subplot
    boundaries = {
        0: None,           # Fruit weight - no boundaries
        1: [15, 25],       # Air temperature
        2: [400, 1600],    # CO2 concentration
        3: [50, 90],        # Relative humidity
        4: None,           # Solar radiation - no boundaries
        5: [0, 1],         # Heating control
        6: [0, 1],         # CO2 control
        7: [0, 1],         # Thermal screen control
        8: None,           # Empty
        9: [0, 1],         # Ventilation control
        10: [0, 1],        # Lighting control
        11: [0, 1],        # Blackout screen control
    }

    # Experiment configuration
    project = "GL-MPC-RL"
    model_name = "daily-glade-107"
    ablations = [f"rlmpc-switch-{model_name}-0.05"]
    location = "Netherlands"
    
    # Multiple years for averaging
    growth_years = list(range(2011, 2021))
    start_day = 151  # Same start day for all years
    
    dt = 300  # Time step in seconds
    horizon = 1
    mpc_solver = "mpc_linear_solver_ma57"
    
    # Plotting configuration
    colors = {
        "rl": "C0" ,     # Blue
        "mpc": "#00a693",    # Teal
        "rlmpc": "C3",  # Red
    }
    labels = {
        "rl": "RL",
        "mpc": "MPC",
        "rlmpc": "RL-MPC",
    }
    
    # ==============================================================================
    # Load and aggregate data across multiple years
    # ==============================================================================
    
    print(f"Loading data for {len(growth_years)} years: {growth_years}")
    print(f"Start day: {start_day}")
    
    aggregated_data = aggregate_trajectories_multi_year(
        project=project,
        ablations=ablations,
        model_names=[model_name],
        location=location,
        growth_years=growth_years,
        start_day=start_day,
        horizon=horizon,
        mpc_solver=mpc_solver
    )
    
    # Print summary
    print("\nData aggregation summary:")
    for controller_type in ["rl", "mpc", "rlmpc"]:
        n_years = aggregated_data[controller_type]["n_years"]
        print(f"  {labels[controller_type]}: {n_years} years loaded")
    
    # ==============================================================================
    # Create figure and plot trajectories
    # ==============================================================================
    
    # Get time vector from RL-MPC data
    if aggregated_data["rlmpc"]["X_mean"] is not None:
        n_steps = aggregated_data["rlmpc"]["X_mean"].shape[-1]
        t = np.arange(0, n_steps * dt, dt) / 86400
    else:
        t = np.arange(0, 100)  # Fallback
    
    # Create figure
    fig, axes = create_figure(y_labels, t, boundaries)
    
    # Plot weather data first (as background)
    # Set plot_std_bands=True to show standard deviation bands around trajectories
    plot_std_bands = False
    fig, axes = plot_weather_data(
        fig, axes, aggregated_data, dt,
        color="gray", linestyle="--",
        plot_std=plot_std_bands, alpha=0.5
    )
    
    # Plot controller trajectories
    for controller_type in ["rl", "mpc", "rlmpc"]:
        fig, axes = plot_averaged_trajectories(
            fig=fig,
            axes=axes,
            aggregated_data=aggregated_data,
            dt=dt,
            controller_type=controller_type,
            label=labels[controller_type],
            color=colors[controller_type],
            linestyle="-",
            plot_std=plot_std_bands,
            alpha=0.85
        )

    # ==============================================================================
    # Finalize and save
    # ==============================================================================
    
    # Add legend to the first subplot
    axes[0, 0].legend(loc="upper left", fontsize=7, framealpha=0.9)
    
    # Add x-label to bottom subplots
    for ax in axes[-1, :]:
        ax.set_xlabel("Time (days)")
    
    # Remove axis for empty subplot [2, 0]
    axes[2, 0].axis("off")
    
    # Adjust layout
    fig.tight_layout()
    
    # Save figures
    output_dir = "figures/GL-MPC-RL"
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig.savefig(f"{output_dir}/trajectory_plots_averaged.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{output_dir}/trajectory_plots_averaged.svg", bbox_inches="tight")
    
    print(f"\nFigures saved to {output_dir}/")
    print(f"  - trajectory_plots_averaged.png")
    print(f"  - trajectory_plots_averaged.svg")
    print(f"\nTo enable standard deviation bands, set plot_std_bands=True in the configuration section.")
