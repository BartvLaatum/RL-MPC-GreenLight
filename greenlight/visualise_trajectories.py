import argparse

import pandas as pd
import matplotlib.pyplot as plt

### Latex font in plots
plt.rcParams['font.serif'] = "cmr10"
plt.rcParams['font.family'] = "serif"
plt.rcParams['font.size'] = 24

plt.rcParams['legend.fontsize'] = 24
plt.rcParams['legend.loc'] = 'upper right'
plt.rcParams['axes.labelsize'] = 22
plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['xtick.labelsize'] = 24
plt.rcParams['ytick.labelsize'] = 24
plt.rcParams['text.usetex'] = False
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["axes.grid"] = False
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams['axes.linewidth'] = 4   # Default for all spines
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
# plt.rcParams['text.usetex'] = True
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['xtick.major.size'] = 4  # Thicker major x-ticks
plt.rcParams['xtick.major.width'] = 2  # Thicker major x-
plt.rcParams['ytick.major.size'] = 4  
plt.rcParams['ytick.major.width'] = 2 
plt.rc('axes', unicode_minus=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--month', type=str, default='january', help='Month to visualise')
    args = parser.parse_args()
    month = args.month

    # Load state variable data
    path = f'data/{month}/'
    states_diff = pd.read_csv(path + 'states-OL.csv', header=None).T
    states_non_diff = pd.read_csv(path + 'states-OL-non-diff.csv', header=None).T

    # Select rows for comparison
    cols_to_compare = [0, 2, 15, 25]
    states_diff_selected = states_diff.iloc[cols_to_compare]
    states_non_diff_selected = states_non_diff.iloc[cols_to_compare]

    # Plot state variables
    time_steps = list(range(states_diff.shape[1]))
    fig, axes = plt.subplots(len(cols_to_compare), 1, figsize=(12, 10), sharex=True)

    state_labels_with_units = {
        0: "CO2",
        2: "Air Temp",
        15: "VPD",
        25: "Fruit Weight"
    }

    for i, row in enumerate(cols_to_compare):
        axes[i].plot(time_steps, states_diff_selected.loc[row], label="Diff", linestyle='--')
        axes[i].plot(time_steps, states_non_diff_selected.loc[row], label="Diff", linestyle='--')
        axes[i].set_ylabel(state_labels_with_units.get(row, f'State {row}'))
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Steps')
    # fig.suptitle('Comparison of Selected State Variables over Time')
    fig.legend(["Diff", "Non diff"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'figures/{month}/states.png')

    # Load control input data
    control_inputs = pd.read_csv(path + 'controls-OL.csv', header=None)
    control_inputs_non_diff = pd.read_csv(path + 'controls-OL-non-diff.csv', header=None)

    control_labels_with_units = {
        0: r"$u_{heat}$ (kW)",
        1: r"$u_{CO_2}$ (kg/hr)",
        2: r"$u_{ThScr}$ (%)",
        3: r"$u_{vent}$ (%)",
        4: r"$u_{light}$ (kW)",
        5: r"$u_{BlScr}$ (%)",
    }

    # Plot control inputs
    time_steps_control = list(range(control_inputs.shape[0]))
    fig, axes = plt.subplots(len(control_labels_with_units), 1, figsize=(12, 12), sharex=True)

    for i, (row, label) in enumerate(control_labels_with_units.items()):
        axes[i].plot(time_steps_control, control_inputs[row])
        axes[i].plot(time_steps_control, control_inputs_non_diff[row])
        axes[i].set_ylabel(label)
        axes[i].grid(True)
        axes[i].set_ylim(0, 1)  # Normalise control inputs

    fig.legend(["Diff", "Non diff"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))


    axes[-1].set_xlabel('Time Steps')
    # fig.suptitle('Control inputs over time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'figures/{month}/controls.png')


    # Load weather data
    weather_data = pd.read_csv(path + 'weather.csv', header=None)

    updated_weather_labels = {
        0: "iGlob",
        1: "Out temp",
        2: "Out VP",
        3: "Out CO2",
        4: "Out wind speed",
        5: "Sky temp",
        6: "Out soil temp "
    }

    # Plot weather variables
    time_steps_weather = list(range(weather_data.shape[0]))
    fig, axes = plt.subplots(len(updated_weather_labels), 1, figsize=(12, 14), sharex=True)

    for i, (col, label) in enumerate(updated_weather_labels.items()):
        axes[i].plot(time_steps_weather, weather_data[col])
        axes[i].set_ylabel(label)
        axes[i].grid(True)

    axes[-1].set_xlabel('Time Steps')
    # fig.suptitle('Weather Variables Over Time')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'figures/{month}/weahter_inputs.png')
