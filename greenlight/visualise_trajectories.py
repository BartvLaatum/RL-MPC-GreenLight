import pandas as pd
import matplotlib.pyplot as plt

month = 'june'

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
    0: "CO2 (mg/m³)",
    2: "Air Temperature (°C)",
    15: "VP (kg/m²)",
    25: "Fruit Weight (mg{DW}/m²)"
}

for i, row in enumerate(cols_to_compare):
    axes[i].plot(time_steps, states_diff_selected.loc[row], label="Diff", linestyle='--')
    axes[i].plot(time_steps, states_non_diff_selected.loc[row], label="Diff", linestyle='--')
    axes[i].set_ylabel(state_labels_with_units.get(row, f'State {row}'))
    axes[i].grid(True)

axes[-1].set_xlabel('Time Steps')
fig.suptitle('Comparison of Selected State Variables over Time')
fig.legend(["Diff", "Non diff"], loc='upper center', ncol=2, bbox_to_anchor=(0.5, 0.98))

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'figures/{month}/states.png')

# Load control input data
control_inputs = pd.read_csv(path + 'controls-OL.csv', header=None)
control_inputs_non_diff = pd.read_csv(path + 'controls-OL-non-diff.csv', header=None)

control_labels_with_units = {
    0: "Heat Input (kW)",
    1: "CO2 Injection (kg/hr)",
    2: "Thermal Screen (%)",
    3: "Ventilation (%)",
    4: "Lighting (kW)",
    5: "Blackout Screen (%)"
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
fig.suptitle('Control inputs over time')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'figures/{month}/controls.png')


# Load weather data
weather_data = pd.read_csv(path + 'weather.csv', header=None)

updated_weather_labels = {
    0: "iGlob [W/m²]",
    1: "Out temp [°C]",
    2: "Out VP [Pa]",
    3: "Out CO2 [mg/m³]",
    4: "Out wind speed [m/s]",
    5: "Sky temp [°C]",
    6: "Out soil temp [°C]"
}

# Plot weather variables
time_steps_weather = list(range(weather_data.shape[0]))
fig, axes = plt.subplots(len(updated_weather_labels), 1, figsize=(12, 14), sharex=True)

for i, (col, label) in enumerate(updated_weather_labels.items()):
    axes[i].plot(time_steps_weather, weather_data[col])
    axes[i].set_ylabel(label)
    axes[i].grid(True)

axes[-1].set_xlabel('Time Steps')
fig.suptitle('Weather Variables Over Time')

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(f'figures/{month}/weahter_inputs.png')
