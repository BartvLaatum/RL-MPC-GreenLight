import pandas as pd
import numpy as np

import plot_config

method = "comparison"

matlab_states = pd.read_csv(f"results/{method}/june/matlab-states-300dt.csv")
py_states = pd.read_csv(f"results/{method}/june/states-300dt.csv", delimiter=",")
controls = pd.read_csv(f"results/{method}/june/control-inputs-with-time.csv", delimiter=",")
# cols = matlab_states.columns
# cols = cols.delete(18).append(cols[18:19])

# py_states = pd.DataFrame(py_states, columns=cols)
print(py_states.head())
import matplotlib.pyplot as plt

# Create a plot for all states and controls
n_cols = 4
all_vars = list(py_states.columns) + list(controls.columns)
n_rows = (len(all_vars) + n_cols - 1) // n_cols  # Ceiling division
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))

for idx, var in enumerate(all_vars):
    row = idx // n_cols
    col_idx = idx % n_cols
    ax = axes[row, col_idx]
    
    if var in py_states.columns:
        ax.plot(matlab_states[var], label='MATLAB')
        ax.plot(py_states[var], label='Python')
    else:
        ax.plot(controls[var], label='Control')
    ax.set_ylabel(var)
    ax.set_xlabel('Time (s)')
    ax.legend()

# Hide empty subplots if any
for idx in range(len(all_vars), n_rows * n_cols):
    row = idx // n_cols
    col_idx = idx % n_cols
    axes[row, col_idx].set_visible(False)

plt.tight_layout()
plt.savefig(f"figures/{method}/june/all-states-and-controls-300dt.png")
