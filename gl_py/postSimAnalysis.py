# Post Sim Analysis 
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt

month = "june"
method = "finite-differences"

# load weather data
d = np.loadtxt('weather/june/weather-300dt.csv',delimiter=',',skiprows=0)
# load state data
x_300_12  = np.loadtxt(f'results/{method}/{month}/states-300.0dt-12Np.csv',delimiter=',',skiprows=0)
x_300_72  = np.loadtxt(f'results/{method}/{month}/states-300.0dt-72Np.csv',delimiter=',',skiprows=0)
x_300_144 = np.loadtxt(f'results/{method}/{month}/states-300.0dt-144Np.csv',delimiter=',',skiprows=0)
x_300_288 = np.loadtxt(f'results/{method}/{month}/states-300.0dt-288Np.csv',delimiter=',',skiprows=0)
# load control input data
u_300_12  = np.loadtxt(f'results/{method}/{month}/control-inputs-300.0dt-12Np.csv',delimiter=',',skiprows=0)
u_300_72  = np.loadtxt(f'results/{method}/{month}/control-inputs-300.0dt-72Np.csv',delimiter=',',skiprows=0)
u_300_144 = np.loadtxt(f'results/{method}/{month}/control-inputs-300.0dt-144Np.csv',delimiter=',',skiprows=0)
u_300_288 = np.loadtxt(f'results/{method}/{month}/control-inputs-300.0dt-288Np.csv',delimiter=',',skiprows=0)
# load timing data
tmean_300_12  = np.mean(np.loadtxt(f'results/{method}/{month}/exectime-300.0dt-12Np.csv',delimiter=',',skiprows=0))
tmean_300_72  = np.mean(np.loadtxt(f'results/{method}/{month}/exectime-300.0dt-72Np.csv',delimiter=',',skiprows=0))
tmean_300_144 = np.mean(np.loadtxt(f'results/{method}/{month}/exectime-300.0dt-144Np.csv',delimiter=',',skiprows=0))
tmean_300_288 = np.mean(np.loadtxt(f'results/{method}/{month}/exectime-300.0dt-288Np.csv',delimiter=',',skiprows=0))


# Plot Control Inputs
# Create a time vector (assuming time steps are sequential)
time = np.arange(289)*300/86400  
# Define labels for datasets
datasets = {
    "Np=12": u_300_12,
    "Np=72": u_300_72,
    "Np=144": u_300_144,
    "Np=288": u_300_288
}
control_labels_with_units = {
    0: r"$u_{heat}$",
    1: r"$u_{CO_2}$",
    2: r"$u_{ThScr}$",
    3: r"$u_{vent}$",
    4: r"$u_{light}$",
    5: r"$u_{BlScr}$",
}
# Plot each control input variable (columns 0-5) in a single figure
plt.figure(figsize=(10, 6))

for i in range(6):  # Loop over 6 control input variables
    plt.subplot(3, 2, i+1)  # 3 rows, 2 columns    
    for label, data in datasets.items():
        plt.plot(time, data[:, i], label=label)  
    plt.xlabel("Day")
    plt.ylabel(control_labels_with_units[i])    
    plt.grid(True)
    plt.ylim(-0.1, 1)    
    
plt.legend(loc="best", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.tight_layout()
plt.show()

# Plot States
datasets = {
    "Np=12": x_300_12,
    "Np=72": x_300_72,
    "Np=144": x_300_144,
    "Np=288": x_300_288
}
# Select only columns 0, 2, and 4 (Python uses 0-based indexing)
selected_columns = [2, 0, 15, 25]
state_labels_with_units = {
    0: r"Air Temperature ($^\circ$C)",
    1: r"CO$_2$ (mg/m$^3$)",
    2: r"Vapor Pressure (Pa)",
    3: r"Fruit Weight (DM mg/m$^2$)"
}
for idx, col in enumerate(selected_columns):  # Loop over selected columns
    plt.subplot(2, 2, idx+1) 
    for label, data in datasets.items():
        plt.plot(time, data[:, col], label=label)  

    plt.xlabel("Time step")
    plt.ylabel(state_labels_with_units[idx])  # Use custom labels
    plt.grid(True)
    
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=10)
plt.tight_layout()
plt.show()

# Print Mean Execution Times
# Example data (Replace with actual execution times)
experiments = ["fin-diff dt300 Np12", "fin-diff dt300 Np72", "fin-diff dt300 Np144", "fin-diff dt300 Np288"]
mean_times = [tmean_300_12, tmean_300_72, tmean_300_144, tmean_300_288]  
# Create a table
table = zip(experiments, mean_times)
# Print the table with headers
print(tabulate(table, headers=["Experiment", "Mean Execution Time (s)"], tablefmt="grid"))

# Print Costs & Benefits
heat_cost_300_12 = 0.09 * 44*144 * 300/3600 * 1e-3 * np.sum(u_300_12[:,0])
heat_cost_300_72 = 0.09 * 44*144 * 300/3600 * 1e-3 * np.sum(u_300_72[:,0])
heat_cost_300_144 = 0.09 * 44*144 * 300/3600 * 1e-3 * np.sum(u_300_144[:,0])
heat_cost_300_288 = 0.09 * 44*144 * 300/3600 * 1e-3 * np.sum(u_300_288[:,0])

lamp_cost_300_12 = 0.09 * 116*144 * 300/3600 * 1e-3 * np.sum(u_300_12[:,4])
lamp_cost_300_72 = 0.09 * 116*144 * 300/3600 * 1e-3 * np.sum(u_300_72[:,4])
lamp_cost_300_144 = 0.09 * 116*144 * 300/3600 * 1e-3 * np.sum(u_300_144[:,4])
lamp_cost_300_288 = 0.09 * 116*144 * 300/3600 * 1e-3 * np.sum(u_300_288[:,4])

co2_cost_300_12 = 0.1 * 144 * 300 * 1e-6 * 720 * np.sum(u_300_12[:,1])
co2_cost_300_72 = 0.1 * 144 * 300 * 1e-6 * 720 * np.sum(u_300_72[:,1])
co2_cost_300_144 = 0.1 * 144 * 300 * 1e-6 * 720 * np.sum(u_300_144[:,1])
co2_cost_300_288 = 0.1 * 144 * 300 * 1e-6 * 720 * np.sum(u_300_288[:,1])

yield_ben_300_12 =   1e-6 * 144 * 1.2 * (x_300_12[-1, 25] - x_300_12[0, 25])/0.06
yield_ben_300_72 =   1e-6 * 144 * 1.2 * (x_300_72[-1, 25] - x_300_72[0, 25])/0.06
yield_ben_300_144 =  1e-6 * 144 * 1.2 * (x_300_144[-1, 25] - x_300_144[0, 25])/0.06 
yield_ben_300_288 =  1e-6 * 144 * 1.2 * (x_300_288[-1, 25] - x_300_288[0, 25])/0.06 

experiments = ["fin-diff dt300 Np12", "fin-diff dt300 Np72", "fin-diff dt300 Np144", "fin-diff dt300 Np288"]
heat_costs = [heat_cost_300_12, heat_cost_300_72, heat_cost_300_144, heat_cost_300_288]  
lamp_costs = [lamp_cost_300_12, lamp_cost_300_72, lamp_cost_300_144, lamp_cost_300_288]  
co2_costs = [co2_cost_300_12, co2_cost_300_72, co2_cost_300_144, co2_cost_300_288]  
yield_benefit = [yield_ben_300_12, yield_ben_300_72, yield_ben_300_144, yield_ben_300_288]
# Create a table
table = zip(experiments, heat_costs, lamp_costs, co2_costs, yield_benefit, (np.array(yield_benefit) - np.array(heat_costs) - np.array(lamp_costs) - np.array(co2_costs)).tolist())
# Print the table with headers
print(tabulate(table, headers=["Experiment", "Heat Cost (\euro)", "Lamp Cost (\euro)", "CO2 Cost (\euro)", "Yield Benefit (\euro)", "Yield - Costs (\euro)"], tablefmt="grid"))
