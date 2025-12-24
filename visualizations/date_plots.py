import os
import numpy as np
import pandas as pd

from performance import plot_rewards, plot_states, plot_control_trajectories

def load_rl_mpc_data(project, suffixes):
    data = {}
    BASE_DIR = os.path.join("results", project, "deterministic", "rlmpc")
    horizons = [1]

    data = {suffix: {} for suffix in suffixes}

    for suffix in suffixes:
        for H in horizons:
            U = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"control-inputs-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            X = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"states-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            times = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"times-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            rewards = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"rewards-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            EPI = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"EPI-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            penalties = np.loadtxt(os.path.join(BASE_DIR, "full-horizon-fixed-region-norm-x-0.01",  f"penalties-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            data[suffix][H] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }
            if suffix == "offline_rl_trajectory":
                offline_rl_x = np.loadtxt(os.path.join(BASE_DIR, suffix,  f"x-offline-rl.csv"), delimiter=",").T
                data[suffix][H]["rollout_x"] = offline_rl_x
    return data

def load_rl_data(project, model_name, suffixes):
    data = {
        "rl": {f"{model_name}-{suffix}": {} for suffix in suffixes}
    }
    BASE_DIR = os.path.join("results", project, "deterministic", "ppo")

    for suffix in suffixes:
        rl_data = pd.read_csv(os.path.join(BASE_DIR, f"{model_name}-{suffix}.csv"))
        
        U = rl_data[["uBoil", "uCo2", "uThScr", "uVent", "uLamp", "uBlScr"]].values.T
        X = rl_data[["co2_air", "temp_air", "rh_air", "pipe_temp", "cFruit"]].values.T
        EPI = rl_data["EPI"].values.T
        rewards = rl_data["Rewards"].values.T
        penalties = rl_data["Penalty"].values.T
        data["rl"][f"{model_name}-{suffix}"] = {
            "U": U,
            "X": X,
            "rewards": rewards,
            "EPI": EPI,
            "penalties": penalties
        }
    return data

def load_mpc_data(project, mpc_folder, suffixes):
    data = {suffix: {} for suffix in suffixes}
    BASE_DIR = os.path.join("results", project, "deterministic", "mpc", mpc_folder)
    horizons = [1]

    for H in horizons:
        for suffix in suffixes:
            U = np.loadtxt(os.path.join(BASE_DIR, f"control-inputs-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            X = np.loadtxt(os.path.join(BASE_DIR, f"states-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            times = np.loadtxt(os.path.join(BASE_DIR, f"times-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            rewards = np.loadtxt(os.path.join(BASE_DIR, f"rewards-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            EPI = np.loadtxt(os.path.join(BASE_DIR, f"EPI-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            penalties = np.loadtxt(os.path.join(BASE_DIR, f"penalties-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            data[suffix][H] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties
            }
    return data


if __name__ == "__main__":
    project = "GL-MPC-RL"
    suffixes = [
        "Netherlands-2023-151",
        "Netherlands-2023-152",
        "Netherlands-2020-151",
        ]
    model_name = "trim-durian-32"
    # rl_mpc_data = load_rl_mpc_data(project, suffixes)
    
    rl_data = load_rl_data(project, model_name, suffixes)
    # rl_mpc_data = load_rl_mpc_data(project, suffixes)
    # data = {**rl_data, **rl_mpc_data}
    data = {**rl_data}
    dt = 300
    labels = ["2023-06-01", "2023-06-02", "2020-06-01"]
    colors = ["C0", "C1", "C2", "C3", "C4", "C5"]
    linestyles = ["-", "-", "-", "--", "--", "--"]
    output_path = os.path.join("figures", project, f"{model_name}-date-plots")
    os.makedirs(output_path, exist_ok=True)

    plot_rewards(data, dt, labels, colors, linestyles, output_path=os.path.join(output_path, "rewards.png"), show=False)
    plot_states(data, dt, labels, colors, linestyles, output_path=os.path.join(output_path, "states.png"), show=False)
    plot_control_trajectories(data, dt, labels, colors, linestyles, output_path=os.path.join(output_path, "controls.png"), show=False)
