import os
import numpy as np
import pandas as pd

from performance import plot_rewards, plot_states, plot_control_trajectories
import plot_config

from environments.utils import load_weather_data

def load_rl_results(project, model_name, suffixes):
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


def load_rlmpc_results(project, suffixes, experiment_name, horizons):
    data = {}
    BASE_DIR = os.path.join("results", project, "deterministic", "rlmpc")

    data = {suffix: {} for suffix in suffixes}

    for suffix in suffixes:
        for H in horizons:
            U = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"control-inputs-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            X = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"states-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            times = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"times-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            rewards = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"rewards-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            EPI = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"EPI-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            penalties = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"penalties-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            rl_costs = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"rl_costs-300dt-{H}H-{suffix}.csv"), delimiter=",").T
            mpc_costs = np.loadtxt(os.path.join(BASE_DIR, experiment_name,  f"mpc_costs-300dt-{H}H-{suffix}.csv"), delimiter=",").T

            data[suffix][H] = {
                "U": U,
                "X": X,
                "times": times,
                "rewards": rewards,
                "EPI": EPI,
                "penalties": penalties,
                "rl_costs": np.sum(rl_costs, axis=0),
                "mpc_costs": np.sum(mpc_costs, axis=0),
                "rl_costs_trajectory": rl_costs,
                "mpc_costs_trajectory": mpc_costs,
                "chosen_is_mpc": np.where(np.sum(mpc_costs, axis=0) < np.sum(rl_costs, axis=0), 1, 0)
            }
    return data

def plot_solver_costs(data, dt, labels, colors, linestyles, output_path=None, show=False, N_to_plot=None):
    import matplotlib.pyplot as plt
    t = np.arange(0, data["rl_costs"].shape[-1] * dt, dt) / 86400
    plt.plot(t, data["rl_costs"], "o-", label="RL", color=colors[0])
    plt.plot(t, data["mpc_costs"], "o-", label="RL-MPC", color=colors[1])
    plt.legend()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved solver costs plot to {output_path}")

def plot_solver_costs(data, dt, labels, colors, linestyles, output_path=None, show=False, N_to_plot=None):
    import matplotlib.pyplot as plt
    WIDTH = 130 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    # HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    t = np.arange(0, data["rl_costs"].shape[-1] * dt, dt) / 86400
    ax.plot(t, data["rl_costs"], "o-", label="RL", color=colors[0], alpha=0.8)
    ax.plot(t, data["mpc_costs"], "o-", label="RL-MPC", color=colors[1], alpha=0.8)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel(r'Solver Cost ($\mathcal{J}$)')
    # ax.legend()
    ax.set_ylim(-0.1, 0.1)
    ax.yaxis.set_major_locator(plt.LinearLocator(3))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))

    fig.tight_layout()
    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
    
    print(f"Saved solver costs plot to {output_path}")

def relative_solver_costs(data, dt, labels, colors, linestyles, output_path=None, show=False, N_to_plot=None):
    import matplotlib.pyplot as plt
    WIDTH = 75 * 0.0393700787
    HEIGHT = WIDTH * 0.75
    color_counter  = 0
    fig, ax = plt.subplots(figsize=(WIDTH, HEIGHT), dpi=300)

    t = np.arange(0, data["rl_costs"].shape[-1] * dt, dt) / 86400
    relative_costs = (data["rl_costs"] - data["mpc_costs"])/np.abs(np.min(data["rl_costs"] - data["mpc_costs"]))
    # / np.minimum(data["rl_costs"], data["mpc_costs"])
    # Color the dots based on sign: blue for negative, red for positive
    colors_sign = np.where(relative_costs < 0, 'C0', 'C3')
    # ax.plot(t, relative_costs, "-", color="grey", alpha=0.5)  # thin lines for context
    ax.scatter(t, relative_costs, c=colors_sign, marker="o", alpha=0.8)
    ax.plot(t, np.zeros_like(t), "--", color="black", alpha=0.5)
    ax.set_xlabel('Time (days)')
    ax.set_ylabel('Relative cost')
    ax.set_ylim(-0.01, 0.01)
    ax.yaxis.set_major_locator(plt.LinearLocator(3))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter('%.3f'))
    fig.tight_layout()

    fig.savefig(output_path)
    fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
    print(f"Saved solver costs plot to {output_path}")


import numpy as np
import matplotlib.pyplot as plt


def compute_relative_costs(data):
    return (data["rl_costs"] - data["mpc_costs"])/np.abs(np.min(data["rl_costs"] - data["mpc_costs"]))

def compute_horizon_heatmap_matrix(ell_A, ell_B):
    if ell_A.ndim != 2 or ell_B.ndim != 2:
        raise ValueError("ell_A and ell_B must be 2D arrays of shape (T, H).")

    T_A, H_A = ell_A.shape
    T_B, H_B = ell_B.shape
    if T_A != T_B:
        raise ValueError(f"Time dimension mismatch: ell_A has T={T_A}, ell_B has T={T_B}.")
    T = T_A

    H_min = min(H_A, H_B)
    has_terminal_mismatch = (H_A != H_B)

    # Shared per-lookahead differences: D(t,k) = ell_B - ell_A
    D_shared = np.cumsum(ell_B[:, :H_min], axis=1) - np.cumsum(ell_A[:, :H_min], axis=1)  # (T, H_min)

    # Optional terminal-only row (if one controller has an extra step)
    D_terminal = None
    terminal_label = None
    if has_terminal_mismatch:
        if abs(H_A - H_B) != 1:
            raise ValueError(
                f"Expected horizon lengths to match or differ by 1, got H_A={H_A}, H_B={H_B}."
            )
        if H_B == H_min + 1:
            # B has an extra terminal step: contribution to (ell_B - ell_A) is +ell_B_extra
            D_terminal = np.sum(ell_B[:, :], axis=1)  # (T,)
            # terminal_label = f"terminal ({label_B})" if terminal_owner in (None, "B") else "terminal (extra)"
        else:
            # A has an extra terminal step: contribution is ell_A_extra
            D_terminal = np.sum(ell_B[:, :H_min], axis=1) - np.sum(ell_A[:, :], axis=1)  # (T,)
            # terminal_label = f"$V_f(x, x_f)$ ({label_A})" if terminal_owner in (None, "A") else "terminal (extra)"

    # Build the heatmap matrix with rows = lookahead index, cols = time
    # We'll add a top "choice" strip row if chosen_is_A is provided.
    rows = []
    row_labels = []

    # Shared k rows
    # We store as (H_min, T) for imshow
    for kk in range(H_min):
        rows.append(D_shared[:, kk])
        row_labels.append(f"k={kk}")

    # Terminal-only row at the end (if mismatch)
    if D_terminal is not None:
        rows.append(D_terminal)
        row_labels.append(terminal_label)

    M = np.vstack(rows)  # (num_rows, T)
    return M

def plot_panel_c_horizon_heatmap(
    ell_A: np.ndarray,
    ell_B: np.ndarray,
    chosen_is_A: np.ndarray | None = None,
    *,
    label_A: str = "RL-MPC",
    label_B: str = "RL",
    terminal_owner: str | None = None,  # "A" or "B" if one controller has an extra terminal step
    show_kstar: bool = True,
    title: str = "Panel C — Horizon decomposition heatmap",
    output_path: str = None,
):
    """
    Panel C: Horizon decomposition heatmap for two MPC-style controllers.

    Inputs
    ------
    ell_A : array (T, H_A)  stage costs along controller A's planned rollout at each time t
    ell_B : array (T, H_B)  stage costs along controller B's planned rollout at each time t
        Note: H_A and H_B may differ by 1 because of a terminal cost step.

    chosen_is_A : bool array (T,), optional
        If provided, a 'choice' strip is shown at the top.

    terminal_owner : {"A","B",None}
        If one controller includes an extra terminal cost (horizon length differs by 1),
        set terminal_owner accordingly. If None, the function still works, but the
        terminal-only row will be labeled generically.

    What gets plotted
    -----------------
    For shared lookahead indices k in [0, min(H_A,H_B)-1]:
        D(t,k) = ell_B(t,k) - ell_A(t,k)
    If one controller has an extra last step, an additional heatmap row is added:
        D_terminal(t) = (+ ell_B_extra) if.toggle else (- ell_A_extra)
    i.e. contribution to (ell_B - ell_A).

    This lets you see both the per-k differences and the terminal-only effect.
    """
    if ell_A.ndim != 2 or ell_B.ndim != 2:
        raise ValueError("ell_A and ell_B must be 2D arrays of shape (T, H).")

    T_A, H_A = ell_A.shape
    T_B, H_B = ell_B.shape
    if T_A != T_B:
        raise ValueError(f"Time dimension mismatch: ell_A has T={T_A}, ell_B has T={T_B}.")
    T = T_A

    if chosen_is_A is not None:
        chosen_is_A = np.asarray(chosen_is_A, dtype=bool)
        if chosen_is_A.shape != (T,):
            raise ValueError("chosen_is_A must have shape (T,).")

    H_min = min(H_A, H_B)
    has_terminal_mismatch = (H_A != H_B)

    # Shared per-lookahead differences: D(t,k) = ell_B - ell_A
    D_shared = np.cumsum(ell_B[:, :H_min], axis=1) - np.cumsum(ell_A[:, :H_min], axis=1)  # (T, H_min)

    # Optional terminal-only row (if one controller has an extra step)
    D_terminal = None
    terminal_label = None
    if has_terminal_mismatch:
        if abs(H_A - H_B) != 1:
            raise ValueError(
                f"Expected horizon lengths to match or differ by 1, got H_A={H_A}, H_B={H_B}."
            )
        if H_B == H_min + 1:
            # B has an extra terminal step: contribution to (ell_B - ell_A) is +ell_B_extra
            D_terminal = np.sum(ell_B[:, :], axis=1)  # (T,)
            terminal_label = f"terminal ({label_B})" if terminal_owner in (None, "B") else "terminal (extra)"
        else:
            # A has an extra terminal step: contribution is ell_A_extra
            D_terminal = np.sum(ell_B[:, :H_min], axis=1) - np.sum(ell_A[:, :], axis=1)  # (T,)
            terminal_label = f"$V_f(x, x_f)$ ({label_A})" if terminal_owner in (None, "A") else "terminal (extra)"

    # Build the heatmap matrix with rows = lookahead index, cols = time
    # We'll add a top "choice" strip row if chosen_is_A is provided.
    rows = []
    row_labels = []

    # Choice strip row (scaled later to match D color range)
    if chosen_is_A is not None:
        # We'll fill after computing the scale, for consistent visibility
        rows.append(np.zeros(T))
        row_labels.append("choice")

    # Shared k rows
    # We store as (H_min, T) for imshow
    for kk in range(H_min):
        rows.append(D_shared[:, kk])
        row_labels.append(f"k={kk}")

    # Terminal-only row at the end (if mismatch)
    if D_terminal is not None:
        rows.append(D_terminal)
        row_labels.append(terminal_label)

    M = np.vstack(rows)  # (num_rows, T)

    # Scale the choice strip to be visually obvious but not dominate the color scale.
    # We'll set it to +/- max_abs_D (or a small fallback) so it shows as strong colors.
    max_abs_D = float(np.max(np.abs(M[1:] if chosen_is_A is not None else M))) if M.size else 1.0
    max_abs_D = max(max_abs_D, 1e-6)

    if chosen_is_A is not None:
        choice_row = np.where(chosen_is_A, 1.0, -1.0) * max_abs_D
        M[0, :] = choice_row

    # Optional k* line: where advantage magnitude concentrates along the *shared* horizon (+ terminal row if present)
    kstar = None
    if show_kstar:
        # Use all rows except the choice strip for kstar computation
        start_row = 1 if chosen_is_A is not None else 0
        D_for_kstar = M[start_row:, :]  # (K_rows, T)
        absD = np.abs(D_for_kstar).T  # (T, K_rows)
        kk_idx = np.arange(D_for_kstar.shape[0], dtype=float)  # 0..K_rows-1
        denom = absD.sum(axis=1) + 1e-12
        kstar = (absD @ kk_idx) / denom  # (T,)

        # Map to imshow row coordinates
        # If there's a choice strip, add 1. Also kstar index is relative to start_row.
        kstar_plot = (start_row + kstar)
    else:
        kstar_plot = None

    # Plot
    fig , ax = plt.subplots(figsize=(12, 5))
    # Set colorbar limits to symmetric about zero so center is always 0
    vmax = np.max(np.abs(M))
    im = ax.imshow(M, aspect="auto", origin="upper", cmap="coolwarm", vmin=-0.01, vmax=0.01)
    # To render multi-character subscripts in matplotlib, wrap them in \mathrm{} within the subscript block.
    plt.colorbar(im, label=rf"$D(t,k)=\ell_{{\mathrm{{{label_B}}}}}(t,k)-\ell_{{\mathrm{{{label_A}}}}}(t,k)$ (positive ⇒ {label_A} better)")

    ax.set_xlabel("time step t")
    ax.set_ylabel("lookahead index")
    ax.set_title(title)

    # Ticks: don't label every k if it's large
    num_rows = M.shape[0]
    if num_rows <= 25:
        yticks = list(range(num_rows))
        yticklabels = row_labels
    else:
        # show a few meaningful ticks
        yticks = [0] if chosen_is_A is not None else []
        yticklabels = ["choice"] if chosen_is_A is not None else []
        # show k=0, mid, last shared, terminal
        first_k_row = (1 if chosen_is_A is not None else 0)
        mid_k_row = first_k_row + H_min // 2
        last_shared_row = first_k_row + H_min - 1
        yticks += [first_k_row, mid_k_row, last_shared_row]
        yticklabels += ["k=0", f"k={H_min//2}", f"k={H_min-1}"]
        if D_terminal is not None:
            yticks += [num_rows - 1]
            yticklabels += ["terminal"]
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)

    # Overlay k* if requested
    if kstar_plot is not None:
        t = np.arange(T)
        ax.scatter(t, kstar_plot, linewidth=2, c="grey", alpha=0.8, label=r"$k^*(t)$ advantage center-of-mass")
        plt.legend(loc="upper right")
    # Do not display grid on heatmap
    ax.grid(False)
    fig.tight_layout()
    plt.savefig(output_path)
    print(f"Saved panel C heatmap plot to {output_path}")


def plot_B_over_C(
    r,                      # (T,) relative margin series
    M,                      # (rows, T) heatmap matrix for Panel C (already includes choice row if you want)
    *,
    cmap="coolwarm",              # optional
    cbar_label=r"$\Delta \hat{J}$",
    ylabel_B=r"$\Delta \hat{J}(t)$",
    ylabel_C="prediction step",
    xlabel="time step t",
    label_A="RL-MPC",
    label_B="RL",
    daytime_indices=None,
    output_path=None,
):
    T = len(r)
    if M.shape[1] != T:
        raise ValueError("M must have the same number of columns as len(r) (time steps).")
    WIDTH = 130 * 0.03937007874
    # Dynamically set the height of the figure so that Panel C (bottom) grows with M.shape[0]
    base_height_unit = 0.2   # you can adjust this factor for preferred scaling
    shared_panel_ratio = 2   # default height for Panel B (top)
    panelC_rows = M.shape[0]
    # The height ratio for Panel C grows linearly with number of rows, but always >= 3 for backward compatibility
    panelC_ratio = max(3, int(panelC_rows * base_height_unit))
    
    HEIGHT = WIDTH * (shared_panel_ratio + panelC_ratio) / 5  # scale total height proportional to total ratios
    fig, (axB, axC) = plt.subplots(
        2, 1, sharex=True,
        figsize=(WIDTH, HEIGHT),
        gridspec_kw={"height_ratios": [shared_panel_ratio, panelC_ratio]},
        constrained_layout=True,
    )
    axB.grid(False)
    axC.grid(False)
    # ---- Panel B (top) ----
    t = np.arange(0, T * dt, dt) / 86400
    # Color the dots based on sign: blue for negative, red for positive
    colors_sign = np.where(r < 0, 'C0', 'C3')

    # ax.plot(t, relative_costs, "-", color="grey", alpha=0.5)  # thin lines for context
    vmax = 0.01

    x_start, x_end = None, None
    if daytime_indices is not None:
        idx_arr = daytime_indices[0] if isinstance(daytime_indices, tuple) else np.asarray(daytime_indices)
        start_index = int(idx_arr[0])
        end_index = int(idx_arr[-1])
        # Clip to valid range
        start_index = max(0, min(start_index, len(t) - 1))
        end_index = max(0, min(end_index, len(t) - 1))
        if start_index <= end_index:
            x_start, x_end = t[start_index], t[end_index]
            daytime_color = "#DDEAFD"  # warm amber/gold for daytime
            axB.axvspan(x_start, x_end, alpha=0.8, color=daytime_color, zorder=0)
            axC.axvspan(x_start, x_end, alpha=0.8, color=daytime_color, zorder=0)
            axB.axvline(x_start, color="black", linestyle="--", alpha=0.4)
            axB.axvline(x_end, color="black", linestyle="--", alpha=0.4)

    axB.scatter(t, r, c=colors_sign, marker="o", s=10, alpha=0.8)
    axB.plot(t, np.zeros_like(t), "--", color="black", alpha=0.6)

    axB.set_ylim(-vmax, vmax)
    if x_start is not None and x_end is not None:
        axB.text(
            (x_start + x_end) / 2, vmax * 0.9,
            "Daytime", ha="center", va="top",
            color="black"
        )
    axB.set_xticks(np.linspace(0, 1, 3))
    axB.tick_params(labelbottom=False)  # hide x tick labels on top panel
    axB.yaxis.set_major_locator(plt.LinearLocator(3))
    axB.yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
    axB.set_ylabel(ylabel_B)

    # ---- Panel C (bottom) ----
    im = axC.imshow(
        M,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=[t[0], 1, M.shape[0], 0],
    )
    # Set the y-ticks so that the bottom row is labeled "terminal value function"
    num_rows = M.shape[0]
    # Only mark the first, middle, and last ytick
    indices = [0]
    if num_rows > 2:
        indices.append(num_rows // 2)
    if num_rows > 1:
        indices.append(num_rows - 1)
    centers = np.array(indices) + 0.5

    yticklabels = [str(i) for i in range(num_rows)]
    # if len(yticklabels) > 0:
        # yticklabels[-1] = r"$V_f$"
    shown_labels = [yticklabels[i] for i in indices]

    axC.set_yticks(centers)
    axC.set_yticklabels(shown_labels)
    # To render multi-character subscripts in matplotlib, wrap them in \mathrm{} within the subscript block.

    axC.set_xlabel("Time (days)")
    axC.set_ylabel("Prediction step ($k$)")
    # Instead of using tight_layout() after creating a colorbar (which is incompatible with the new layout engine),
    # use the recommended 'constrained_layout=True' in plt.subplots (already set above) and do not call tight_layout here.
    cbar = fig.colorbar(im, ax=axC, label=cbar_label, pad=0.01, orientation="vertical", location="right")
    cbar_ticks = np.linspace(-vmax, vmax, 3)
    cbar.set_ticks(cbar_ticks)
    cbar.set_ticklabels([f"{tick:.2f}" for tick in cbar_ticks])

    # fig.tight_layout()    
    if output_path is not None:
        fig.savefig(output_path)
        fig.savefig(output_path.replace(".png", ".svg"), format="svg", dpi=300)
        print(f"Saved panel B over C plot to {output_path}")
    return fig, (axB, axC)


def get_daytime_from_weather_data(weather_data):
    """
    Get the daytime from the weather data.
    """
    daytime = weather_data[:, 8]
    return daytime



if __name__ == "__main__":
    project = "GL-MPC-RL"
    suffixes = [
        # "Netherlands-2012-151",
        # "Netherlands-2020-151",
        "Netherlands-2023-151",
        ]
    weather_data = load_weather_data(
        weather_data_dir="weather",
        location="Netherlands",
        growth_year=2023,
        start_day=151,
        n_days=1,
        pred_horizon=0,
        h=300,
        nd=10
    )
    daytime_indices = np.where(weather_data[:, 0] > 5)
    horizon = 1
    model_name = "daily-glade-107"
    experiment_name = f"rlmpc-switch-{model_name}-0.05"
    rl_mpc_data = load_rlmpc_results(project, suffixes, experiment_name, horizons=[horizon])

    dt = 300
    labels = ["RL", "RL-MPC"]
    colors = ["C0", "C3"]
    linestyles = ["-", "--"]
    output_path = os.path.join("figures", project, f"{experiment_name}-solver-costs-2023")
    os.makedirs(output_path, exist_ok=True)
    # Get the length of the rl_costs array for the first suffix and horizon 1
    first_suffix = suffixes[0]
    rl_costs_array = rl_mpc_data[first_suffix][horizon]["rl_costs"]
    rl_costs_length = len(rl_costs_array)

    plot_solver_costs(rl_mpc_data[first_suffix][horizon], dt, labels, colors, linestyles, output_path=os.path.join(output_path, f"solver_costs-{horizon}H.png"), show=False, N_to_plot=rl_costs_length)
    relative_solver_costs(rl_mpc_data[first_suffix][horizon], dt, labels, colors, linestyles, output_path=os.path.join(output_path, f"relative_solver_costs-{horizon}H.png"), show=False, N_to_plot=rl_costs_length)
    # plot_panel_c_horizon_heatmap(
    #     rl_mpc_data[first_suffix][1]["mpc_costs_trajectory"][:,:].T,
    #     rl_mpc_data[first_suffix][1]["rl_costs_trajectory"][:,:].T,
    #     output_path=os.path.join(output_path, "panel_c_horizon_heatmap.png")
    # )

    relative_costs = \
        (rl_mpc_data[first_suffix][horizon]["rl_costs"] - rl_mpc_data[first_suffix][horizon]["mpc_costs"])
            # / np.max(np.abs(rl_mpc_data[first_suffix][horizon]["rl_costs"] - rl_mpc_data[first_suffix][horizon]["mpc_costs"]))
    M = compute_horizon_heatmap_matrix(
        rl_mpc_data[first_suffix][horizon]["mpc_costs_trajectory"][:,:].T,
        rl_mpc_data[first_suffix][horizon]["rl_costs_trajectory"][:,:].T
    )
    plot_B_over_C(
        relative_costs,
        M,
        daytime_indices=daytime_indices,
        output_path=os.path.join(output_path, f"panel_b_over_c-{horizon}H.png")
        )    
