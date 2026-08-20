"""Compare RL-MPC performance with dynamic and fixed state normalization."""

import argparse
import csv
import warnings
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import plot_config  # noqa: F401  # Apply shared matplotlib styling.


METRICS = {
    "objective": ("rewards", r"$\mathcal{J}$"),
    "epi": ("EPI", r"$\mathcal{E}$"),
    "penalties": ("penalties", r"$\mathcal{P}$"),
}


def load_run(
    folder: Path,
    location: str,
    growth_year: int,
    start_day: int,
    horizon: int,
) -> dict[str, float] | None:
    paths = {
        metric: folder
        / f"{prefix}-300dt-{horizon}H-{location}-{growth_year}-{start_day}.csv"
        for metric, (prefix, _) in METRICS.items()
    }
    missing = [path for path in paths.values() if not path.exists()]
    if missing:
        return None
    return {
        metric: float(np.sum(np.loadtxt(path, delimiter=",")))
        for metric, path in paths.items()
    }


def confidence_interval(values: list[float]) -> dict[str, float | int]:
    data = np.asarray(values, dtype=float)
    n = data.size
    mean = float(np.mean(data))
    if n > 1:
        sem = stats.sem(data)
        half_width = float(stats.t.ppf(0.975, n - 1) * sem)
    else:
        half_width = 0.0
    return {
        "mean": mean,
        "ci_lower": mean - half_width,
        "ci_upper": mean + half_width,
        "n": n,
    }


def analyze(
    fixed_folder: Path,
    dynamic_folder: Path,
    growth_years: list[int],
    start_days: list[int],
    location: str,
    horizon: int,
) -> dict[str, dict[str, dict[str, float | int]]]:
    values = {
        metric: {"fixed": [], "dynamic": [], "difference": [], "relative": []}
        for metric in METRICS
    }
    matched = 0

    for growth_year, start_day in product(growth_years, start_days):
        fixed = load_run(fixed_folder, location, growth_year, start_day, horizon)
        dynamic = load_run(dynamic_folder, location, growth_year, start_day, horizon)
        if fixed is None or dynamic is None:
            warnings.warn(
                f"Skipping incomplete pair: year={growth_year}, start_day={start_day}"
            )
            continue

        matched += 1
        for metric in METRICS:
            difference = dynamic[metric] - fixed[metric]
            values[metric]["fixed"].append(fixed[metric])
            values[metric]["dynamic"].append(dynamic[metric])
            values[metric]["difference"].append(difference)
            if fixed[metric] != 0:
                values[metric]["relative"].append(
                    100 * difference / abs(fixed[metric])
                )
            else:
                warnings.warn(
                    f"Relative {metric} is undefined for zero fixed baseline: "
                    f"year={growth_year}, start_day={start_day}"
                )

    if matched == 0:
        raise ValueError("No complete fixed/dynamic result pairs found")

    return {
        metric: {
            statistic: confidence_interval(statistic_values)
            for statistic, statistic_values in metric_values.items()
        }
        for metric, metric_values in values.items()
    }


def print_report(
    results: dict[str, dict[str, dict[str, float | int]]]
) -> None:
    for metric, statistics in results.items():
        print(f"\n{metric.capitalize()}")
        for name in ("fixed", "dynamic", "difference", "relative"):
            result = statistics[name]
            suffix = "%" if name == "relative" else ""
            print(
                f"  {name:10s}: {result['mean']:.6g}{suffix} "
                f"(95% CI [{result['ci_lower']:.6g}, "
                f"{result['ci_upper']:.6g}], n={result['n']})"
            )


def save_report(
    results: dict[str, dict[str, dict[str, float | int]]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["metric", "statistic", "mean", "ci_lower", "ci_upper", "n"],
        )
        writer.writeheader()
        for metric, statistics in results.items():
            for statistic, result in statistics.items():
                writer.writerow(
                    {"metric": metric, "statistic": statistic, **result}
                )


def make_plot(
    results: dict[str, dict[str, dict[str, float | int]]],
    output_path: Path,
    show: bool,
) -> None:
    metrics = list(METRICS)
    means = [float(results[metric]["difference"]["mean"]) for metric in metrics]
    lower = [
        mean - float(results[metric]["difference"]["ci_lower"])
        for metric, mean in zip(metrics, means)
    ]
    upper = [
        float(results[metric]["difference"]["ci_upper"]) - mean
        for metric, mean in zip(metrics, means)
    ]

    width = 120 / 3 * 0.03937
    fig, ax = plt.subplots(figsize=(width, width), dpi=300)
    x = np.arange(len(metrics))
    bars = ax.bar(
        x,
        means,
        width=0.5,
        color="#1f77b4",
        edgecolor="white",
        linewidth=0.5,
        alpha=0.85,
    )
    ax.errorbar(
        x,
        means,
        yerr=[lower, upper],
        fmt="none",
        color="black",
        capsize=3,
        linewidth=1,
    )
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.set_ylabel(r"$\Delta$ Performance (dynamic $-$ fixed)")
    ax.set_xticks(x)
    ax.set_xticklabels([METRICS[metric][1] for metric in metrics])

    for bar, mean in zip(bars, means):
        ax.annotate(
            f"{mean:.3g}",
            (bar.get_x() + bar.get_width() / 2, mean),
            xytext=(0, 3 if mean >= 0 else -3),
            textcoords="offset points",
            ha="center",
            va="bottom" if mean >= 0 else "top",
            fontsize=7,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".svg"), bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def main(args: argparse.Namespace) -> None:
    results_root = Path(args.results_dir)
    results = analyze(
        results_root / args.fixed_folder,
        results_root / args.dynamic_folder,
        args.test_years,
        args.start_days,
        args.location,
        args.horizon,
    )

    output_path = Path(args.out_dir) / args.output_name
    print_report(results)
    save_report(results, output_path.with_suffix(".csv"))
    make_plot(results, output_path, args.show)
    print(f"\nSaved report and figure to {output_path.parent}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare dynamic and fixed RL-MPC state normalization."
    )
    parser.add_argument("--dynamic_folder", required=True)
    parser.add_argument("--fixed_folder", required=True)
    parser.add_argument("--test_years", type=int, nargs="+", required=True)
    parser.add_argument("--start_days", type=int, nargs="+", required=True)
    parser.add_argument(
        "--results_dir",
        default="results/GL-MPC-RL/deterministic/rlmpc",
    )
    parser.add_argument("--out_dir", default="outputs/figures")
    parser.add_argument("--output_name", default="normalization_performance")
    parser.add_argument("--location", default="Netherlands")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--show", action="store_true")
    main(parser.parse_args())
