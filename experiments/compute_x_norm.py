import argparse
from itertools import product
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from common.utils import load_env, load_env_params


PROJECT = "GL-MPC-RL"
ALGORITHM = "ppo"
MODE = "deterministic"
ENV_ID = "TomatoEnv"


def parse_start_days(value: str) -> list[int]:
    if "-" not in value:
        return [int(value)]

    start, end = map(int, value.split("-", 1))
    if start > end:
        raise argparse.ArgumentTypeError("start-day range must be ascending")
    return list(range(start, end + 1))


def simulation_mean(model: PPO, env) -> np.ndarray:
    observations = env.reset()
    policy_state = None
    episode_starts = np.ones((1,), dtype=bool)
    states = []

    for _ in range(env.get_attr("N")[0]):
        # Log before stepping because VecEnv automatically resets after the terminal step.
        states.append(env.env_method("get_state")[0])
        actions, policy_state = model.predict(
            observations,
            state=policy_state,
            episode_start=episode_starts,
            deterministic=True,
        )
        observations, _, episode_starts, _ = env.step(actions)

    return np.mean(states, axis=0)


def main(args: argparse.Namespace) -> None:
    load_path = Path("train_data") / PROJECT / ALGORITHM / MODE
    model_path = load_path / "models" / args.model_name / "best_model.zip"
    model = PPO.load(model_path, device="cpu")

    simulation_means = []
    for growth_year, start_day in product(args.growth_years, args.start_days):
        print(f"Simulating growth year {growth_year}, start day {start_day}")

        base_params, specific_params = load_env_params(ENV_ID, "configs/envs/")
        specific_params["eval_options"]["location"] = args.location
        specific_params["eval_options"]["eval_years"] = [growth_year]
        specific_params["eval_options"]["eval_days"] = [start_day]

        env = load_env(
            ENV_ID,
            args.model_name,
            base_params,
            specific_params,
            str(load_path),
            args.frame_stack,
            args.n_stack,
            True if args.plant_state_env else None,
        )
        try:
            simulation_means.append(simulation_mean(model, env))
        finally:
            env.close()

    x_norm = np.mean(simulation_means, axis=0)
    output_path = load_path / "envs" / args.model_name / "x_norm.npy"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, x_norm)

    print(f"x_norm: {x_norm}")
    print(f"Saved normalization vector to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute a general state normalization vector from RL rollouts."
    )
    parser.add_argument("--model_name", required=True, help="W&B model name")
    parser.add_argument("--growth_years", type=int, nargs="+", required=True)
    parser.add_argument(
        "--start_days",
        type=parse_start_days,
        nargs="+",
        required=True,
        help="Start days or inclusive ranges, e.g. 90 105 or 90-151",
    )
    parser.add_argument("--location", default="Netherlands")
    parser.add_argument("--frame_stack", action="store_true")
    parser.add_argument("--n_stack", type=int, default=1)
    parser.add_argument("--plant_state_env", action="store_true")
    args = parser.parse_args()
    args.start_days = [day for values in args.start_days for day in values]
    main(args)
