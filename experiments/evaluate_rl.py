import argparse
import os
from os.path import join
from tqdm import tqdm
import numpy as np

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack

from common.results import Results
from common.utils import load_env_params, load_env, load_model_hyperparams

ALG = {"ppo": PPO, 
       "sac": SAC}


def evaluate(model, env):
    N = env.get_attr("N")[0]
    epi, penalty, revenue, heat_cost, co2_cost, elec_cost = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1), np.zeros(N+1),np.zeros(N+1)
    temp_violation, co2_violation, rh_violation = np.zeros(N+1), np.zeros(N+1), np.zeros(N+1)
    episodic_obs = np.zeros((N+1, 20))
    episode_rewards = np.zeros(N+1)
    episode_rewards = np.zeros(N+1)
    dones = np.zeros((1,), dtype=bool)
    episode_starts = np.ones((1,), dtype=bool)

    observations = env.reset()
    timestep = 0
    states = None
    frame_stack = False
    if isinstance(env, VecFrameStack):
        frame_stack = True
        n_stack = env.stacked_obs.n_stack

    for timestep in range(N):
        actions, states = model.predict(
            observations,  # type: ignore[arg-type]
            state=states,
            episode_start=episode_starts,
            deterministic=True,
    )
        observations, rewards, dones, infos = env.step(actions)
        episode_rewards[timestep] += rewards[0]
        # episodic_obs[timestep] += env.unnormalize_obs(observations)[0, :23]
        if frame_stack:
            frames = np.split(observations, n_stack, axis=1)
            episodic_obs[timestep] += env.unnormalize_obs(frames[-1])[0, :20]
        else:
            episodic_obs[timestep] += env.unnormalize_obs(observations)[0, :20]

        epi[timestep] += infos[0]["EPI"]
        penalty[timestep] += infos[0]["penalty"]
        revenue[timestep] += infos[0]["revenue"]
        heat_cost[timestep] += infos[0]["heat_cost"]
        elec_cost[timestep] += infos[0]["elec_cost"]
        co2_cost[timestep] += infos[0]["co2_cost"]
        temp_violation[timestep] += infos[0]["temp_violation"]
        co2_violation[timestep] += infos[0]["co2_violation"]
        rh_violation[timestep] += infos[0]["rh_violation"]
        timestep += 1

    result_data = np.column_stack((episodic_obs, episode_rewards, epi, penalty, revenue, heat_cost, co2_cost, elec_cost, temp_violation, co2_violation, rh_violation))
    return result_data[:-1]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="GL-MPC-RL", help="Name of the project (in wandb)")
    parser.add_argument("--env_id", type=str, default="TomatoEnv", help="Environment ID")
    parser.add_argument("--model_name", type=str, default="cosmic-music-45", help="Name of the trained RL model")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Name of the algorithm (ppo or sac)")
    parser.add_argument("--uncertainty_scale", type=float, help="Uncertainty scale", required=True)
    parser.add_argument("--mode", type=str, choices=['deterministic', 'stochastic'], required=True)
    parser.add_argument("--frame_stack", action="store_true", help="Whether to use frame stacking")
    parser.add_argument("--n_stack", type=int, default=1, help="Number of frames to stack")
    args = parser.parse_args()

    assert not (args.mode == "deterministic" and args.uncertainty_scale != 0.0), \
        "Uncertainty scale must be 0.0 for deterministic mode"

    env_config_path = f"configs/envs/"
    load_path = f"train_data/{args.project}/{args.algorithm}/{args.mode}/"
    if args.mode == "stochastic":
        save_dir = f"results/{args.project}/{args.mode}/{args.algorithm}/{args.uncertainty_scale}/"
        n_sims = 30
    else:
        save_dir = f"results/{args.project}/{args.mode}/{args.algorithm}"
        n_sims = 1
    os.makedirs(save_dir, exist_ok=True)

    # load in the environment and model
    env_base_params, env_specific_params = load_env_params(args.env_id, env_config_path)
    model_params = load_model_hyperparams(args.algorithm, args.env_id)
    # env_specific_params["uncertainty_scale"] = args.uncertainty_scale
    eval_env = load_env(args.env_id, args.model_name, env_base_params, env_specific_params, load_path, args.frame_stack, args.n_stack)

    model = ALG[args.algorithm].load(join(load_path + f"models", f"{args.model_name}/best_model.zip"), device="cpu")

    result_columns = eval_env.env_method("get_obs_names")[0][:20]
    print(result_columns)
    result_columns.extend(["Rewards",  "EPI", "Penalty", "Revenue", "Heat costs", "CO2 costs", "Elec costs"])
    result_columns.extend(["temp_violation", "co2_violation", "rh_violation"])
    result_columns.extend(["episode"])
    result = Results(result_columns)

    for sim in tqdm(range(n_sims)):
        # eval_env.set_seed(sim)
        eval_env.env_method("set_seed", 666+sim)
        result_data = evaluate(model, eval_env)
        sim_column = np.full((result_data.shape[0], 1), sim)
        result_data = np.column_stack((result_data, sim_column))

        result.update_result(result_data)

    start_day = eval_env.get_attr("start_day")[0]
    growth_year = eval_env.get_attr("growth_year")[0]
    location = eval_env.get_attr("location")[0]

    save_name = f"{args.model_name}-{location}-{growth_year}-{start_day}.csv"
    print("saving results to", save_name)
    result.save(f"{save_dir}/{save_name}")
