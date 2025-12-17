
from typing import Any, Dict, List, Optional, Tuple, SupportsFloat

import numpy as np
import casadi as ca
from gymnasium import spaces

from environments.base_env import GreenLightEnv
from environments.observations import *
from environments.rewards import BaseReward, GreenhouseReward, DummyReward
from environments.utils import load_weather_data, load_dummy_weather, define_model, init_state
from model.parameters import init_default_params
from environments.noise import parametric_crop_uncertainty

from copy import deepcopy

REWARDS = {
    "GreenhouseReward": GreenhouseReward,
    "DummyReward": DummyReward
}


OBSERVATION_MODULES = {
    "StateObservations": StateObservations,
    "IndoorClimateObservations": IndoorClimateObservations,
    "BasicCropObservations": BasicCropObservations,
    "ControlObservations": ControlObservations,
    "WeatherObservations": WeatherObservations,
    "WeatherForecastObservations": WeatherForecastObservations,
    "TimeObservations": TimeObservations
}

class TomatoEnv(GreenLightEnv):
    def __init__(self,
        reward_function: str,                   # reward function
        observation_modules: List[str],         # observation function
        constraints: Dict[str, Any],            # constraints for the environment
        eval_options: Dict[str, Any],           # days for evaluation
        reward_params: Dict[str, Any] = {},     # reward function arguments
        base_env_params: Dict[str, Any] = {},   # base environment parameters
        uncertainty_scale = 0.0
        ) -> None:
        super(TomatoEnv, self).__init__(**base_env_params)

        self.uncertainty_scale = uncertainty_scale

        # set year and days for the evaluation environment 
        self.eval_options = eval_options

        # initialise the observation and action spaces
        self.observation_modules = self._init_observations(observation_modules)
        self.observation_space = self._generate_observation_space()
        self.action_space = self._generate_action_space()

        self.F = define_model(
            nx=self.nx,
            nu=self.nu,
            nd=self.nd,
            n_params=self.num_params,
            dt=self.dt,
        )

        self.constraints_low = np.array([
            constraints["co2_min"],
            constraints["temp_min"],
            constraints["rh_min"],
        ])

        self.constraints_high = np.array([
            constraints["co2_max"],
            constraints["temp_max"],
            constraints["rh_max"],
        ])
        # set the default parameters
        self.p = init_default_params(self.num_params)

        # initialise the reward function
        self.reward = self._init_rewards(reward_function, reward_params)

    def get_state(self) -> np.ndarray:
        """
        Function that returns the current state of the environment.
        The state is a concatenation of the current state and the control input.
        """
        return self.x

    def freeze(self) -> None:
        """
        Function that freezes the current state of the environment.
        This is used to save the current state of the environment for later use.
        """
        self.freeze_k = deepcopy(self.timestep)
        self.freeze_x = deepcopy(self.x)
        self.freeze_observation = deepcopy(self.obs)
        # self.freeze_y = deepcopy(self.y)
        self.freeze_x_prev = deepcopy(self.x_prev)
        # self.freeze_y_prev = deepcopy(self.y_prev)
        self.freeze_done = deepcopy(self.terminated)
        self.freeze_u = deepcopy(self.u)
        self.freeze_day_of_year = deepcopy(self.day_of_year)
        self.freeze_hour_of_day = deepcopy(self.hour_of_day)

    def unfreeze(self) -> None:
        """
        Function that unfreezes the current state of the environment.
        This is used to restore the state of the environment after it has been frozen.
        """
        self.timestep = deepcopy(self.freeze_k)
        self.x = deepcopy(self.freeze_x)
        self.x_prev = deepcopy(self.freeze_x_prev)
        self.obs = deepcopy(self.freeze_observation)
        # self.y              = deepcopy(self.freeze_y)
        # self.y_prev         = deepcopy(self.freeze_y_prev)
        self.done = deepcopy(self.freeze_done)
        self.u = deepcopy(self.freeze_u)
        self.day_of_year = deepcopy(self.freeze_day_of_year)
        self.hour_of_day = deepcopy(self.freeze_hour_of_day)

    def _terminalState(self) -> bool:
        """
        Function that checks whether the simulation has reached a terminal state.
        Terminal states are reached when the simulation has reached the end of the growing season.
        """
        if self.timestep >= self.N:
            return True
        return False

    def _init_observations(
        self,
        observation_modules: List[str],
    ) -> List[BaseObservations]:
        return [OBSERVATION_MODULES[module](self) for module in observation_modules]

    def _generate_observation_space(self) -> spaces.Box:
        spaces_low_list = []
        spaces_high_list = []

        for module in self.observation_modules:
            module_obs_space = module.observation_space()
            spaces_low_list.append(module_obs_space.low)
            spaces_high_list.append(module_obs_space.high)

        low = np.concatenate(spaces_low_list, axis=0)  # Concatenate low bounds
        high = np.concatenate(spaces_high_list, axis=0)  # Concatenate high bounds

        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _generate_action_space(self) -> spaces.Box:
        return spaces.Box(low=-1, high=1, shape=(self.nu,), dtype=np.float32)

    def _init_rewards(self, reward_function: str, reward_params: Dict[str, Any]) -> BaseReward:
        return REWARDS[reward_function](self, **reward_params)

    def _get_reward(self) -> SupportsFloat:
        return self.reward.compute_reward()

    def _scale(self, action, action_min, action_max):
        return (action + 1) * (action_max - action_min) / 2 + action_min

    def action_to_control(self, action: np.ndarray) -> np.ndarray:
        """
        Function that converts the action to control inputs.
        """
        return np.clip(self.u + action*self.delta_u_max, self.u_min, self.u_max) 

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, SupportsFloat, bool, bool, Dict[str, Any]]:
        # scale the action from controller (between -1, 1) to (u_min, u_max)
        self.u = self.action_to_control(action)
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        try:
            p_dyn = ca.vertcat(ca.DM(self.weather_data[self.timestep]), self.p)
            res = self.F(x0=ca.DM(self.x), u=ca.DM(self.u), p=p_dyn)
            self.x = res["xf"].full().flatten()
        except:
            print("Error in ODE approximation")
            self.terminated = True

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day += (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()
        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()
        # additional information to return
        info = self._get_info()
        self.timestep += 1
        self.x_prev = np.copy(self.x)

        return (
                self.obs,
                reward, 
                self.terminated, 
                False,
                info
                )

    def step_raw_control(self, control: np.ndarray):
        self.u = control
        params = parametric_crop_uncertainty(self.p, self.uncertainty_scale, self._np_random)
        self.x = self.F(self.x, self.u, self.weather_data[self.timestep], params)

        # update time
        self.day_of_year += (self.dt/self.c) % 365
        self.hour_of_day +=  (self.dt/3600)
        self.hour_of_day = self.hour_of_day % 24

        self.obs = self._get_obs()

        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()
        self.timestep += 1
        self.x_prev = np.copy(self.x)
        # additional information to return
        return (
                self.obs,
                reward,
                self.terminated,
                False,
                self._get_info()
                )

    def step_raw_control_pipeinput(self, control: np.ndarray):
        self.u = control

        self.x = self.F(self.x, self.u, self.weather_data[self.timestep], self.p)

        if self._terminalState():
            self.terminated = True
        # compute reward
        reward = self._get_reward()

        # additional information to return
        # info = self._get_info()
        self.timestep += 1
        return (
                self.x,
                self.terminated, 
                )

    def _get_obs(self):
        obs = []
        for module in self.observation_modules:
            obs.append(module.compute_obs())
        obs = np.concatenate(obs, axis=0)
        return obs

    def get_obs_names(self):
        """
        """
        obs_names = []
        for module in self.observation_modules:
            obs_names.extend(module.obs_names)
        return obs_names

    def _get_info(self) -> Dict[str, Any]:
        return {
            "EPI": self.reward.profit,
            "revenue": self.reward.gains,
            "penalty": self.reward.penalty,
            "variable_costs": self.reward.variable_costs,
            "fixed_costs": self.reward.fixed_costs,
            "co2_cost": self.reward.co2_costs,
            "heat_cost": self.reward.heat_costs,
            "elec_cost": self.reward.elec_costs,
            "temp_violation": self.reward.temp_violation,
            "co2_violation": self.reward.co2_violation,
            "rh_violation": self.reward.rh_violation,
            "lamp_violation": self.reward.lamp_violation,
            "controls": self.u,
        }

    def set_crop_state(self, cBuf: float, cLeaf: float, cStem: float, cFruit: float, tCanSum: float) -> None:
        self.x[22] = cBuf
        self.x[23] = cLeaf
        self.x[24] = cStem
        self.x[25] = cFruit
        self.x[26] = tCanSum

    def set_env_state(self, x, x_prev, u, timestep, hour_of_day, day_of_year) -> None:
        self.x = np.copy(x)
        self.x_prev = np.copy(x_prev)

        self.timestep = timestep
        self.u = np.copy(u).ravel()
        # d = self.get_d().ravel()
        self.day_of_year = day_of_year
        self.hour_of_day = hour_of_day
        self.obs = self._get_obs()
        # self.obs = np.concatenate([y, self.u, [self.timestep], d], dtype=np.float32)

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)


        # pick a random growth year and start day if we are training
        if self.training:
            self.growth_year = self._np_random.choice(self.train_years)
            self.start_day = self._np_random.choice(self.train_days)
        else:
            self.growth_year = self._np_random.choice(self.eval_options["eval_years"])
            self.start_day = self._np_random.choice(self.eval_options["eval_days"])
            self.location = self.eval_options["location"]
            self.increase_eval_idx()

        self.day_of_year = self.start_day
        self.hour_of_day = 0

        # load in weather data for specific simulation
        self.weather_data = load_weather_data(
            weather_data_dir=self.weather_data_dir,
            location=self.location,
            growth_year=self.growth_year,
            start_day=self.start_day,
            n_days=self.season_length,
            pred_horizon=self.pred_horizon,
            h=self.dt,
            nd=self.nd
        )

        self.u = np.ones(self.nu) * 0.5  # initial control input
        self.x = init_state(self.weather_data[0], 85.)
        self.x_prev = np.copy(self.x)
        self.timestep = 0
        self.obs = self._get_obs()


        self.terminated = False
        return self.obs, {}
