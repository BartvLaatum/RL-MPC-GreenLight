import os
import argparse
from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np

from controllers.mpc import MPC

from stable_baselines3 import PPO, SAC
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecFrameStack

from environments.tomato_env import TomatoEnv
from environments.utils import co2ppm2dens, rh2vaporDens, define_model 

class RLMPC(MPC):
    def __init__(
        self,
        nx: int,
        nu: int,
        ns: int,
        n_params: int,
        nd: int,
        dt: float,
        horizon: int,
        u_min: List[float],
        u_max: List[float],
        delta_u_max: float,
        constraints: Dict[str, Any],
        reward_params: Dict[str, Any],
        region_range: float,
        nlp_opts: Dict[str, Any],
        terminal_constraint: bool,
        eval_env: DummyVecEnv,
        model: BaseAlgorithm,
        terminal_penalty: bool = False,
    ) -> None:
        super().__init__(
            nx,
            nu,
            ns,
            n_params,
            nd,
            dt,
            horizon,
            u_min,
            u_max,
            delta_u_max,
            constraints,
            reward_params,
            nlp_opts,
        )
        self.terminal_constraint = terminal_constraint
        self.terminal_penalty = terminal_penalty
        self.eval_env = eval_env
        self.model = model
        self.region_range = region_range

        # penalize deviations in first 3 climate states and fruit biomass
        # self.terminal_penalty_weights = np.zeros(shape=(3,))
        self.terminal_penalty_weights = np.zeros(shape=(self.nx,))
        self.terminal_penalty_weights[...] = 1e-5

        # defining model dynamics
        # self.F = define_model(self.nx, self.nu, self.nd, self.n_params, self.dt)

    def unroll_actor(self, horizon=1, freeze=True):
        """
        Unrolls the actor function over the environment for a specified number of steps.
        Parameters:
            arg_actor_function (function): Actor function to determine actions based on normalized observations.
            freeze (bool): If True, freezes the environment"s state during the unrolling process. Default is True.
        Returns:
        dict: A dictionary containing logs of observations, states, actions, normalized observations, total reward, and reward log.
            - "obs" (numpy.ndarray): Transposed array of logged observations.
            - "x" (numpy.ndarray): Transposed array of logged states.
            - "u" (numpy.ndarray): Transposed array of logged actions.
            - "obs_norm" (numpy.ndarray): Transposed array of logged normalized observations.
            - "total_reward" (float): Total accumulated reward.
            - "reward_log" (numpy.ndarray): Array of logged rewards.
        """

        # Define empty log variables
        log = {
            "obs":[],
            "x":[],
            "u":[],
        }
        obs_log, obs_norm_log, x_log, u_log = [],[],[],[]
        cumulative_reward = 0
        rewards_log = []

        # if using frame stacking we need to handle observation synchronization
        if isinstance(self.eval_env, VecFrameStack):
            n_stack = self.eval_env.stacked_obs.n_stack
            # Get raw observation for logging (current frame, unnormalized)
            raw_obs = self.eval_env.env_method("_get_obs")[0]
            obs_log.append(raw_obs)
            x_log.append(self.eval_env.env_method("get_state")[0])
                        
            # Sync the frame stack buffer with current environment state.
            # This is necessary because set_env_state() may have been called,
            # which changes the underlying env but not the VecFrameStack buffer.
            # Get the current normalized observation from the inner VecNormalize
            current_obs = self.eval_env.venv.normalize_obs(raw_obs).reshape(1, -1)
            # Update the stacked observations buffer with the current observation.
            # This shifts previous observations and adds the new one, preserving history.
            dones = np.array([False])  # Not a terminal state
            infos = [{}]  # Empty info dict
            if self.eval_env.get_attr("timestep")[0] == 0:
                self.eval_env.stacked_obs.reset(current_obs)
            else:
                self.eval_env.stacked_obs.update(current_obs, dones, infos)

            # Get stacked observation for prediction (already normalized by inner VecNormalize)
            obs = self.eval_env.stacked_obs.stacked_obs.copy()
        else:
            n_stack = 1
            obs = self.eval_env.env_method("_get_obs")[0]
            obs_log.append(obs)
            x_log.append(self.eval_env.env_method("get_state")[0])
            obs = self.eval_env.normalize_obs(obs).reshape(1, -1)  # Normalize the observation

        states = None # States for the model, can be used for recurrent models
        episode_starts = np.ones((1,), dtype=bool) # Not used in this context, can be used for recurrent models
        terminated = False

        # Freeze environment (save state for restoration after rollout)
        if freeze:
            self.eval_env.env_method("freeze")  # freeze environment variables
            # Also save the stacked observations buffer for VecFrameStack
            if isinstance(self.eval_env, VecFrameStack):
                frozen_stacked_obs = self.eval_env.stacked_obs.stacked_obs.copy()
        for i in range(0, horizon):
            action, states = self.model.predict(
                obs,
                state=states,
                episode_start=episode_starts,
                deterministic=True,
        )
            obs, rewards, terminated, infos = self.eval_env.step(action)
            x = self.eval_env.env_method("get_state")[0]

            cumulative_reward += rewards
            rewards_log.append(rewards)

            obs_norm_log.append(obs[0])
            # For stacked observations, extract the last frame before unnormalizing
            if n_stack > 1:
                frames = np.split(obs, n_stack, axis=1)
                obs_log.append(self.eval_env.unnormalize_obs(frames[-1]).ravel())
            else:
                obs_log.append(self.eval_env.unnormalize_obs(obs[0]).ravel())
            x_log.append(x)
            u_log.append(infos[0]["controls"])
        # Unfreeze environment (restore state after rollout)
        if freeze:
            self.eval_env.env_method("unfreeze")
            # Also restore the stacked observations buffer for VecFrameStack
            if isinstance(self.eval_env, VecFrameStack):
                self.eval_env.stacked_obs.stacked_obs[:] = frozen_stacked_obs

        # Store data
        log["obs"] = np.vstack(obs_log).transpose()

        log["x"] = np.vstack(x_log).transpose()
        log["u"] = np.vstack(u_log).transpose()
        log["obs_norm"] = np.vstack(obs_norm_log).transpose()
        log["cumulative_reward"] = cumulative_reward
        log["reward_log"] = np.array(rewards_log)

        return log

    def clip_bounds_terminal_state(self, x):
        """
        Clip selected elements of terminal state vector `x` to physical bounds.

        - x[0]  in [co2ppm2dens(x[2], y_min[0]),  co2ppm2dens(x[2], y_max[0])]
        - x[2]  in [y_min[1],                      y_max[1]]
        - x[15] in [rh2vaporDens(x[2], y_min[2]), rh2vaporDens(x[2], y_max[2])]
        """

        def clip(v, lo, hi):
            return ca.fmin(ca.fmax(v, lo), hi)

        repl = {
            0: clip(x[0],
                    co2ppm2dens(x[2], self.y_min[0]),
                    co2ppm2dens(x[2], self.y_max[0])),
            2: clip(x[2], self.y_min[1], self.y_max[1]),
            15: clip(x[15],
                    rh2vaporDens(x[2], self.y_min[2]),
                    rh2vaporDens(x[2], self.y_max[2])),
        }

        return ca.vertcat(*[repl.get(i, x[i]) for i in range(x.size1())])

    def define_nlp(self):
        """
        Defining the Non-linear program using CasADi.
        """
        # Control variables (decision variables)
        U = ca.MX.sym("U", self.nu, self.Np)
        S = ca.MX.sym("S", self.ns, self.Np)
        XN = ca.MX.sym("XN", 3)

        # Parameters (initial state, disturbances, and parameters)
        X0 = ca.MX.sym("X0", self.nx)
        D = ca.MX.sym("D", self.nd, self.Np)
        P = ca.MX.sym("P", self.n_params)
        U0 = ca.MX.sym("U0", self.nu)  # Initial control input

        # Terminal state (optional, can be used for terminal constraints)
        X_terminal = ca.MX.sym("X_terminal", 3, 1)

        # Initialize cost and constraints
        J = 0

        # Initialize state trajectory
        Xk = X0

        # initialize constraints
        g =  []
        self.lbg = []
        self.ubg = []

        g.append(U[:, 0] - U0)
        self.lbg.extend(-self.du_max)
        self.ubg.extend(self.du_max)

        # Terminal state constraints (optional, can be used for terminal constraints)        
        # lower‐inequality: X[-1] ≥ 0.95 X_terminal
        print(f"Using terminal constraints: {self.terminal_constraint}")
        if self.terminal_constraint:

            terminal_penalty = ca.dot(ca.DM(self.terminal_penalty_weights), ca.fabs(XN - X_terminal))
            J += terminal_penalty

        # Loop over prediction horizon
        for k in range(self.Np):
            Uk = U[:, k]
            Dk = D[:, k]

            if k > 0:
                g.append(Uk - U[:, k-1])
                self.lbg.extend(-self.du_max)
                self.ubg.extend(self.du_max)

            # Integrate to get the next state
            res = self.F(x0=Xk, u=Uk, p=ca.vertcat(Dk, P))
            Xk = res['xf']

            # economic objective
            # convert boil power to kWh (costs for heating)
            # convert lamp electricity to kWh (costs for lighting)
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/self.hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/self.hour_conversion + \
                0.3 * Uk[1]* P[109]/P[46] * 1e-6 * self.dt                        # costs for CO2

            S, S_constraints, S_lbg, S_ubg = self.set_slack_variables(k, Xk, S)
            g.extend(S_constraints)
            self.lbg.extend(S_lbg)
            self.ubg.extend(S_ubg)

            J += ca.sum1(S[:, k])


        g.append(XN - Xk[0, 2, 15])
        self.lbg.extend([0]*X_terminal.size1())
        self.ubg.extend([0]*X_terminal.size1())

        J += - (Xk[25]-X0[25])* 1e-6 / 0.06 * 1.2   # revenue from selling tomatoes

        # Decision variables
        w = ca.vertcat(ca.vec(U), ca.vec(S), ca.vec(XN))

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(*g)

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, U0, ca.vec(D), P, X_terminal)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver
        self.solver_single = ca.nlpsol("solver", "ipopt", nlp, self.nlp_opts)

    def define_nlp_multi(self):
        """
        Defining the Non-linear program using CasADi.
        """
        # Control variables (decision variables)
        U = ca.MX.sym("U", self.nu, self.Np)
        X = ca.MX.sym("X", self.nx, self.Np+1)             # state trajectory over the horizon
        S = ca.MX.sym("S", self.ns, self.Np)  # Slack variables

        # Parameters (initial state, disturbances, and parameters)
        X0 = ca.MX.sym("X0", self.nx)
        D = ca.MX.sym("D", self.nd, self.Np)
        P = ca.MX.sym("P", self.n_params)
        U0 = ca.MX.sym("U0", self.nu)  # Initial control input

        # Terminal state (optional, can be used for terminal constraints)
        X_terminal = ca.MX.sym("X_terminal", self.nx, 1)

        # Initialize cost and constraints
        J = 0

        # Initialize state trajectory
        # Xk = X0

        # initialize constraints
        g =  []
        self.lbg = []
        self.ubg = []

        # Initial state constraints
        g.append(X[:, 0] - X0)
        self.lbg.extend([0]*self.nx)
        self.ubg.extend([0]*self.nx)

        # Initial control input constraints
        g.append(U[:, 0] - U0)
        self.lbg.extend(-self.du_max)
        self.ubg.extend(self.du_max)

        # Terminal state constraints (optional, can be used for terminal constraints)        
        print(f"Using terminal constraints: {self.terminal_constraint}")
        if self.terminal_constraint:
            # lower-inequality: X[:, -1] >= 0.95*X_terminal
            # lower_region_constraint = self.clip_bounds_terminal_state(0.95*X_terminal)
            # g.append(X[:, -1] - lower_region_constraint)
            # g.append(X[:, -1] - 0.975*X_terminal)
            g.append(X[:, -1] - (1-self.region_range)*X_terminal)

            self.lbg.extend([0.0]*X_terminal.size1())
            self.ubg.extend([float('inf')]*X_terminal.size1())

            # upper‐inequality: X[-1] <= 1.05*X_terminal
            # upper_region_constraint = self.clip_bounds_terminal_state(1.05*X_terminal)
            # g.append(X[:, -1] - upper_region_constraint)
            # g.append(X[:, -1] - 1.025*X_terminal)
            g.append(X[:, -1] - (1+self.region_range)*X_terminal)
            self.lbg.extend([-float('inf')]*X_terminal.size1())
            self.ubg.extend([0.0]*X_terminal.size1())

        if self.terminal_penalty:
            terminal_penalty = ca.dot(self.terminal_penalty_weights, ca.fabs(X[:, -1] - X_terminal))
            J += terminal_penalty

        # Loop over prediction horizon
        for k in range(self.Np):
            Uk = U[:, k]
            Dk = D[:, k]

            if k > 0:
                g.append(Uk - U[:, k-1])
                self.lbg.extend(-self.du_max)
                self.ubg.extend(self.du_max)

            # Integrate to get the next state
            res = self.F(x0=X[:, k], u=Uk, p=ca.vertcat(Dk, P))
            X_next = res["xf"]

            # Add dynamic constraints: the next state decision variable must equal the integration result
            g.append(X[:, k+1] - X_next)
            self.lbg.extend([0]*self.nx)
            self.ubg.extend([0]*self.nx)

            # economic objective
            # convert boil power to kWh (costs for heating)
            # convert lamp electricity to kWh (costs for lighting)
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/self.hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/self.hour_conversion + \
                0.3 * Uk[1]* P[109]/P[46] * 1e-6 * self.dt                        # costs for CO2

            S, S_constraints, S_lbg, S_ubg = self.set_slack_variables(k, X_next, S)
            g.extend(S_constraints)
            self.lbg.extend(S_lbg)
            self.ubg.extend(S_ubg)

            J += ca.sum1(S[:, k])

        J += - (X[25, -1]-X0[25])* 1e-6 / 0.06 * 1.2              # revenue from selling tomatoes

        # Decision variables
        w = ca.vertcat(ca.vec(U), ca.vec(X), ca.vec(S))

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(*g)

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, U0, ca.vec(D), P, X_terminal)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver options

        # Create solver
        self.solver_multi = ca.nlpsol("solver", "ipopt", nlp, self.nlp_opts)
