from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np

from gl_py.agents.mpc import MPC

from model.utils import define_model, co2dens2ppm, vaporPres2rh #, init_state, load_dummy_weather
from stable_baselines3 import PPO, SAC

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from gl_py.environments.tomato_env import TomatoEnv

GAMMA = 1.0
ALGS = {
    "ppo": PPO,
    "sac": SAC,
}


class RLMPC(MPC):
    solver: ca.Function
    def __init__(
        self,
        nx: int,
        nu: int,
        ns: int,
        n_params: int,
        nd: int,
        dt: float,
        Np: int,
        nlp_opts: Dict[str, Any],
        algorithm: str,
        env_path: str,
        rl_env_params: Dict[str, Any],
        rl_model_path: str,
    ):
        super().__init__(nx, nu, ns, n_params, nd, dt, Np, nlp_opts)
        self.nx = nx
        self.nu = nu
        self.ns = ns
        self.n_params = n_params
        self.nd = nd
        self.Np = Np
        self.dt = dt
        self.nlp_opts = nlp_opts

        env_norm = TomatoEnv(**rl_env_params)
        env_norm = DummyVecEnv([lambda: env_norm])
        env_norm = VecNormalize(
            env_norm, 
            norm_obs = True, 
            norm_reward = False, 
            clip_obs = 10.,
            gamma=GAMMA,
        )
        env_norm = env_norm.load(env_path, env_norm)
        env_norm.training = False

        self.model = ALGS[algorithm].load(rl_model_path, env=self.eval_env)

        self.n_vars = nx + nu + ns

        # defining model dynamics
        self.F = define_model(self.nx, self.nu, self.nd, self.n_params, self.dt)
        self.constraints()

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
        total_cost = 0
        rewards_log = []
        
        
        obs = self.eval_env._get_obs()
        obs_log.append(obs)
        x_log.append(self.eval_env.get_numpy_state().ravel())

        # Freeze environment 
        if freeze:
            self.eval_env.freeze() # freeze variables

        done = False
        for i in range (0, horizon):

            # obs_norm = norm_obs(obs).toarray().squeeze(-1)
            obs_norm = self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel()
            action = self.actor_function(obs_norm).toarray().ravel()
            obs, reward, done, _,info = self.eval_env.step(action)
            x = self.eval_env.get_state()
            
            total_cost += reward
            rewards_log.append(reward)
            obs_log.append(obs)
            obs_norm_log.append(self.norm_obs_agent(obs, self.mean, self.variance).toarray().ravel())
            x_log.append(x)
            u_log.append(obs[4:7])
            
        # Unfreeze environment
        if freeze:
            self.eval_env.unfreeze()

        # Store data
        log["obs"] = np.vstack(obs_log).transpose()

        log["x"] = np.vstack(x_log).transpose()
        log["u"] = np.vstack(u_log).transpose()
        log["obs_norm"] = np.vstack(obs_norm_log).transpose()
        log["total_reward"] = total_cost
        log["reward_log"] = np.array(rewards_log)

        return log


    def define_nlp(self):
        """
        Defining the Non-linear program using CasADi.
        """
        # Control variables (decision variables)
        U = ca.MX.sym("U", self.nu, self.Np)
        S = ca.MX.sym("S", self.ns, self.Np)  # Slack variables

        # Parameters (initial state, disturbances, and parameters)
        X0 = ca.MX.sym("X0", self.nx)
        D = ca.MX.sym("D", self.nd, self.Np)
        P = ca.MX.sym("P", self.n_params)
        U0 = ca.MX.sym("U0", self.nu)  # Initial control input

        # Initialize cost and constraints
        J = 0

        # Initialize state trajectory
        Xk = X0

        # # Initialize c as a (n+m) vector of zeros
        hour_conversion = 3600/self.dt

        # initialize constraints
        g =  []
        self.lbg = []
        self.ubg = []

        g.append(U[:, 0] - U0)
        self.lbg.extend(-self.du_max)
        self.ubg.extend(self.du_max)

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
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
                0.3 * Uk[1]* P[109]/P[46] * 1e-6 * self.dt                        # costs for CO2


            S, S_constraints, S_lbg, S_ubg = self.set_slack_variables(k, Xk, S)
            g.extend(S_constraints)
            self.lbg.extend(S_lbg)
            self.ubg.extend(S_ubg)

            J += ca.sum1(S[:, k])


        J += - (Xk[25]-X0[25])* 1e-6 / 0.06 * 1.2              # revenue from selling tomatoes

        # Decision variables
        w = ca.vertcat(ca.vec(U), ca.vec(S))

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(*g)

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, U0, ca.vec(D), P)

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

        # Initialize cost and constraints
        J = 0

        # Initialize state trajectory
        # Xk = X0

        hour_conversion = 3600/self.dt

        # initialize constraints
        g =  []
        self.lbg = []
        self.ubg = []

        g.append(X[:, 0] - X0)
        self.lbg.extend([0]*self.nx)
        self.ubg.extend([0]*self.nx)

        g.append(U[:, 0] - U0)
        self.lbg.extend(-self.du_max)
        self.ubg.extend(self.du_max)

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
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
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
        p_nlp = ca.vertcat(X0, U0, ca.vec(D), P)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver options

        # Create solver
        self.solver_multi = ca.nlpsol("solver", "ipopt", nlp, self.nlp_opts)
