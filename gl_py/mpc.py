from typing import Any, Dict

import casadi as ca
import numpy as np


# from model.aux_states import update
# from model.ode import ODE
from model.utils import define_model #, init_state, load_dummy_weather
# from model.parameters import init_default_params
# from model.utils import load_dummy_weather, init_state
# from model.parameters import init_default_params


class MPC:
    solver: ca.Function
    def __init__(
        self,
        nx: int,
        nu: int,
        n_params: int,
        nd: int,
        dt: float,
        Np: int,
        nlp_opts: Dict[str, Any],
    ):
        self.nx = nx
        self.nu = nu
        self.n_params = n_params
        self.nd = nd
        self.Np = Np
        self.dt = dt
        self.nlp_opts = nlp_opts

        # defining model dynamics
        self.F = define_model(self.nx, self.nu, self.nd, self.n_params, self.dt)


    def define_nlp(self):
        """
        Defining the Non-linear program using CasADi.
        """

        # Control variables (decision variables)
        U = ca.MX.sym("U", self.nu, self.Np)

        # Parameters (initial state, disturbances, and parameters)
        X0 = ca.MX.sym("X0", self.nx)
        D = ca.MX.sym("D", self.nd, self.Np)
        P = ca.MX.sym("P", self.n_params)

        # Initialize cost and constraints
        J = 0


        # Initialize state trajectory
        Xk = X0

        # # Initialize c as a (n+m) vector of zeros
        hour_conversion = 3600/self.dt

        # initialize constraints
        g =  []

        # Loop over prediction horizon
        for k in range(self.Np):
            Uk = U[:, k]
            Dk = D[:, k]

            # Concatenate state and control vectors

            # Integrate to get the next state
            res = self.F(x0=Xk, u=Uk, p=ca.vertcat(Dk, P))

            Xk = res['xf']

            # economic objective
            # convert boil power to kWh (costs for heating)
            # convert lamp electricity to kWh (costs for lighting)
            # J += 0.09 * P[108] *1e-3 * Uk[0]/hour_conversion + \
            #     0.2 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
            #     0.3 * Uk[1] * 1e-6 * self.dt                        # costs for CO2
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/hour_conversion + \
                0.09 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
                0.1 * Uk[1]* P[109] * 1e-6 * self.dt                        # costs for CO2

        # J += - (Xk[25]-X0[25])* 1e-6 / 0.08 * 1.2              # revenue from selling tomatoes
        J += - (Xk[25]-X0[25])/0.06 * 1e-6 * 1.2              # revenue from selling tomatoes = (mg m^{-2}) * 10^{-6} * m^{2} * euro/kg 

        # Decision variables
        w = ca.vec(U)

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(*g)

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, ca.vec(D), P)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver options

        # Create solver
        self.solver_single = ca.nlpsol("solver", "ipopt", nlp, self.nlp_opts)




    def define_nlp_multi(self):
        # Decision variables
        U = ca.MX.sym("U", self.nu, self.Np)             # control inputs over the horizon
        X = ca.MX.sym("X", self.nx, self.Np+1)             # state trajectory over the horizon

        # Parameters (initial state, disturbances, and other parameters)
        X0 = ca.MX.sym("X0", self.nx)
        D  = ca.MX.sym("D", self.nd, self.Np)
        P  = ca.MX.sym("P", self.n_params)

        # Initialize cost and constraints
        J = 0
        g = []

        hour_conversion = 3600 / self.dt

        # Initial condition constraint: ensure the first shooting node matches the initial state
        g.append(X[:, 0] - X0)

        # Loop over prediction horizon for dynamics and cost
        for k in range(self.Np):
            Uk = U[:, k]
            Dk = D[:, k]

            # Integrate dynamics from current state shooting node
            res = self.F(x0=X[:, k], u=Uk, p=ca.vertcat(Dk, P))
            X_next = res["xf"]

            # Add dynamic constraints: the next state decision variable must equal the integration result
            g.append(X[:, k+1] - X_next)

            # Accumulate stage cost
            J += 0.09 * P[108] * 1e-3 * Uk[0] / hour_conversion + \
                0.2  * P[172] * 1e-3 * Uk[4] / hour_conversion + \
                0.3  * Uk[1] * 1e-6 * self.dt

        # Terminal cost: revenue from selling tomatoes (using the final state shooting node)
        J += - (X[25, self.Np] - X0[25]) * 1e-6 / 0.08 * 1.2

        # Concatenate decision variables (stacking state and control trajectories)
        w = ca.vertcat(ca.vec(X), ca.vec(U))

        # Concatenate constraints into a single vector
        g_all = ca.vertcat(*g)

        # Parameters for the NLP (initial state, disturbances, parameters)
        p_nlp = ca.vertcat(X0, ca.vec(D), P)

        # Define the NLP problem
        nlp = {"x": w, "f": J, "g": g_all, "p": p_nlp}

        # Create and store the solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp, self.nlp_opts)
