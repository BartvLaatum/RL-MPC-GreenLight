import casadi as ca
import numpy as np

import plot_config

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
    ):
        self.nx = nx
        self.nu = nu
        self.n_params = n_params
        self.nd = nd
        self.Np = Np
        self.dt = dt

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
            z = ca.vertcat(Xk, Uk)

            # economic objective
            # convert boil power to kWh (costs for heating)
            # convert lamp electricity to kWh (costs for lighting)
            J += 0.09 * P[108] *1e-3 * Uk[0]/hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
                0.3 * Uk[1] * 1e-6 * self.dt                        # costs for CO2

        J += - (Xk[25]-X0[25])* 1e-6 / 0.08 * 1.2              # revenue from selling tomatoes

        # Decision variables
        w = ca.vec(U)

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(g)

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, ca.vec(D), P)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver options
        nlp_opts = {}
        nlp_opts["ipopt.print_level"] = 1
        nlp_opts["ipopt.max_iter"] = 5000
        nlp_opts["ipopt.tol"] = 1e-4
        nlp_opts["ipopt.acceptable_tol"] = 1e-4
        nlp_opts["print_time"] = True
        # nlp_opts["ipopt.jacobian_approximation"] = "finite-difference-values"
        # nlp_opts["ipopt.hessian_approximation"] = "limited-memory"
        nlp_opts["ipopt.linear_solver"] = "ma57"

        # Create solver
        self.solver = ca.nlpsol("solver", "ipopt", nlp, nlp_opts)
