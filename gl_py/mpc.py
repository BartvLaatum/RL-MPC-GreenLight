from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np


# from model.aux_states import update
# from model.ode import ODE
from model.utils import define_model, co2dens2ppm, vaporPres2rh #, init_state, load_dummy_weather
# from model.parameters import init_default_params
# from model.utils import load_dummy_weather, init_state
# from model.parameters import init_default_params


class MPC:
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
    ):
        self.nx = nx
        self.nu = nu
        self.ns = ns
        self.n_params = n_params
        self.nd = nd
        self.Np = Np
        self.dt = dt
        self.nlp_opts = nlp_opts

        self.n_vars = nx + nu + ns

        # defining model dynamics
        self.F = define_model(self.nx, self.nu, self.nd, self.n_params, self.dt)
        self.constraints()

    def constraints(self):
        self.u_min = np.zeros(self.nu)
        self.u_max = np.ones(self.nu)
        self.du_max = self.u_max * 0.1

        self.pen_w = np.array([5e-5, 5e-3, 7e-4])
        self.y_min = np.array([400, 15, 0])
        self.y_max = np.array([1600, 25, 90])


    def set_slack_variables(
        self,
        ll: int,
        Xk: ca.MX,
        S: ca.MX
    ) -> Tuple[List[ca.MX], List[float], List[float]]:
        """
        Define slack variable constraints without using Opti stack.

        Returns:
            constraints (List[ca.SX]): List of constraint expressions.
            lbg (List[float]): Lower bounds for the constraints.
            ubg (List[float]): Upper bounds for the constraints.
        """
        constraints = []
        lbg = []
        ubg = []

        for i in range(self.ns):
            constraints.append(S[i, ll])
            lbg.append(0)
            ubg.append(ca.inf)

        # # # CO2 Lower Bound Penalty
        co2ppm = co2dens2ppm(Xk[2], Xk[0]*1e-6)
        expr = S[0, ll] - self.pen_w[0] * (self.y_min[0] - co2ppm)
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # CO2 Upper Bound Penalty
        expr = S[1, ll] - self.pen_w[0] * (co2ppm - self.y_max[0])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Temperature Lower Bound Penalty
        expr = S[2, ll] - self.pen_w[1] * (self.y_min[1] - Xk[2])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Temperature Upper Bound Penalty
        expr = S[3, ll] - self.pen_w[1] * (Xk[2] - self.y_max[1])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # Humidity Lower Bound Penalty
        rh = vaporPres2rh(Xk[2], Xk[15])
        expr = S[4, ll] - self.pen_w[2] * (self.y_min[2] - rh)
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        # # Humidity Upper Bound Penalty
        expr = S[5, ll] - self.pen_w[2] * (rh - self.y_max[2])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        return S, constraints, lbg, ubg

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
                0.09 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
                0.1 * Uk[1]* P[109] * 1e-6 * self.dt                        # costs for CO2


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

        # Create solver options

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

        # # Initialize c as a (n+m) vector of zeros
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
