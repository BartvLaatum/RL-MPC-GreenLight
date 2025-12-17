from typing import Any, Dict, List, Tuple

import casadi as ca
import numpy as np


from environments.utils import co2dens2ppm, vaporPres2rh, define_model

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
        horizon: int,
        u_min: List[float],
        u_max: List[float],
        delta_u_max: float,
        constraints: Dict[str, Any],
        reward_params: Dict[str, Any],
        nlp_opts: Dict[str, Any],
    ):
        self.nx = nx
        self.nu = nu
        self.ns = ns
        self.n_params = n_params
        self.nd = nd
        self.horizon = horizon
        self.Np = int(horizon * 3600 / dt)
        self.dt = dt
        self.u_min = u_min
        self.u_max = u_max
        self.delta_u_max = delta_u_max
        self.reward_params = reward_params
        self.nlp_opts = nlp_opts
        self.n_vars = nx + nu + ns
        self.hour_conversion = 3600/dt
        self.pen_w = np.array(reward_params["pen_weights"])

        # defining model dynamics
        self.F = define_model(self.nx, self.nu, self.nd, self.n_params, self.dt)
        self.set_constraints(constraints)

    def set_constraints(self, constraints: Dict[str, Any]):
        self.du_max = self.delta_u_max * np.ones(self.nu)

        self.y_min = np.array([constraints["co2_min"], constraints["temp_min"], constraints["rh_min"]])
        self.y_max = np.array([constraints["co2_max"], constraints["temp_max"], constraints["rh_max"]])

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

        # CO2 Lower Bound Penalty
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

        # Humidity Upper Bound Penalty
        expr = S[5, ll] - self.pen_w[2] * (rh - self.y_max[2])
        constraints.append(expr)
        lbg.append(0)
        ubg.append(ca.inf)

        return S, constraints, lbg, ubg

    def constraint_violation(self, y: np.ndarray):
        """
        Function that computes the absolute penalty for violating state constraints.
        State constraints are currently non-dynamical, and based on the boundaries of the system.
        We do not look at dry mass bounds, since those are non-existent in real greenhouse.
        """
        lowerbound = self.y_min[:] - y[:]
        lowerbound[lowerbound < 0] = 0
        upperbound = y[:] - self.y_max[:]
        upperbound[upperbound < 0] = 0
        return lowerbound, upperbound

    def compute_penalties(self, X: np.ndarray):
        # Transform state variable to ppm and relative humidity
        y = X[[0, 2, 15]].copy()
        y[0] = co2dens2ppm(y[1], y[0]*1e-6)
        y[2] = vaporPres2rh(y[1], y[2])

        lowerbound, upperbound = self.constraint_violation(y)
        penalties = np.dot(self.pen_w, lowerbound) + np.dot(self.pen_w, upperbound)
        return np.sum(penalties)

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
            J += 0.09 * P[108]/P[46] *1e-3 * Uk[0]/self.hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/self.hour_conversion + \
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
            J += self.reward_params["heating_price"] * P[108]/P[46] * 1e-3 * Uk[0]/self.hour_conversion + \
                self.reward_params["elec_price"] * P[172] * 1e-3 * Uk[4]/self.hour_conversion + \
                self.reward_params["co2_price"] * Uk[1]* P[109]/P[46] * 1e-6 * self.dt                        # costs for CO2

            S, S_constraints, S_lbg, S_ubg = self.set_slack_variables(k, X_next, S)
            g.extend(S_constraints)
            self.lbg.extend(S_lbg)
            self.ubg.extend(S_ubg)

            J += ca.sum1(S[:, k])

        # revenue from selling tomatoes (EUR/m2/day)
        J += - (X[25, -1]-X0[25])* 1e-6 / self.reward_params["dmfm"] * self.reward_params["fruit_price"]              
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
