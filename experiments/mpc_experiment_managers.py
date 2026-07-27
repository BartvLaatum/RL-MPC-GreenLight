import os
import time
import argparse
from tqdm import tqdm
import time

import numpy as np
import casadi as ca

from controllers.mpc import MPC
from environments.utils import load_weather_data, load_dummy_weather, init_state, plant_state_for_day
from model.parameters import init_default_params


class MPCExperimentManager:
    def __init__(
            self,
            mpc: MPC,
            n_days: int,
            location: str,
            growth_year: int,
            start_day: int,
            plant_state_env: bool = False,
        ):

        self.mpc = mpc
        self.L = n_days*86400
        self.t = np.arange(0, self.L, mpc.dt)
        self.N = len(self.t)
        self.exec_time = np.zeros((self.N, 1))
        self.hour_conversion = 3600/mpc.dt
        self.location = location
        self.growth_year = growth_year
        self.start_day = start_day
        self.pred_horizon = mpc.horizon/24
        self.plant_state_env = plant_state_env
        self.EPI = np.zeros((self.N, 1))
        self.penalties = np.zeros((self.N, 1))
        self.rewards = np.zeros((self.N, 1))
        self.solver_failure = np.zeros((self.N, 1)) 

        # Load or define your disturbance trajectory
        # self.d_values = load_dummy_weather(
        #     season_length=n_days,
        #     start_day=0,
        #     dt=mpc.dt,
        #     pred_horizon=mpc.Np,
        # )
        self.d_values = load_weather_data(
            weather_data_dir=f"weather/",
            location=self.location,
            growth_year=self.growth_year,
            start_day=self.start_day,
            n_days=n_days,
            pred_horizon=self.pred_horizon,
            h=mpc.dt,
            nd=10
        )

        self.x0 = init_state(self.d_values[0], 85.0)
        if self.plant_state_env:
            plant_weight, tCanSum = plant_state_for_day(self.start_day)
            self.x0[23] = plant_weight*0.2
            self.x0[24] = plant_weight*0.15
            self.x0[25] = plant_weight*0.65
            self.x0[26] = tCanSum

        self.X = np.zeros((mpc.nx, self.N+1))
        self.U = np.zeros((mpc.nu, self.N+1))

        self.X[:, 0] = self.x0
        self.U[:, 0] = np.ones(mpc.nu)*0.5
        # Initial state
        self.p = init_default_params(mpc.n_params)

    def get_x_init(self, ll, U_init=None):
        # Retrieve dimensions
        nx = self.mpc.nx
        nu = self.mpc.nu
        Np = self.mpc.Np
        # Initial guess for control inputs
        if U_init is None:
            U_init = 0.5 * np.ones((nu, Np))

        # Build an initial guess for the state trajectory by propagating X0.
        X_init = np.zeros((nx, Np+1))
        # Set the initial state from the current state at time ll.
        X_init[:, 0] = self.X[:, ll]  # assuming self.X[:, ll] is a numpy array

        # Propagate the dynamics using the initial guess for controls.
        for k in range(Np):

            # Use the k-th control guess
            u_k = U_init[:, k]

            # Use the k-th disturbance from d_values (ensure proper shape)
            D_k = self.d_values[ll + k, :]

            # Build the parameter vector for F as: [D_k; self.p]
            p_dyn = ca.vertcat(ca.DM(D_k), self.p)
            # Propagate the state: note that F returns a dictionary with key "xf"
            res = self.mpc.F(x0=ca.DM(X_init[:, k]), u=ca.DM(u_k), p=p_dyn)
            # Extract the next state; convert to a 1D NumPy array.
            X_next = res["xf"].full().flatten()
            X_init[:, k+1] = X_next

        # Now, form the overall decision vector initial guess by concatenating X_init and U_init.
        return X_init.T.flatten()

    def solve_nmpc(self):
        """
        Solve the nonlinear MPC problem.

        Args:
            p (Dict[str, Any]): the model parameters

        Returns:
            np.ndarray: the optimal control inputs
            float: the cost value
            np.ndarray: the constraints
            Dict[str, Any]: the optimization output
            np.ndarray: the control input changes
            np.ndarray: the gradient of the cost function
            np.ndarray: the hessian of the cost function
        """

        u_min = [0.0] * (self.mpc.nu * self.mpc.Np)   # Lower bounds
        u_max = [1.0] * (self.mpc.nu * self.mpc.Np)   # Upper bounds

        s_min = [-ca.inf] * (self.mpc.ns * self.mpc.Np)   # Lower bounds
        s_max = [ca.inf] * (self.mpc.ns * self.mpc.Np)   # Upper bounds

        w_min = u_min + s_min
        w_max = u_max + s_max


        u_init = np.ones((self.mpc.nu, self.mpc.Np))*0.5
        s_init = np.zeros((self.mpc.ns, self.mpc.Np))
        s_init.T.flatten()
        w_init = np.concatenate([u_init.T.flatten(), s_init.T.flatten()])

        for ll in tqdm(range(self.N)):
            t = time.time()
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
            p_all = ca.vertcat(self.X[:, ll], self.U[:, ll], reshape_d, self.p)
            # Set up initial guess and bounds for decision variables
            # Set up constraints bounds

            # g_min = [*-self.mpc.du_max]*self.mpc.Np  # Lower bounds on constraints
            # g_max = [*self.mpc.du_max]*self.mpc.Np   # Upper bounds on constraints
            solution = self.mpc.solver_single(
                x0=ca.DM(w_init),
                lbx=ca.DM(w_min),
                ubx=ca.DM(w_max),
                lbg=ca.DM(self.mpc.lbg),
                ubg=ca.DM(self.mpc.ubg),
                p=p_all
            )

            # Extract the optimal control inputs from the solution
            w_opt = solution["x"].full().flatten()
            us_opt = w_opt[:self.mpc.nu*self.mpc.Np].reshape(self.mpc.Np, self.mpc.nu).T
            s_opt = w_opt[-(self.mpc.ns*self.mpc.Np):].reshape(self.mpc.Np, self.mpc.ns).T     

            # Update the initial guess for the next iteration; using the rolled previous solution 
            u_init = np.concatenate([us_opt[:, 1:].T.flatten(), us_opt[:, -1].T.flatten()])
            s_init = np.concatenate([s_opt[:, 1:].T.flatten(), s_opt[:, -1].T.flatten()])
            w_init = np.concatenate([u_init, s_init])

            # simulate the next time step
            self.U[:, ll+1] = us_opt[:, 0]
            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()
            self.exec_time[ll, :] = time.time()-t

            # Compute closed-loop performance
            self.EPI[ll, :] = \
                (self.X[25, ll+1]-self.X[25, ll])* 1e-6 / self.mpc.reward_params["dmfm"] * self.mpc.reward_params["fruit_price"] - \
                (self.mpc.reward_params["heating_price"] * self.p[108]/self.p[46] * 1e-3 * us_opt[:, 0][0]/self.hour_conversion + \
                self.mpc.reward_params["elec_price"] * self.p[172] * 1e-3 * us_opt[:, 0][4]/self.hour_conversion + \
                self.mpc.reward_params["co2_price"] * us_opt[:, 0][1]* self.p[109]/self.p[46] * 1e-6 * self.mpc.dt)
            self.penalties[ll, :] = self.mpc.compute_penalties(self.X[:,ll+1])

            self.rewards[ll, :] = self.EPI[ll, :] - self.penalties[ll, :]


        print(f"Average solver time per iteration: {np.mean(self.exec_time)} (s)")

    def solve_nmpc_multi(self):
        """
        Solve the nonlinear MPC problem using the multiple-shooting formulation.

        Args:
            p (Dict[str, Any]): the model parameters

        Returns:
            np.ndarray: the optimal control inputs
            float: the cost value
            np.ndarray: the constraints
            Dict[str, Any]: the optimization output
            np.ndarray: the control input changes
            np.ndarray: the gradient of the cost function
            np.ndarray: the hessian of the cost function
        """
        u_min = [0.0] * (self.mpc.nu * self.mpc.Np)   # Lower bounds
        u_max = [1.0] * (self.mpc.nu * self.mpc.Np)   # Upper bounds

        s_min = [-ca.inf] * (self.mpc.ns * self.mpc.Np)   # Lower bounds
        s_max = [ca.inf] * (self.mpc.ns * self.mpc.Np)   # Upper bounds

        x_min = [-ca.inf] * (self.mpc.nx * (self.mpc.Np+1))   # Lower bounds
        x_max = [ca.inf] * (self.mpc.nx * (self.mpc.Np+1))   # Upper bounds

        w_min = u_min + s_min + x_min
        w_max = u_max + s_max + x_max

        u_init = np.ones((self.mpc.nu, self.mpc.Np))*0.5

        x_init = self.get_x_init(0, u_init)

        s_init = np.zeros((self.mpc.ns, self.mpc.Np))
        s_init.T.flatten()
        w_init = np.concatenate([u_init.T.flatten(), x_init.T.flatten(), s_init.T.flatten()])

        for ll in tqdm(range(self.N)):
            t = time.time()
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
            p_all = ca.vertcat(self.X[:, ll], self.U[:, ll], reshape_d, self.p)
            # Set up initial guess and bounds for decision variables
            # Set up constraints bounds

            # g_min = [*-self.mpc.du_max]*self.mpc.Np  # Lower bounds on constraints
            # g_max = [*self.mpc.du_max]*self.mpc.Np   # Upper bounds on constraints
            solution = self.mpc.solver_multi(
                x0=ca.DM(w_init),
                lbx=ca.DM(w_min),
                ubx=ca.DM(w_max),
                lbg=ca.DM(self.mpc.lbg),
                ubg=ca.DM(self.mpc.ubg),
                p=p_all
            )
            # Extract the optimal decision variables from the solution
            w_opt = solution["x"].full().flatten()

            us_opt = w_opt[:self.mpc.nu*self.mpc.Np].reshape(self.mpc.Np, self.mpc.nu).T
            xs_opt = w_opt[self.mpc.nu*self.mpc.Np:self.mpc.nu*self.mpc.Np+self.mpc.nx*(self.mpc.Np+1)].reshape(self.mpc.Np+1, self.mpc.nx).T
            s_opt = w_opt[-(self.mpc.ns*self.mpc.Np):].reshape(self.mpc.Np, self.mpc.ns).T

            # Update the initial guess for the next iteration; using the rolled previous solution 
            u_init = np.concatenate([us_opt[:, 1:].T.flatten(), us_opt[:, -1].T.flatten()])
            x_init = np.concatenate([xs_opt[:, 1:].T.flatten(), xs_opt[:, -1].T.flatten()])
            s_init = np.concatenate([s_opt[:, 1:].T.flatten(), s_opt[:, -1].T.flatten()])

            w_init = np.concatenate([u_init, x_init, s_init])

            # simulate the next time step
            self.U[:, ll+1] = us_opt[:, 0]
            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()
            self.exec_time[ll, :] = time.time()-t

            # costs
            self.EPI[ll, :] = \
                (
                    self.X[25, ll+1]-self.X[25, ll])* 1e-6 / self.mpc.reward_params["dmfm"] * self.mpc.reward_params["fruit_price"] - \
                    (self.mpc.reward_params["heating_price"] * self.p[108]/self.p[46] * 1e-3 * us_opt[:, 0][0]/self.hour_conversion + \
                    self.mpc.reward_params["elec_price"] * self.p[172] * 1e-3 * us_opt[:, 0][4]/self.hour_conversion + \
                    self.mpc.reward_params["co2_price"] * us_opt[:, 0][1]* self.p[109]/self.p[46] * 1e-6 * self.mpc.dt
                )
            self.penalties[ll, :] = self.mpc.compute_penalties(self.X[:,ll+1])

            self.rewards[ll, :] = self.EPI[ll, :] - self.penalties[ll, :]

        print(f"Average solver time per iteration: {np.mean(self.exec_time)} (s)")

    def save_data(self, save_dir):
        """Save the data to a file."""
        os.makedirs(save_dir, exist_ok=True)
        print(f"{save_dir}/control-inputs-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv")
        np.savetxt(f"{save_dir}/control-inputs-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.U.T, delimiter=",")
        np.savetxt(f"{save_dir}/solver-failure-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.solver_failure, delimiter=",")
        np.savetxt(f"{save_dir}/states-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.X.T, delimiter=",")
        np.savetxt(f"{save_dir}/times-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.exec_time, delimiter=",")
        np.savetxt(f"{save_dir}/EPI-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.EPI, delimiter=",")
        np.savetxt(f"{save_dir}/penalties-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.penalties, delimiter=",")
        np.savetxt(f"{save_dir}/rewards-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.rewards, delimiter=",")

class RLMPCExperimentManager(MPCExperimentManager):
    def __init__(
        self,
        rl_mpc: MPC,
        n_days: int,
        location: str,
        growth_year: int,
        start_day: int,
        offline_rl: bool = False,
        extend_ocp_region: bool = True,
        plant_state_env: bool = False,
        selector_mechanism: bool = False,
    ) -> None:
        super().__init__(
            rl_mpc,
            n_days,
            location,
            growth_year,
            start_day,
            plant_state_env,
        )
        self.offline_rl = offline_rl
        self.extend_ocp_region = extend_ocp_region
        self.rl_costs = []
        self.mpc_costs = []
        self.selector_mechanism = selector_mechanism

    def solve_nmpc_multi(self):
        """
        Solve the nonlinear MPC problem using the multiple-shooting formulation.

        Args:
            p (Dict[str, Any]): the model parameters

        Returns:
            np.ndarray: the optimal control inputs
            float: the cost value
            np.ndarray: the constraints
            Dict[str, Any]: the optimization output
            np.ndarray: the control input changes
            np.ndarray: the gradient of the cost function
            np.ndarray: the hessian of the cost function
        """
        # Define the lower and upper bounds for the decision variables
        u_min = [0.0] * (self.mpc.nu * self.mpc.Np)
        u_max = [1.0] * (self.mpc.nu * self.mpc.Np)
        s_min = [-ca.inf] * (self.mpc.ns * self.mpc.Np)
        s_max = [ca.inf] * (self.mpc.ns * self.mpc.Np)
        x_min = [-ca.inf] * (self.mpc.nx * (self.mpc.Np+1))
        x_max = [ca.inf] * (self.mpc.nx * (self.mpc.Np+1))
        w_min = u_min + s_min + x_min
        w_max = u_max + s_max + x_max

        self.mpc.eval_env.reset()
        # Generate roll-outs using the RL policy
        if self.offline_rl:
            logs = self.mpc.unroll_actor(horizon=self.N+self.mpc.Np+1, freeze=False)
        elif self.extend_ocp_region:
            logs = self.mpc.unroll_actor(horizon=self.mpc.Np, freeze=False)
        else:
            logs = self.mpc.unroll_actor(horizon=self.mpc.Np, freeze=True)

        # Extract the initial guesses for control inputs and states from roll-out logs
        if self.offline_rl:
            u_init = np.array(logs["u"])[:, :self.mpc.Np]
            x_init = np.array(logs["x"])[:, :self.mpc.Np+1]
        # if self.extend_ocp_region:
        #     u_init = np.array(logs["u"])
        #     x_init = np.array(logs["x"])
        else:
            u_init = np.array(logs["u"])
            x_init = np.array(logs["x"])

        # intial guess for slack variables
        s_init = np.zeros((self.mpc.ns, self.mpc.Np))
        s_init.T.flatten()

        # Concatenate the initial guesses into a single decision variable vector
        w_init = np.concatenate([u_init.T.flatten(), x_init.T.flatten(), s_init.T.flatten()])
        for ll in tqdm(range(self.N)):
            t = time.time()
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
            p_all = ca.vertcat(self.X[:, ll], self.U[:, ll], reshape_d, self.p, x_init[:, -1])
            try:
                solution = self.mpc.solver_multi(
                    x0=ca.DM(w_init),
                    lbx=ca.DM(w_min),
                    ubx=ca.DM(w_max),
                    lbg=ca.DM(self.mpc.lbg),
                    ubg=ca.DM(self.mpc.ubg),
                    p=p_all
                )

            # self.previous_solution = solution
            except Exception as e:
                print(f"Solver failed at iteration {ll}: {e}")
                self.solver_failure[ll] = 1
                # u = self.previous_solution.copy()

            # Extract the optimal decision variables from the solution
            w_opt = solution["x"].full().flatten()
            mpc_costs_trajectory = self.mpc.reconstruct_costs(w_opt, x_init[:, -1], self.p)
            self.rl_costs.append(-logs["reward_log"].flatten())
            self.mpc_costs.append(mpc_costs_trajectory.flatten())

            if self.selector_mechanism:
            # select either control inputs and states from the RL policy or the MPC solution
                if -logs["cumulative_reward"][0] < np.sum(mpc_costs_trajectory):
                    print("RL cost is better than MPC cost")
                    us_opt = logs["u"]
                    xs_opt = logs["x"]
                else:
                    us_opt = w_opt[:self.mpc.nu*self.mpc.Np].reshape(self.mpc.Np, self.mpc.nu).T
                    xs_opt = w_opt[self.mpc.nu*self.mpc.Np:self.mpc.nu*self.mpc.Np+self.mpc.nx*(self.mpc.Np+1)].reshape(self.mpc.Np+1, self.mpc.nx).T
            else:
                us_opt = w_opt[:self.mpc.nu*self.mpc.Np].reshape(self.mpc.Np, self.mpc.nu).T
                xs_opt = w_opt[self.mpc.nu*self.mpc.Np:self.mpc.nu*self.mpc.Np+self.mpc.nx*(self.mpc.Np+1)].reshape(self.mpc.Np+1, self.mpc.nx).T
            s_opt = w_opt[-(self.mpc.ns*self.mpc.Np):].reshape(self.mpc.Np, self.mpc.ns).T

            # simulate the next time step
            self.U[:, ll+1] = us_opt[:, 0]
            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()
            self.exec_time[ll, :] = time.time()-t

            # if using online rollouts, shift solution
            if self.offline_rl:
                u_init = np.array(logs["u"])[:, ll+1:self.mpc.Np+ll+1]
                x_init = np.array(logs["x"])[:, ll+1:self.mpc.Np+ll+2]

            # generate new roll-outs using the RL policy
            elif self.extend_ocp_region:
                # Set the environment state for the next roll-out
                day_of_year = self.mpc.eval_env.get_attr("day_of_year")[0]
                hour_of_day = self.mpc.eval_env.get_attr("hour_of_day")[0]

                self.mpc.eval_env.env_method(
                    "set_env_state",
                    *(xs_opt[:, -1], xs_opt[:, -2], us_opt[:, -1], ll+self.mpc.Np, hour_of_day, day_of_year)
                )
                logs = self.mpc.unroll_actor(horizon=1, freeze=False)
                u_init = np.concatenate([u_init[:, 1:], logs["u"][:, -1:]], axis=1)
                x_init = np.concatenate([x_init[:, 1:], logs["x"][:, -1:]], axis=1)
            else:
                day_of_year = self.mpc.eval_env.get_attr("day_of_year")[0] + (self.mpc.dt/86400) % 365
                hour_of_day = (self.mpc.eval_env.get_attr("hour_of_day")[0] + (self.mpc.dt/3600)) % 24
                self.mpc.eval_env.env_method(
                    "set_env_state", 
                    *(self.X[:, ll+1], self.X[:,ll], self.U[:,ll+1], ll+1, hour_of_day, day_of_year)
                )

                logs = self.mpc.unroll_actor(horizon=self.mpc.Np, freeze=True)
                u_init = np.array(logs["u"])
                x_init = np.array(logs["x"])

            # Update the initial guess for slack variables; using the rolled previous solution 
            s_init = np.concatenate([s_opt[:, 1:].T.flatten(), s_opt[:, -1].T.flatten()])
            w_init = np.concatenate([u_init.T.flatten(), x_init.T.flatten(), s_init])

            # Compute closed-loop performance and penalties
            self.EPI[ll, :] =  -(self.mpc.revenue_stage_cost(self.X[:, ll], self.X[:, ll+1])+self.mpc.economic_stage_cost(self.p, us_opt[:, 0]))
            self.penalties[ll, :] = self.mpc.compute_penalties(self.X[:,ll+1])
            self.rewards[ll, :] = self.EPI[ll, :] - self.penalties[ll, :]

        print(f"Average solver time per iteration: {np.mean(self.exec_time)} (s)")

    def save_costs(self, save_dir):
        """Save the costs to a file."""
        os.makedirs(save_dir, exist_ok=True)
        np.savetxt(f"{save_dir}/rl_costs-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.rl_costs, delimiter=",")
        np.savetxt(f"{save_dir}/mpc_costs-{int(self.mpc.dt)}dt-{self.mpc.horizon}H-{self.location}-{self.growth_year}-{self.start_day}.csv", self.mpc_costs, delimiter=",")
