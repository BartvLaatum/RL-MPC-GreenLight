import argparse

import os
import time

import numpy as np
import casadi as ca

from mpc import MPC
from model.utils import load_dummy_weather, init_state, convert_rh_ppm
from model.parameters import init_default_params

from visualizations.trajectories import plot_control_trajectories, plot_states

import time

class Experiment:
    def __init__(
            self,
            mpc: MPC,
            month,
            n_days,
            method,
        ):

        self.mpc = mpc
        self.month = month
        self.L = n_days*86400
        self.t = np.arange(0, self.L, mpc.dt)
        self.N = len(self.t)
        self.method = method
        self.exectime = np.zeros((1, self.N))
        # Load or define your disturbance trajectory
        self.d_values = load_dummy_weather(self.N+mpc.Np, mpc.dt, month=month)

        self.x0 = init_state(self.d_values[0], 85.0, 0.0)
        self.X = np.zeros((mpc.nx, self.N+1))
        self.U = np.zeros((mpc.nu, self.N+1))

        self.X[:, 0] = self.x0
        self.U[:, 0] = np.ones(mpc.nu)*0.5
        # Initial state
        self.p = init_default_params(mpc.n_params)
        # self.plot_weather()

    def get_init(self, ll, U_init=None):
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

        print(X_init.T.flatten().shape)
        print(U_init.T.flatten().shape)
        # Now, form the overall decision vector initial guess by concatenating X_init and U_init.
        w_init = np.concatenate([X_init.T.flatten(), U_init.T.flatten()])
        return w_init

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

        w_min = [0.0] * (self.mpc.nu * self.mpc.Np)   # Lower bounds
        w_max = [1.0] * (self.mpc.nu * self.mpc.Np)   # Upper bounds
        w_init = np.ones((self.mpc.nu*self.mpc.Np))*0.5

        for ll in range(self.N):
            print(f"Solving for timestep: {ll}")
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
            p_all = ca.vertcat(self.X[:, ll], reshape_d, self.p)

            # Set up initial guess and bounds for decision variables
            # Set up constraints bounds
            g_min = []  # Lower bounds on constraints
            g_max = []  # Upper bounds on constraints
            solution = self.mpc.solver_single(
                x0=ca.DM(w_init),
                lbx=ca.DM(w_min),
                ubx=ca.DM(w_max),
                lbg=ca.DM(g_min),
                ubg=ca.DM(g_max),
                p=p_all
            )
            # Extract the optimal control inputs from the solution
            w_opt = solution["x"].full().flatten().reshape(self.mpc.Np, self.mpc.nu).T  # Convert to a NumPy array
            self.U[:, ll+1] = w_opt[:, 0]
            w_init = w_opt.T.flatten()
            w_init = np.concatenate([w_init[self.mpc.nu:], w_init[-self.mpc.nu:]])
            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()

    # def get_init_multi(self, ll, U_init=None):
    #     print("obtaining single shooting guess")
        
    #     # Retrieve dimensions
    #     nx = self.mpc.nx
    #     nu = self.mpc.nu
    #     Np = self.mpc.Np
        
    #     w_min = [0.0] * (self.mpc.nu * self.mpc.Np)   # Lower bounds
    #     w_max = [1.0] * (self.mpc.nu * self.mpc.Np)   # Upper bounds

    #     w_init = np.ones((self.mpc.nu*self.mpc.Np))*0.5
    #     g_min = []  # Lower bounds on constraints
    #     g_max = []  # Upper bounds on constraints
    #     reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
    #     p_all = ca.vertcat(self.X[:, ll], reshape_d, self.p)

    #     solution = self.mpc.solver_single(
    #         x0=ca.DM(w_init),
    #         lbx=ca.DM(w_min),
    #         ubx=ca.DM(w_max),
    #         lbg=ca.DM(g_min),
    #         ubg=ca.DM(g_max),
    #         p=p_all
    #     )

    #     U_init = solution["x"].full().flatten().reshape(self.mpc.Np, self.mpc.nu).T  # Convert to a (Np, nu) array

    #     X_init = np.zeros((nx, Np+1))
    #     # Set the initial state from the current state at time ll.
    #     X_init[:, 0] = self.X[:, ll]  # assuming self.X[:, ll] is a numpy array

    #     # Propagate the dynamics using the initial guess for controls.
    #     for k in range(Np):
    #         # Use the k-th control guess
    #         u_k = U_init[:, k]
    #         # Use the k-th disturbance from d_values (ensure proper shape)
    #         D_k = self.d_values[ll + k, :]  
    #         # Build the parameter vector for F as: [D_k; self.p]
    #         p_dyn = ca.vertcat(ca.DM(D_k), self.p)
    #         # Propagate the state: note that F returns a dictionary with key "xf"
    #         res = self.mpc.F(x0=ca.DM(X_init[:, k]), u=ca.DM(u_k), p=p_dyn)
    #         # Extract the next state; convert to a 1D NumPy array.
    #         X_next = res["xf"].full().flatten()
    #         X_init[:, k+1] = X_next

    #     w_init = np.concatenate([X_init.T.flatten(), U_init.T.flatten()])
    #     return w_init


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

        times = []
        for ll in range(self.N):
            print(f"Solving for timestep: {ll}")
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
            times.append(time.time()-t)

        print(f"Average solver time per iteration: {np.mean(times)} (s)")

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

        # x_init = np.zeros((self.mpc.nx, self.mpc.Np+1))

        s_init = np.zeros((self.mpc.ns, self.mpc.Np))
        s_init.T.flatten()
        w_init = np.concatenate([u_init.T.flatten(), x_init.T.flatten(), s_init.T.flatten()])

        times = []
        for ll in range(self.N):
            print(f"Solving for timestep: {ll}")
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
            times.append(time.time()-t)

        print(f"Average solver time per iteration: {np.mean(times)} (s)")
        # self.plot_control_trajectories(self.U, self.mpc.dt)
        # self.plot_states(self.X, self.mpc.dt)

            # Optionally, one could extract additional solver outputs (e.g., cost, gradients, etc.)
            # cost_value = solution["f"]
            # constraints = solution["g"]
            # Additional outputs could be processed as needed.


    def save_data(self, approach):
        """Save the data to a file."""
        dir = f"results/{self.method}/{approach}/{self.month}"
        os.makedirs(dir, exist_ok=True)
        np.savetxt(f"{dir}/control-inputs-cs-{int(self.mpc.dt)}dt.csv", self.U.T, delimiter=",")
        np.savetxt(f"{dir}/states-cs-{int(self.mpc.dt)}dt.csv", self.X.T, delimiter=",")

def solver_opts(method):
    nlp_opts = {}
    nlp_opts["ipopt.print_level"] = 2
    nlp_opts["ipopt.warm_start_init_point"] = "yes"
    nlp_opts["ipopt.max_iter"] = 1000
    nlp_opts["ipopt.tol"] = 1e-2
    nlp_opts["ipopt.acceptable_tol"] = 1e-2
    nlp_opts["print_time"] = True
    nlp_opts["ipopt.linear_solver"] = "ma57"

    if method == "finite-difference":
        nlp_opts["ipopt.jacobian_approximation"] = "finite-difference-values"
        nlp_opts["ipopt.hessian_approximation"] = "limited-memory"

    elif method == "exact":
        nlp_opts["ipopt.jacobian_approximation"] = "exact"
        nlp_opts["ipopt.hessian_approximation"] = "limited-memory"
    return nlp_opts

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["exact", "finite-difference"], required=True)
    parser.add_argument("--approach", type=str, choices=["single", "multi"], required=True)
    args = parser.parse_args()

    n_params = 208
    nx = 28
    nu = 6
    ns = 6
    nd = 7
    dt = 300.
    n_days = 1
    month = "june"
    Np = 12

    print(f"Running {args.method} method")
    print(f"Using {args.approach}-shooting approach")
    nlp_opts = solver_opts(args.method)

    mpc = MPC(nx, nu, ns, n_params, nd, dt, Np, nlp_opts)
    exp = Experiment(mpc, month, n_days, args.method)

    if args.approach == "single":
        exp.mpc.define_nlp()
        exp.solve_nmpc()
    elif args.approach == "multi":
        mpc.define_nlp_multi()
        exp.solve_nmpc_multi()

    exp.X = convert_rh_ppm(exp.X)

    data = {
        args.method: {
            "U": exp.U,
            "X": exp.X
        }
    }

    # plot_control_trajectories(data, args.approach,dt, None)
    # plot_states(data, args.approach, dt, None)

    exp.save_data(args.approach)

if __name__ == "__main__":
    main()
