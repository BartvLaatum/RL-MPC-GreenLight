import numpy as np
import casadi as ca
import os

from mpc import MPC
from model.utils import load_dummy_weather, init_state
from model.parameters import init_default_params
import matplotlib.pyplot as plt

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
        # Load or define your disturbance trajectory
        self.d_values = load_dummy_weather(self.N+mpc.Np, mpc.dt, month=month)

        self.x0 = init_state(self.d_values[0], 85.0, 0.0)
        self.X = np.zeros((mpc.nx, self.N+1))
        self.U = np.zeros((mpc.nu, self.N+1))

        self.X[:, 0] = self.x0
        # Initial state
        self.p = init_default_params(mpc.n_params)
        self.plot_weather()

    def get_init(self, ll):
        # Retrieve dimensions
        nx = self.mpc.nx
        nu = self.mpc.nu
        Np = self.mpc.Np

        # Initial guess for control inputs
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
        w_init = np.concatenate([X_init.T.flatten(), U_init.flatten()])
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
            solution = self.mpc.solver(
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
            w_init = w_opt.flatten()
            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()
        self.plot_control_trajectories(self.U, self.mpc.dt)
        self.plot_states(self.X, self.mpc.dt)

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
        for ll in range(self.N):
            print(f"Solving for timestep: {ll}")
            # Build the disturbance vector for the prediction horizon.
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :], axis=0)
            # The parameter vector now must include the current state, the disturbances over the horizon, and other parameters.
            # The NLP was defined with p = [X0; vec(D); P], where X0 is the current state.
            p_all = ca.vertcat(self.X[:, ll], reshape_d, self.p)

            # Retrieve dimensions
            nx = self.mpc.nx
            nu = self.mpc.nu
            Np = self.mpc.Np

            # --- Initial Guess ---
            # For the state trajectory, a reasonable guess is to assume the state remains constant.
            X_init = np.tile(self.X[:, ll], (Np+1, 1))  # Shape: (nx, Np+1)

            # For control inputs, use a constant initial guess.
            U_init = 0.0 * np.ones((nu, Np))
            # Concatenate the initial guesses (flattened)
            w_init = self.get_init(ll)

            # --- Bounds for Decision Variables ---
            # For states, if there are no explicit state bounds, we use large numbers.
            X_lb = -np.inf * np.ones((nx, Np+1))
            X_ub =  np.inf * np.ones((nx, Np+1))
            # For control inputs, we use the original bounds [0, 1].
            U_lb = np.zeros((nu, Np))
            U_ub = np.ones((nu, Np))

            # Combine bounds for states and controls.
            w_lb = np.concatenate([X_lb.flatten(), U_lb.flatten()])
            w_ub = np.concatenate([X_ub.flatten(), U_ub.flatten()])

            # --- Constraint Bounds ---
            # There are equality constraints enforcing the initial state and dynamic consistency:
            #   - Initial condition: X[:, 0] - X0 = 0   (nx constraints)
            #   - Dynamic constraints: X[:, k+1] - F(X[:, k], U[:, k]) = 0 for each k (nx constraints per interval)
            # Total constraints = nx + (nx * Np) = nx * (Np + 1)
            n_con = nx * (Np + 1)
            g_lb = np.zeros(n_con)
            g_ub = np.zeros(n_con)

            # --- Solve the NLP ---
            solution = self.mpc.solver(
                x0=ca.DM(w_init),
                lbx=ca.DM(w_lb),
                ubx=ca.DM(w_ub),
                lbg=ca.DM(g_lb),
                ubg=ca.DM(g_ub),
                p=p_all
            )

            # --- Process the Solution ---
            w_opt = solution["x"].full().flatten()
            # Extract the state trajectory and control inputs from the decision vector.
            X_opt = w_opt[:nx*(Np+1)].reshape((nx, Np+1))
            U_opt = w_opt[nx*(Np+1):].reshape((nu, Np))

            # Apply the first control input to the system.
            self.U[:, ll+1] = U_opt[:, 0]
            # Update the state using the predicted state from the multiple-shooting trajectory.
            self.X[:, ll+1] = X_opt[:, 1]

            # Optionally, one could extract additional solver outputs (e.g., cost, gradients, etc.)
            # cost_value = solution["f"]
            # constraints = solution["g"]
            # Additional outputs could be processed as needed.

    def save_data(self):
        """Save the data to a file."""
        dir = f"results/{self.method}/{self.month}"
        os.makedirs(dir, exist_ok=True)
        np.savetxt(f"{dir}/control-inputs-{self.mpc.dt}dt.csv", self.U.T, delimiter=",")
        np.savetxt(f"{dir}/states-{self.mpc.dt}dt.csv", self.X.T, delimiter=",")

    def plot_control_trajectories(self, U, dt):
        """Plot the optimized control trajectories."""
        control_labels_with_units = {
            0: r"$u_{heat}$",
            1: r"$u_{CO_2}$",
            2: r"$u_{ThScr}$",
            3: r"$u_{vent}$",
            4: r"$u_{light}$",
            5: r"$u_{BlScr}$",
        }
        WIDTH = 175 * 0.03937
        HEIGHT = WIDTH * 0.75

        fig, axes = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True, sharey=True)

        t = np.arange(0, U.shape[-1]*self.mpc.dt, self.mpc.dt)/86400

        axes[0,0].step(t[:-1], U[0, 1:], color="C0", where='post')
        axes[0,1].step(t[:-1], U[1, 1:], color="C1", where='post')
        axes[1,0].step(t[:-1], U[2, 1:], color="C2", where='post')
        axes[1,1].step(t[:-1], U[3, 1:], color="C3", where='post')
        axes[2,0].step(t[:-1], U[4, 1:], color="C4", where='post')
        axes[2,1].step(t[:-1], U[5, 1:], color="C5", where='post')

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(control_labels_with_units[i])

        fig.supxlabel('Time (days)')
        fig.supylabel('Control Input')
        fig.suptitle('Closed-loop Control Trajectories')
        fig.tight_layout()
        fig.savefig(f"figures/{self.method}/{self.month}/control-inputs-{int(self.mpc.dt)}dt.png")

    def plot_states(self, X, dt):
        """Plot the optimized control trajectories."""
        state_labels_with_units = {
            0: r"Air Temperature ($^\circ$C)",
            1: r"CO$_2$ (mg/m$^3$)",
            2: r"Vapor Pressure (Pa)",
            3: r"Fruit Weight (DM mg/m$^2$)"
        }
        WIDTH = 175 * 0.03937
        HEIGHT = WIDTH * 0.75

        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True)

        t = np.arange(0, X.shape[-1]*self.mpc.dt, dt)/86400

        axes[0,0].step(t, X[2,:], color="C0", where='post')
        axes[0,1].step(t, X[0, :], color="C0", where='post')
        axes[1,0].step(t, X[15, :], color="C0", where='post')
        axes[1,1].step(t, X[25, :], color="C0", where='post')

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(state_labels_with_units[i])

        fig.supxlabel('Time (days)')
        fig.supylabel('State variable')
        fig.suptitle('Closed-loop State Trajectories')
        fig.tight_layout()
        fig.savefig(f"figures/{self.method}/{self.month}/states-{int(self.mpc.dt)}dt.png")
        # plt.show()

    def plot_weather(self,):
        """Plot the optimized control trajectories."""
        state_labels_with_units = {
            0: r"Global Radiation (W/m$^2$)",
            1: r"Air Temperature ($^\circ$C)",
            2: r"CO$_2$ (mg/m$^3$)",
            3: r"Vapor Pressure (Pa)",
            4: r"Wind Speed (m/s)",
            5: r"Sky Temperature ($^\circ$C)",
        }
        WIDTH = 175 * 0.03937
        HEIGHT = WIDTH * 0.75

        fig, axes = plt.subplots(3, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True)

        t = np.arange(0, self.N*self.mpc.dt, self.mpc.dt)/86400

        axes[0,0].step(t, self.d_values[:self.N, 0], color="C0", where='post')
        axes[0,1].step(t, self.d_values[:self.N, 1], color="C0", where='post')
        axes[1,0].step(t, self.d_values[:self.N, 2], color="C0", where='post')
        axes[1,1].step(t, self.d_values[:self.N, 3], color="C0", where='post')
        axes[2,0].step(t, self.d_values[:self.N, 4], color="C0", where='post')
        axes[2,1].step(t, self.d_values[:self.N, 5], color="C0", where='post')

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(state_labels_with_units[i])

        fig.supxlabel('Time (days)')
        fig.supylabel('Variable')
        fig.suptitle('Weather disturbance')
        fig.tight_layout()
        fig.savefig(f"figures/{self.method}/{self.month}/weather-{int(self.mpc.dt)}dt.png")
        # plt.show()


def main():
    n_params = 208
    nx = 28
    nu = 6
    nd = 7
    dt = 300.
    n_days = 1.
    month = "june"
    Np = 12

    method = "exact"

    mpc = MPC(nx, nu, n_params, nd, dt, Np)
    mpc.define_nlp()
    exp = Experiment(mpc, month, n_days, method)
    exp.solve_nmpc()
    exp.save_data()

if __name__ == "__main__":
    main()
