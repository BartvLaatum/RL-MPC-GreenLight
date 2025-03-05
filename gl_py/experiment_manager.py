import numpy as np
import casadi as ca

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
        ):
        self.mpc = mpc
        self.L = n_days*86400
        self.t = np.arange(0, self.L, mpc.dt)
        self.N = len(self.t)
        

        # Load or define your disturbance trajectory
        self.d_values = load_dummy_weather(self.N+mpc.Np, month=month)
        print("weather shape", self.d_values.shape)
        # reshaped_dvalues = np.concatenate(d_values)
        # print(reshaped_dvalues.shape)

        self.x0 = init_state(self.d_values[0], 85.0, 0.0)
        self.X = np.zeros((mpc.nx, self.N+1))
        self.U = np.zeros((mpc.nu, self.N+1))

        self.X[:, 0] = self.x0
        # Initial state
        self.p = init_default_params(mpc.n_params)

        # Parameters
        # p_values = init_default_params(n_params)

        # Concatenate parameters




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

        for ll in range(self.N):
            print(f"Solving for timestep: {ll}")
            reshape_d = np.concatenate(self.d_values[ll:ll+self.mpc.Np, :])
            p_all = ca.vertcat(self.X[:, ll], reshape_d, self.p)

            # Set up initial guess and bounds for decision variables
            w_init = [0.5] * (self.mpc.nu * self.mpc.Np)  # Initial guess
            w_min = [0.0] * (self.mpc.nu * self.mpc.Np)   # Lower bounds
            w_max = [1.0] * (self.mpc.nu * self.mpc.Np)   # Upper bounds

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
            w_opt = solution["x"].full().flatten().reshape(self.mpc.Np, self.mpc.nu)  # Convert to a NumPy array
            self.U[:, ll+1] = w_opt[0, :]

            res = self.mpc.F(x0=self.X[:,ll], u=self.U[:, ll+1], p=ca.vertcat(*[reshape_d[:self.mpc.nd], self.p]))
            self.X[:, ll+1] = res["xf"].toarray().ravel()
        self.plot_control_trajectories(self.U.T, self.mpc.dt)
        self.plot_states(self.X, self.mpc.dt)


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

        t = np.arange(0, U.shape[0]*900, dt)/86400

        axes[0,0].step(t, U[:, 0], color="C0")
        axes[0,1].step(t, U[:, 1], color="C1")
        axes[1,0].step(t, U[:, 2], color="C2")
        axes[1,1].step(t, U[:, 3], color="C3")
        axes[2,0].step(t, U[:, 4], color="C4")
        axes[2,1].step(t, U[:, 5], color="C5")

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(control_labels_with_units[i])

        fig.supxlabel('Time (days)')
        fig.supylabel('Control Input')
        fig.suptitle('open-loop Control Trajectories')
        fig.tight_layout()
        fig.savefig("controls-openloop.png")
        # plt.show()


    def plot_states(self, X, dt):
        """Plot the optimized control trajectories."""
        state_labels_with_units = {
            0: r"Air Temperature ($^\circ$ C)",
            1: r"CO$_2$ (mg/m$^3$)",
            2: r"Vapor Pressure (Pa)",
            3: r"Fruit Weight (DM mg/m$^2$)"
        }
        WIDTH = 175 * 0.03937
        HEIGHT = WIDTH * 0.75

        fig, axes = plt.subplots(2, 2, figsize=(WIDTH, HEIGHT), dpi=180, sharex=True)

        t = np.arange(0, X.shape[-1]*900, dt)/86400

        axes[0,0].step(t, X[2,:], color="C0")
        axes[0,1].step(t, X[0, :], color="C0")
        axes[1,0].step(t, X[15, :], color="C0")
        axes[1,1].step(t, X[25, :], color="C0")

        for i, ax in enumerate(axes.flat):
            ax.set_ylabel(state_labels_with_units[i])

        fig.supxlabel('Time (days)')
        fig.supylabel('State variable')
        fig.suptitle('Closed-loop State Trajectories')
        fig.tight_layout()
        fig.savefig("statev2.png")
        # plt.show()



def main():
    n_params = 208
    nx = 28
    nu = 6
    nd = 7
    dt = 900.
    n_days = 1
    month = 'june'
    # Prediction horizon
    Np = 48

    mpc = MPC(nx, nu, n_params, nd, dt, Np)
    mpc.define_nlp()
    exp = Experiment(mpc, month, n_days)
    exp.solve_nmpc()

if __name__ == "__main__":
    main()
