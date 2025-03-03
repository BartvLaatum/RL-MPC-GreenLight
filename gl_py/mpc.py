import casadi as ca
import numpy as np

from model.aux_states import update
from model.ode import ODE
from model.utils import define_model, init_state, load_dummy_weather
from model.parameters import init_default_params


class MPC:
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
        J = 0;


        # Initialize state trajectory
        Xk = X0;

        # # Initialize c as a (n+m) vector of zeros
        hour_conversion = 3600/self.dt;

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
            z = ca.vertcat(Xk, Uk);

            # economic objective
            # convert boil power to kWh (costs for heating)
            # convert lamp electricity to kWh (costs for lighting)
            J += 0.09 * P[108] *1e-3 * Uk[0]/hour_conversion + \
                0.2 * P[172] * 1e-3 * Uk[4]/hour_conversion + \
                0.3 * Uk[1] * 1e-6 * self.dt                        # costs for CO2

        J += - (Xk[25]-X0[25])* 1e-6 / 0.08 * 1.2;              # revenue from selling tomatoes

        # Decision variables
        w = ca.vec(U);

        # Constraints (empty if no constraints)
        g_all = ca.vertcat(g);

        # Parameters for NLP
        p_nlp = ca.vertcat(X0, ca.vec(D), P)

        # Define the NLP problem
        nlp = {'x': w, 'f': J, 'g': g_all, 'p': p_nlp}

        # Create solver options
        nlp_opts = {}
        nlp_opts["ipopt.print_level"] = 2
        nlp_opts["ipopt.max_iter"] = 1000
        nlp_opts["ipopt.tol"] = 1e-4
        nlp_opts["ipopt.acceptable_tol"] = 1e-4
        nlp_opts["print_time"] = False
        nlp_opts["ipopt.jacobian_approximation"] = "finite-difference-values"
        nlp_opts["ipopt.hessian_approximation"] = "limited-memory"
        nlp_opts["print_time"] = True
        nlp_opts["ipopt.linear_solver"] = "ma57"

        # Create solver
        solver = ca.nlpsol("solver", "ipopt", nlp, nlp_opts)


def main():
    n_params = 208
    nx = 28
    nu = 6
    nd = 7
    dt = 900.

    # Prediction horizon
    Np = 95

    mpc = MPC(nx, nu, n_params, nd, dt, Np)
    mpc.define_nlp()
    # # Load or define your disturbance trajectory
    # d_values = load_dummy_weather(Np)
    # reshaped_dvalues = np.concatenate(d_values)
    # print(d_values.shape)

    # # Initial state
    # x0_init = init_state(d_values[0], 90.0, 0.0)

    # # Parameters
    # p_values = init_default_params(n_params)

    # # Concatenate parameters
    # p_all = ca.vertcat(x0_init, reshaped_dvalues, p_values)

    # print(p.shape)

    # # Set up initial guess and bounds for controls
    # w_init = [0.5] * (nu * Np)  # Initial guess
    # w_min = [0.0] * (nu * Np)   # Lower bounds
    # w_max = [1.0] * (nu * Np)   # Upper bounds

    # # Set up constraints bounds
    # g_min = []  # Lower bounds on constraints
    # g_max = []  # Upper bounds on constraints

    # # Set up solver arguments
    # print(p_all.shape)


    # # Solve the NLP

    # solution = solver(
    #     x0=ca.DM(w_init),
    #     lbx=ca.DM(w_min),
    #     ubx=ca.DM(w_max),
    #     lbg=ca.DM(g_min),
    #     ubg=ca.DM(g_max),
    #     p=p_all
    # )
    # X_all = np.zeros((nx, Np+1))
    # X_all[:, 0] = x0_init
    # print("Solving optimization problem...")
    # print("Optimization successful.")

    # # Extract the optimal control inputs from the solution
    # w_opt = solution["x"].full().flatten()  # Convert to a NumPy array

    # # Compute the optimal state trajectory
    # Xk_num = x0_init  # Start from initial state
    # print("\nOptimal state trajectory:")
    # for k in range(Np):
    #     # Extract control input
    #     uk = w_opt[k * nu:(k + 1) * nu]  # Slice the control inputs
    #     # Extract disturbance
    #     dk = reshaped_dvalues[k * nd:(k + 1) * nd]  # Slice the disturbances

    #     # Prepare integrator inputs
    #     input_args = ca.vertcat(*[dk, p_values])  # Concatenate inputs
    #     integrator_in = {
    #         "x0": Xk_num,
    #         "u": uk,
    #         "p": input_args
    #     }

    #     # Integrate to get next state
    #     res = integrator_func(**integrator_in)
    #     Xk_num = res["xf"]  # Get the next state
    #     print(Xk_num.shape)
    #     X_all[:, k + 1] = Xk_num.toarray().ravel()  # Store the result

    # # Extract optimal controls
    # print("\nOptimal control inputs:")
    # for k in range(Np):
    #     print("Time", k, ":")
    #     for j in range(nu):
    #         print("u[", j, "] =", w_opt[k * nu + j])
    #     print("---")



        # // Save optimal controls to a CSV file
        # std::ofstream control_file("data/june/controls-OL-non-diff.csv");
        # if (control_file.is_open()) {
        #     for (int k = 0; k < N; ++k) {
        #     for (int j = 0; j < nu; ++j) {
        #         control_file << w_opt[k * nu + j];
        #         if (j < nu - 1) {
        #         control_file << ",";
        #         }
        #     }
        #     control_file << "\n";
        #     }
        #     control_file.close();
        #     std::cout << "Optimal controls saved to controls-OL.csv" << std::endl;
        # } else {
        #     std::cerr << "Error opening file to save optimal controls" << std::endl;
        # }

        # // optimal states to a CSV file
        # std::ofstream states_file("data/june/states-OL-non-diff.csv");
        # if (states_file.is_open()) {
        #     for (int k = 0; k < N+1; ++k) {
        #         DM Xk = X_all(Slice(), k);
        #         for (int j = 0; j < nx; ++j) {
        #             states_file << Xk(j);
        #             if (j < nx - 1) {
        #                 states_file << ",";
        #             }
        #         }
        #         states_file << "\n";}
        #     states_file.close();
        #     std::cout << "Optimal states saved to states-OL.csv" << std::endl;
        # } else {
        #     std::cerr << "Error opening file to save optimal states" << std::endl;
        # }

        # // Print solution statistics
        # std::cout << "\nSolver statistics:" << std::endl;
        # std::cout << "Objective value: " << double(solution.at("f")) << std::endl;


if __name__ == "__main__":
    main()
