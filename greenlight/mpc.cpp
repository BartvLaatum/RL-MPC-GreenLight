#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#include "ode.hpp"
#include "utils.hpp"
#include "params.hpp"
#include <omp.h>

int main() {
    int np = 208;
    int nx = 28;
    int nu = 6;
    int nd = 7;
    int na = 240;
    float h = 900.;
    Function integrator_func;

    // Define the symbolic variables for CasADi
    SX p = SX::sym("p", np);
    SX x = SX::sym("x", nx);
    SX u = SX::sym("u", nu);
    SX d = SX::sym("d", nd);
    SX a = SX::sym("a", na);

    Function update_aux = Function("update_aux", {x, u, d, p}, {update(x, u, d, p)});
    SX dxdt = ODE(x, u, d, p);
    SX input_args_sym = SX::vertcat({u, d, p});

    Dict int_opts;
    int_opts["abstol"] = 1e-6;
    int_opts["reltol"] = 1e-6;

    integrator_func = integrator(
        "integrator_func", "cvodes",
        {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
        0.0, h, int_opts
    );

    // Prediction horizon
    const int N = 96;

    // Control variables (decision variables)
    MX U = MX::sym("U", nu, N);

    // Parameters (initial state, disturbances, and parameters)
    MX X0 = MX::sym("X0", nx);
    MX D = MX::sym("D", nd, N);
    MX P = MX::sym("P", np);

    // Initialize Q as a (n+m) x (n+m) matrix of zeros
    MX Q = MX::zeros(nx + nu, nx + nu);
    // Set the quadratic term for Xk(2)^2
    // Note: Xk(2) corresponds to the (2)th index (0-based)
    Q(2, 2) = 1.0;

    // Initialize cost and constraints
    MX J = 0;
    std::vector<MX> g;

    MX Uk;
    MX Dk;
    // Initialize state trajectory
    MX Xk = X0;

    // Initialize c as a (n+m) vector of zeros
    MX c = MX::zeros(nx + nu);
    MX z;
    MXDict res;

    // Set the linear term for Xk(2)
    c(2) = -36.0;
    float hour_conversion = 3600/h;
    // Set the linear terms for Uk(0),  and Uk(4)
    c(nx + 0) = 0.09 * P(108) * 1e-3 / hour_conversion; // Uk(0)
    // c(nx + 1) = 0.2 * P(172) * 1e-3 / hour_conversion;  // Uk(4)x
    c(nx + 4) = 0.2 * P(172) * 1e-3 / hour_conversion;  // Uk(4)

    // Loop over prediction horizon
    for (int k = 0; k < N; ++k) {
        Uk = U(Slice(), k);
        Dk = D(Slice(), k);

        // Concatenate state and control vectors

        // Integrate to get the next state
        res = integrator_func(MXDict{
            {"x0", Xk},
            {"p", vertcat(Uk, Dk, P)}
        });
        Xk = res.at("xf");
        // J +=
        z = vertcat(Xk, Uk);
        // Accumulate cost function (e.g., tracking error)
        // J += mtimes(z.T(), mtimes(Q, z)) + mtimes(c.T(), z);
        // economic objective
        J += 0.09 * P(108) *1e-3 * Uk(0)/hour_conversion +  // convert boil power to kWh (costs for heating)
             0.2 * P(172) * 1e-3 * Uk(4)/hour_conversion +  // convert lamp electricity to kWh (costs for lighting)
             0.3 * Uk(1) * 1e-6 * h;                        // costs for CO2
    }
    J += - (Xk(25)-X0(25))* 1e-6 / 0.08 * 1.2;              // revenue from selling tomatoes
    // Decision variables
    MX w = vec(U);

    // Constraints (empty if no constraints)
    MX g_all = vertcat(g);

    // Parameters for NLP
    MX p_nlp = MX::vertcat({X0, vec(D), P});

    // Define the NLP problem
    MXDict nlp = {{"x", w}, {"f", J}, {"g", g_all}, {"p", p_nlp}};

    // Create solver options
    Dict nlp_opts;
    nlp_opts["ipopt.print_level"] = 2;
    nlp_opts["ipopt.max_iter"] = 1000;
    nlp_opts["ipopt.tol"] = 1e-4;
    nlp_opts["ipopt.acceptable_tol"] = 1e-4;
    nlp_opts["print_time"] = false;
    nlp_opts["ipopt.jacobian_approximation"] = "finite-difference-values";
    nlp_opts["ipopt.hessian_approximation"] = "limited-memory";
    nlp_opts["print_time"] = true;
    nlp_opts["ipopt.linear_solver"] = "ma57";

    // Create solver
    Function solver = nlpsol("solver", "ipopt", nlp, nlp_opts);

    // Load or define your disturbance trajectory
    std::vector<std::vector<double>> d_values = load_dummy_weather(N);
    std::vector<double> reshaped_dvalues;
    for (const auto& d_row : d_values) {
        reshaped_dvalues.insert(reshaped_dvalues.end(), d_row.begin(), d_row.end());
    }

    // Initial state
    DM x0_init = init_state(d_values[0], 90.0, 0.0);

    // Parameters
    std::vector<double> p_values = init_default_params(np);

    // Concatenate parameters
    std::vector<DM> p_concat = {x0_init, reshaped_dvalues, p_values};
    DM p_all = vertcat(p_concat);

    // Set up initial guess for controls
    std::vector<double> w_init(nu * N, 0.5);  // Initial guess for controls

    // Set up bounds for controls
    std::vector<double> w_min(nu * N, 0.0);   // Lower bounds for controls
    std::vector<double> w_max(nu * N, 1.0);   // Upper bounds for controls

    // Set up constraints bounds (if any constraints)
    std::vector<double> g_min;  // Lower bounds on constraints
    std::vector<double> g_max;  // Upper bounds on constraints

        // Set up solver arguments
    DMDict solver_args = {
        {"x0", w_init},
        {"lbx", w_min},
        {"ubx", w_max},
        {"lbg", g_min},
        {"ubg", g_max},
        {"p", p_all}
    };

    // Solve the NLP
    DMDict solution;

    DM X_all = DM::zeros(nx, N+1);
    X_all(Slice(), 0) = x0_init;
    std::cout << "Solving optimization problem..." << std::endl;
    try {
        solution = solver(solver_args);
        std::cout << "Optimization successful." << std::endl;
        // Extract the optimal solution
        std::vector<double> w_opt = std::vector<double>(solution.at("x"));

        // Compute the optimal state trajectory
        DM Xk_num = x0_init;  // Start from initial state
        std::cout << "\nOptimal state trajectory:" << std::endl;
        for (int k = 0; k < N; ++k) {

            // Extract control input
            std::vector<double> u_k(w_opt.begin() + k * nu, w_opt.begin() + (k + 1) * nu);
            // Extract disturbance
            std::vector<double> d_k(reshaped_dvalues.begin() + k * nd, reshaped_dvalues.begin() + (k + 1) * nd);

            // Prepare integrator inputs
            DM input_args = DM::vertcat({u_k, d_k, p_values});
            std::map<std::string, DM> integrator_in;
            integrator_in["x0"] = Xk_num;
            integrator_in["p"] = input_args;

            // Integrate to get next state
            DMDict res = integrator_func(integrator_in);
            Xk_num = res.at("xf");
            X_all(Slice(), k+1) = Xk_num;

        }

        // Extract optimal controls
        std::cout << "\nOptimal control inputs:" << std::endl;
        for (int k = 0; k < N; ++k) {
            std::cout << "Time " << k << ":" << std::endl;
            for (int j = 0; j < nu; ++j) {
                std::cout << "u[" << j << "] = " << w_opt[k * nu + j] << std::endl;
            }
            std::cout << "---" << std::endl;
        }

        // Save optimal controls to a CSV file
        std::ofstream control_file("data/june/controls-OL-non-diff.csv");
        if (control_file.is_open()) {
            for (int k = 0; k < N; ++k) {
            for (int j = 0; j < nu; ++j) {
                control_file << w_opt[k * nu + j];
                if (j < nu - 1) {
                control_file << ",";
                }
            }
            control_file << "\n";
            }
            control_file.close();
            std::cout << "Optimal controls saved to controls-OL.csv" << std::endl;
        } else {
            std::cerr << "Error opening file to save optimal controls" << std::endl;
        }

        // optimal states to a CSV file
        std::ofstream states_file("data/june/states-OL-non-diff.csv");
        if (states_file.is_open()) {
            for (int k = 0; k < N+1; ++k) {
                DM Xk = X_all(Slice(), k);
                for (int j = 0; j < nx; ++j) {
                    states_file << Xk(j);
                    if (j < nx - 1) {
                        states_file << ",";
                    }
                }
                states_file << "\n";}
            states_file.close();
            std::cout << "Optimal states saved to states-OL.csv" << std::endl;
        } else {
            std::cerr << "Error opening file to save optimal states" << std::endl;
        }

        // Print solution statistics
        std::cout << "\nSolver statistics:" << std::endl;
        std::cout << "Objective value: " << double(solution.at("f")) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error while solving the optimization problem: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
