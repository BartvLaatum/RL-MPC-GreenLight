#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <vector>

#include <casadi/casadi.hpp>

#include "ode.hpp"
#include "utils.hpp"
#include "params.hpp"
#include <omp.h>
// omp_set_num_threads(4);

using namespace casadi;

// Function to load dummy weather data from a text file
std::vector<std::vector<double>> load_weather_dummy(int N) {
    std::vector<std::vector<double>> weather_data;
    std::ifstream file("weather.csv");

    if (!file.is_open()) {
        std::cerr << "Error opening weather data file" << std::endl;
        return weather_data;
    }

    std::string line;
    while (std::getline(file, line) && weather_data.size() < N) {
        std::vector<double> data_row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            data_row.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        weather_data.push_back(data_row);
    }

    file.close();
    return weather_data;
}

int main() {
    int np = 208;
    int nx = 28;
    int nu = 6;
    int nd = 7;
    int na = 240;
    float h = 900.;

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
    int_opts["compiler"] = "shell";
    int_opts["abstol"] = 1e-6;
    int_opts["reltol"] = 1e-6;

    Function integrator_func = integrator(
        "integrator_func", "cvodes",
        {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
        0.0, h, int_opts
    );

    // Prediction horizon
    const int N = 12;

    // Decision variables: controls and states
    MX U = MX::sym("U", nu, N);          // Control variables
    MX X = MX::sym("X", nx, N + 1);      // State variables

    // Parameters (initial state, disturbances, and parameters)
    MX X0 = MX::sym("X0", nx);
    MX D = MX::sym("D", nd, N);
    MX P = MX::sym("P", np);

    // Initialize Q as a (n+m) x (n+m) matrix of zeros
    MX Q = MX::zeros(nx + nu, nx + nu);
    // Set the quadratic term for Xk(2)^2
    Q(2, 2) = 1.0;

    // Initialize cost and constraints
    MX J = 0;
    std::vector<MX> g;
    MX Uk;
    MX Dk;

    // Linear cost vector c
    MX c = MX::zeros(nx + nu);
    // Set the linear term for Xk(2)
    c(2) = -36.0;
    float hour_conversion = 3600 / h;
    // Set the linear terms for Uk(0) and Uk(4)
    c(nx + 0) = 0.09 * P(108) * 1e-3 / hour_conversion; // Uk(0)
    c(nx + 4) = 0.2 * P(172) * 1e-3 / hour_conversion;  // Uk(4)

    // Initial condition constraint
    g.push_back(X(Slice(), 0) - X0);

    // Loop over prediction horizon
    for (int k = 0; k < N; ++k) {
        Uk = U(Slice(), k);
        MX Xk = X(Slice(), k);
        MX Xk_next = X(Slice(), k + 1);
        Dk = D(Slice(), k);

        // Integrate to get the next state
        MXDict res = integrator_func(MXDict{
            {"x0", Xk},
            {"p", vertcat(Uk, Dk, P)}
        });
        MX Xk_next_predicted = res.at("xf");

        // Dynamics constraint: Xk+1 - F(Xk, Uk, Dk, P) = 0
        g.push_back(Xk_next - Xk_next_predicted);

        // Concatenate state and control vectors
        MX z = vertcat(Xk, Uk);

        // Accumulate cost function
        J += mtimes(z.T(), mtimes(Q, z)) + mtimes(c.T(), z);
    }

    // Decision variables vector
    MX w = MX::vertcat({vec(U), vec(X)});

    // Constraints
    MX g_all = MX::vertcat(g);

    // Parameters for NLP
    MX p_nlp = MX::vertcat({X0, vec(D), P});

    // Define the NLP problem
    MXDict nlp = {{"x", w}, {"f", J}, {"g", g_all}, {"p", p_nlp}};

    // Create solver options
    Dict nlp_opts;
    nlp_opts["ipopt.print_level"] = 1;
    nlp_opts["ipopt.max_iter"] = 1000;
    nlp_opts["ipopt.tol"] = 1e-4;
    nlp_opts["ipopt.acceptable_tol"] = 1e-4;
    nlp_opts["print_time"] = false;
    nlp_opts["ipopt.jacobian_approximation"] = "finite-difference-values";
    nlp_opts["ipopt.hessian_approximation"] = "limited-memory";
    nlp_opts["ipopt.linear_solver"] = "ma57";

    // Create solver
    Function solver = nlpsol("solver", "ipopt", nlp, nlp_opts);

    // Load or define your disturbance trajectory
    std::vector<std::vector<double>> d_values = load_weather_dummy(N);
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
    std::vector<double> w_init_U(nu * N, 0.5);  // Initial guess for controls

    // Set up initial guess for states
    std::vector<double> w_init_X;

    // Simulate forward to get initial guess for states
    DM Xk_num = x0_init;  // Start from initial state
    std::vector<double> Xk_num_vector = Xk_num.get_elements();
    w_init_X.insert(w_init_X.end(), Xk_num_vector.begin(), Xk_num_vector.end());

    for (int k = 0; k < N; ++k) {
        // Extract control input
        std::vector<double> u_k(w_init_U.begin() + k * nu, w_init_U.begin() + (k + 1) * nu);
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

        // Append state to initial guess
        Xk_num_vector = Xk_num.get_elements();
        w_init_X.insert(w_init_X.end(), Xk_num_vector.begin(), Xk_num_vector.end());
    }

    // Concatenate initial guesses for controls and states
    std::vector<double> w_init;
    w_init.insert(w_init.end(), w_init_U.begin(), w_init_U.end());
    w_init.insert(w_init.end(), w_init_X.begin(), w_init_X.end());

    // Set up bounds for controls
    std::vector<double> w_min_U(nu * N, 0.0);   // Lower bounds for controls
    std::vector<double> w_max_U(nu * N, 1.0);   // Upper bounds for controls

    // Set up bounds for states (assuming no bounds, use -inf and inf)
    std::vector<double> w_min_X((N + 1) * nx, -1e20);
    std::vector<double> w_max_X((N + 1) * nx, 1e20);

    // Concatenate bounds for decision variables
    std::vector<double> w_min;
    w_min.insert(w_min.end(), w_min_U.begin(), w_min_U.end());
    w_min.insert(w_min.end(), w_min_X.begin(), w_min_X.end());

    std::vector<double> w_max;
    w_max.insert(w_max.end(), w_max_U.begin(), w_max_U.end());
    w_max.insert(w_max.end(), w_max_X.begin(), w_max_X.end());

    // Set up constraints bounds (equality constraints g(x) = 0)
    int n_g = g.size() * nx;  // Total number of constraints
    std::vector<double> g_min(n_g, 0.0);
    std::vector<double> g_max(n_g, 0.0);

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

    try {
        solution = solver(solver_args);
        std::cout << "Optimization successful." << std::endl;

        // Extract the optimal solution
        std::vector<double> w_opt = solution.at("x").get_elements();

        // Extract optimal controls and states
        std::vector<double> U_opt(w_opt.begin(), w_opt.begin() + nu * N);
        std::vector<double> X_opt(w_opt.begin() + nu * N, w_opt.end());

        // Print optimal controls
        std::cout << "\nOptimal control inputs:" << std::endl;
        for (int k = 0; k < N; ++k) {
            std::cout << "Time " << k << ":" << std::endl;
            for (int j = 0; j < nu; ++j) {
                std::cout << "u[" << j << "] = " << U_opt[k * nu + j] << std::endl;
            }
            std::cout << "---" << std::endl;
        }

        // Print optimal states
        std::cout << "\nOptimal state trajectory:" << std::endl;
        for (int k = 0; k < N + 1; ++k) {
            std::cout << "Time " << k << ":" << std::endl;
            for (int j = 0; j < nx; ++j) {
                std::cout << "x[" << j << "] = " << X_opt[k * nx + j] << std::endl;
            }
            std::cout << "---" << std::endl;
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
