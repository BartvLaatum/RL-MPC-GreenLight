#include <fstream>
#include <string>
#include <sstream>
#include <iostream>

#include "ode.hpp"
#include "utils.hpp"
#include "params.hpp"

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
    int nu = 8;
    int nd = 10;
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

    Dict opts;
    // opts["jit"] = true;
    // opts["compiler"] = "shell";
    opts["abstol"] = 1e-6;
    opts["reltol"] = 1e-6;
    // opts["linear_multistep_method"] = "bdf"; // adams or bdf 
    // opts["max_num_steps"] = 1000; // Increase as needed

    // Dict jit_options;
    // jit_options["flags"] = "-Ofast";
    // jit_options["compiler"] = "gcc";
    // opts["jit_options"] = jit_options;

    integrator_func = integrator(
        "integrator_func", "cvodes",
        {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
       0.0, h, opts
    );

    // Prediction horizon
    const int N = 2;

    // Control variables (decision variables)
    MX U = MX::sym("U", nu, N);

    // Parameters (initial state, disturbances, and parameters)
    MX X0 = MX::sym("X0", nx);
    MX D = MX::sym("D", nd, N);
    MX P = MX::sym("P", np);

    // Initialize cost and constraints
    MX J = 0;
    std::vector<MX> g;

    // Initialize state trajectory
    MX Xk = X0;

    // Loop over prediction horizon
    for (int k = 0; k < N; ++k) {
        MX Uk = U(Slice(), k);
        MX Dk = D(Slice(), k);

        // Integrate to get the next state
        MXDict res = integrator_func(MXDict{
            {"x0", Xk},
            {"p", vertcat(Uk, Dk, P)}
        });
        Xk = res.at("xf");

        // Accumulate cost function (e.g., tracking error)
        J += (Xk(2) - 18)*(Xk(2) - 18) + 0.09 * 80. * Uk(0);

        // Add any path constraints here (if needed)
    }
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
    nlp_opts["ipopt.print_level"] = 1;
    nlp_opts["ipopt.max_iter"] = 1000;
    nlp_opts["ipopt.tol"] = 1e-4;
    nlp_opts["ipopt.acceptable_tol"] = 1e-4;
    nlp_opts["print_time"] = false;
    nlp_opts["ipopt.hessian_approximation"] = "limited-memory";
    nlp_opts["print_time"] = true;
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

            // Print state
            std::cout << "Time " << k << ":" << std::endl;
            for (int j = 0; j < nx; ++j) {
                std::cout << "x[" << j << "] = " << Xk_num(j).scalar() << std::endl;
            }
            std::cout << "---" << std::endl;
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

        // Print solution statistics
        std::cout << "\nSolver statistics:" << std::endl;
        std::cout << "Objective value: " << double(solution.at("f")) << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Error while solving the optimization problem: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
