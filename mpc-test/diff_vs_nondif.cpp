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
    // Dict jit_options;

    int_opts["compiler"] = "shell";
    int_opts["abstol"] = 1e-6;
    int_opts["reltol"] = 1e-6;

    integrator_func = integrator(
        "integrator_func", "cvodes",
        {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
        0.0, h, int_opts
    );

    unsigned N = 96;

    // controllable inputs
    std::vector<std::vector<double>> d_values = load_dummy_weather(N);

    // non-controllable inputs (weather) 
    std::vector<std::vector<double>> controls = load_dummy_controls(N);

    // parameters
    std::vector<double> p_values = init_default_params(np);

    // Initial state
    DM x0_init = init_state(d_values[0], 90.0, 0.0);
    DM X = DM::zeros(nx, N+1);
    X(Slice(), 0) = x0_init;
    for (int k = 0; k < N; ++k) {
        // Prepare integrator inputs
        
        DM Xk_num = X(Slice(), k);
        DM input_args = DM::vertcat({controls[k], d_values[k], p_values});
        std::map<std::string, DM> integrator_in;
        integrator_in["x0"] = Xk_num;
        integrator_in["p"] = input_args;

        // Integrate to get next state
        DMDict res = integrator_func(integrator_in);
        X(Slice(), k+1) = res.at("xf");
    }

    std::ofstream file("states.csv");
    if (file.is_open()) {
        for (int i = 0; i < X.size1(); ++i) {
            for (int j = 0; j < X.size2(); ++j) {
                file << X(i, j);
                if (j < X.size2() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
    } else {
        std::cerr << "Unable to open file";
    }
}