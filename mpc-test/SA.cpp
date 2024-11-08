#include <iostream>
#include "ode.hpp"
#include "utils.hpp"
#include "params.hpp"
using namespace casadi;

int main() {
    int np = 208;
    int nx = 28;
    int nu = 8;
    int nd = 10;
    int na = 240;
    float h = 900;
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
    opts["jit"] = true;
    opts["compiler"] = "shell";
    opts["abstol"] = 1e-6;
    opts["reltol"] = 1e-6;
    Dict jit_options;
    jit_options["flags"] = "-Ofast";
    jit_options["compiler"] = "gcc";
    opts["jit_options"] = jit_options;

    integrator_func = integrator(
        "integrator_func", "cvodes",
        {{"x", x}, {"p", input_args_sym}, {"ode", dxdt}},
        0., h, opts
    );

    // For sensitivity analysis
    // Note: We should use the original variables x, p instead of creating new ones
    SX s = SX::sym("s", nx, np);  // sensitivity matrix

    // Calculate Jacobians
    SX jac_x = jacobian(dxdt, x);   // This will be nx x nx
    SX jac_p = jacobian(dxdt, p);   // This will be nx x np

    // The sensitivity equation should be:
    // ds/dt = (∂f/∂x)s + ∂f/∂p
    // where f is your ODE function (dxdt)
    SX sdot_jac = mtimes(jac_x, s) + jac_p;

    // Create the sensitivity function
    Function sdot = Function("sdot",
                           {s, x, u, d, p},    // inputs
                           {sdot_jac},         // outputs
                           {"s", "x", "u", "d", "p"},  // input names
                           {"sdot"});          // output names
    
    int N = 2;
    std::vector<double> x_dummy(nx, 0.1);  // Initialize state with 0.1
    std::vector<double> u_dummy(nu, 0.0);  // Initialize inputs with 0
    std::vector<double> d_dummy(nd, 0.0);  // Initialize disturbances with 0
    std::vector<double> p_dummy(np, 1.0);  // Initialize parameters with 1
    std::vector<double> s_dummy(nx * np, 0.0);
    for(int i = 0; i < nx; i++) {
        s_dummy[i * np + i] = 1.0;  // Optional: set diagonal elements to 1
    }
    sdot(
        {DM(nx, np, s_dummy),  // Reshape s_dummy to nx x np matrix
         DM(x_dummy),
         DM(u_dummy),
         DM(d_dummy),
         DM(p_dummy)}
    );

    // s_num(:,0,:) = full(q(squeeze(s(:,kk,:)), x(:,kk,ll), u(:,kk), d(:,kk), p, ops.h, sdot));

    return 0;
}
