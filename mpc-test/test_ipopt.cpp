#include <casadi/casadi.hpp>
#include <iostream>

int main() {
    using namespace casadi;

    // Define optimization variables
    MX x = MX::sym("x", 2);

    // Objective function: minimize x0^2 + x1^2
    MX obj = pow(x(0), 2) + pow(x(1), 2);

    // Constraints: x0 + x1 = 1
    MX g = x(0) + x(1) - 1;

    // Create NLP (Nonlinear Programming) problem structure
    MXDict nlp = {{"x", x}, {"f", obj}, {"g", g}};

    // Define options for the solver
    Dict opts;
    opts["ipopt.linear_solver"] = "ma57";  // Use ma57 linear solver from HSL
    opts["ipopt.print_level"] = 5;         // Optional: Increase verbosity for debugging

    // Create IPOPT solver instance with the specified options
    Function solver = nlpsol("solver", "ipopt", nlp, opts);

    // Initial guess for the optimization variables
    DMDict arg = {{"x0", DM::zeros(2)},   // Starting point (x0 = [0, 0])
                  {"lbg", 0},              // Lower bound for constraints
                  {"ubg", 0}};             // Upper bound for constraints

    // Solve the optimization problem
    DMDict res = solver(arg);

    // Extract and print the optimal solution
    std::cout << "Optimal solution: " << res.at("x") << std::endl;

    return 0;
}
