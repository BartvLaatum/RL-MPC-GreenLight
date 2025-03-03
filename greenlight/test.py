import casadi as ca
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# Problem Setup: Stiff van der Pol ODE
# -------------------------------

# Time horizon and discretization
T = 2.0        # MPC prediction horizon [seconds]
N = 20         # number of shooting intervals
dt = T / N     # time step

# Stiffness parameter (large mu makes the system stiff)
mu = 10.0

# Define state (x1, x2) and control (u) symbols
x = ca.SX.sym('x', 2)
u = ca.SX.sym('u', 1)

# Define the controlled van der Pol dynamics
#   x1_dot = x2
#   x2_dot = mu*(1 - x1^2)*x2 - x1 + u
xdot = ca.vertcat(x[1], mu*(1 - x[0]**2)*x[1] - x[0] + u)

# Create a dictionary for the DAE (ordinary differential equation)
dae = {'x': x, 'p': u, 'ode': xdot}

# Options for the CVODES integrator (from SUNDIALS)
# Options for the CVODES integrator (from SUNDIALS)
int_opts = {
    'abstol': 1e-8,
    'reltol': 1e-6,
    # Change the linear solver from 'dense' to 'csparse' which is usually available by default
    'linear_solver': 'csparse'
}

# Create the CVODES-based integrator (implicit solver for stiff systems)
F = ca.integrator('F', 'cvodes', dae, 0, dt, int_opts)


# -------------------------------
# Setup the Multi-Shooting NLP for MPC
# -------------------------------

# Decision variables: states over the horizon and controls
X = ca.MX.sym('X', 2, N+1)  # states at each node: [x0, x1, ..., xN]
U = ca.MX.sym('U', 1, N)    # control inputs for each interval

# Parameter: the initial state (could be extended to include reference signals)
P = ca.MX.sym('P', 2)

# Define the cost function weights
Q = ca.diag([1, 1])   # state tracking weight
R = 0.1               # control effort weight

# Initialize objective and constraint vector
obj = 0
g = []  # constraints to enforce dynamic continuity

# Loop over each shooting interval
for k in range(N):
    # Current state and control
    x_current = X[:, k]
    u_current = U[:, k]
    
    # Integrate the dynamics over dt using CVODES
    res = F(x0=x_current, p=u_current)
    x_next = res['xf']
    
    # Dynamic continuity constraint: next state in the trajectory equals the integrated state
    g.append(x_next - X[:, k+1])
    
    # Stage cost: quadratic cost on state deviation and control effort.
    obj += ca.mtimes([x_current.T, Q, x_current]) + R * (u_current**2)

# Add a terminal cost
obj += ca.mtimes([X[:, N].T, Q, X[:, N]])

# Concatenate the constraints
g = ca.vertcat(*g)

# Reshape decision variables into a single column vector
opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

# Formulate the NLP
nlp = {'f': obj, 'x': opt_vars, 'g': g, 'p': P}
opts = {'ipopt.print_level': 0, 'print_time': 0}
solver = ca.nlpsol('solver', 'ipopt', nlp, opts)

# -------------------------------
# MPC Closed-loop Simulation Setup
# -------------------------------

# Dimensions of decision variables
nX = 2 * (N + 1)   # total state variables over the horizon
nU = N             # total control inputs

# Bounds for decision variables: no bounds here, but these can be adjusted as needed.
lbx = -ca.inf * np.ones(nX + nU)
ubx = ca.inf * np.ones(nX + nU)

# Equality constraints (dynamics matching) must equal zero.
lbg = np.zeros(2 * N)
ubg = np.zeros(2 * N)

# Initial condition for the simulation
x0_sim = np.array([2.0, 0.0])
sim_time = 4.0   # total simulation time
n_sim_steps = int(sim_time / dt)

# Lists to store simulation data
simX = [x0_sim.copy()]
simU = []
import time
# -------------------------------
# MPC Simulation Loop
# -------------------------------
for i in range(n_sim_steps):
    t = time.time()
    # Set parameter to current state for the NLP
    p_val = x0_sim
    
    # Provide an initial guess: states are constant at the current value, controls are zero.
    x_init = np.tile(x0_sim.reshape(2,1), (1, N+1))
    u_init = np.zeros((1, N))
    init_guess = np.concatenate((x_init.reshape(-1, 1), u_init.reshape(-1, 1)), axis=0)
    
    # Solve the NLP
    sol = solver(x0=init_guess, p=p_val, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    sol_vars = sol['x'].full().flatten()
    
    # Extract the control sequence and apply the first control input
    u_opt = sol_vars[nX:]
    u_apply = u_opt[0]
    simU.append(u_apply)
    
    # Simulate one step of the real system using the same integrator F.
    res = F(x0=x0_sim, p=u_apply)
    x0_sim = res['xf'].full().flatten()
    simX.append(x0_sim.copy())


    print(f"time step took: {time.time()-t}")
    print(f"Step {i+1:02d}: State = {x0_sim}, Control = {u_apply}")

# -------------------------------
# Plot the results
# -------------------------------
simX = np.array(simX)
time_grid = np.linspace(0, sim_time, simX.shape[0])
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(time_grid, simX[:, 0], label='x1')
plt.plot(time_grid, simX[:, 1], label='x2')
plt.xlabel('Time [s]')
plt.ylabel('States')
plt.legend()
plt.title('State Trajectories')

plt.subplot(1, 2, 2)
plt.step(time_grid[:-1], simU, where='post')
plt.xlabel('Time [s]')
plt.ylabel('Control Input')
plt.title('Control Profile')

plt.tight_layout()
plt.savefig("test.png")
plt.show()
