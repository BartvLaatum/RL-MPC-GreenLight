import os 
from time import time

import numpy as np
import casadi as ca

from agents.baseline import RuleBasedController
from model.utils import define_model, init_state, load_dummy_weather, co2dens2ppm, vaporPres2rh, convert_rh_ppm
from model.parameters import init_default_params
from utils import load_model_hyperparams


def constraint_violation(y: np.ndarray):
    """
    Function that computes the absolute penalty for violating state constraints.
    State constraints are currently non-dynamical, and based on the boundaries of the system.
    We do not look at dry mass bounds, since those are non-existent in real greenhouse.
    """        
    y_min = np.array([400, 15, 0])
    y_max = np.array([1600, 25, 90])
    lowerbound = y_min[:] - y[:]
    lowerbound[lowerbound < 0] = 0
    upperbound = y[:] - y_max[:]
    upperbound[upperbound < 0] = 0
    return lowerbound, upperbound

def compute_penalties(X: np.ndarray):
    pen_w = np.array([5e-5, 5e-3, 7e-4])
    # Transform state variable to ppm and relative humidity
    y = X[[0, 2, 15]].copy()
    y[0] = co2dens2ppm(y[1], y[0]*1e-6)
    y[2] = vaporPres2rh(y[1], y[2])

    lowerbound, upperbound = constraint_violation(y)
    penalties = np.dot(pen_w, lowerbound) + np.dot(pen_w, upperbound)
    return np.sum(penalties)


env_id = "TomatoEnv"
nx, nu, nd, n_params, dt = 28, 6, 10, 208, 300
dt = 300.
n_days = 1
month = "june"
L = n_days*86400
t = np.arange(0, L, dt)
N = len(t)
d = load_dummy_weather(N, dt, month=month)
F = define_model(nx, nu, nd, n_params, dt)

# initiate Rule-Based Controller
rb_params = load_model_hyperparams('rule_based', env_id)
rb_controller = RuleBasedController(nu=nu, **rb_params)

x0 = init_state(d[0], 85.0, 0.0)

X = np.zeros((nx, N+1))
U = np.zeros((nu, N))
EPI = np.zeros((N, 1))
penalties = np.zeros((N, 1))
rewards = np.zeros((N, 1))
exec_time = np.zeros((N, 1))

X[:, 0] = x0


hour_of_day = 0
day_of_year = 151
p = init_default_params(n_params)
hour_conversion = 3600/dt


for k in range(N):
    u = rb_controller.predict(X[:, k], d[k], hour_of_day, day_of_year)
    U[:, k] = u
    t = time()
    res = F(x0=X[:, k], u=U[:, k], p=ca.vertcat(*[d[k], p]))
    X[:, k+1] = res["xf"].toarray().ravel()

    hour_of_day = (hour_of_day + 1) % 24
    if hour_of_day == 0:    
        day_of_year = (day_of_year + 1) % 365
    EPI[k, :] = \
        (X[25, k+1]-X[25, k])* 1e-6 / 0.06 * 1.2 - \
        (0.09 * p[108]/p[46] * 1e-3 * U[0, k]/hour_conversion + \
        0.2 * p[172] * 1e-3 * U[4, k]/hour_conversion + \
        0.3 * U[1, k]* p[109]/p[46] * 1e-6 * dt)
    penalties[k, :] = compute_penalties(X[:,k+1])
    exec_time[k, :] = time() - t

    rewards[k, :] = EPI[k, :] - penalties[k, :]

convert_rh_ppm(X)

method = "rbc" 
approach = "single"
dir = f"results/test/{method}/{approach}/{month}"
os.makedirs(dir, exist_ok=True)
np.savetxt(f"{dir}/control-inputs-cs-{int(dt)}dt.csv", U.T, delimiter=",")
np.savetxt(f"{dir}/states-cs-{int(dt)}dt.csv", X.T, delimiter=",")
np.savetxt(f"{dir}/times-cs-{int(dt)}dt.csv", exec_time, delimiter=",")
np.savetxt(f"{dir}/EPI-cs-{int(dt)}dt.csv", EPI, delimiter=",")
np.savetxt(f"{dir}/penalties-cs-{int(dt)}dt.csv", penalties, delimiter=",")
np.savetxt(f"{dir}/rewards-cs-{int(dt)}dt.csv", rewards, delimiter=",")
