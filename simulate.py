import numpy as np
import casadi as ca
import pandas as pd
from environments.utils import define_model, init_state
from model.parameters import init_default_params


controls = np.loadtxt("results/exact/june/control-inputs-300dt.csv", delimiter=",")
weather = np.loadtxt("weather/june/weather-300dt.csv", delimiter=",")
print(controls.shape)
N = 288
nx = 28
nu = 6
nd = 7
n_params = 208
dt = 300

x0 = init_state(weather[0], 85.0)
X = np.zeros((nx, N+1))
p = init_default_params(n_params)
F = define_model(nx, nu, nd, n_params, dt)

X[:, 0] = x0
for i, u in enumerate(controls):
    res= F(x0=X[:, i], u=u, p=ca.vertcat(weather[i], p))
    X[:, i+1] = res["xf"].full().flatten()

columns = ["co2Air", "co2Top", "tAir", "tTop", "tCan", "tCovIn", "tCovE", "tThScr", "tFlr", "tPipe", "tSo1", "tSo2", "tSo3", "tSo4", "tSo5", "vpAir", "vpTop", "tLamp", "tIntLamp", "tGroPipe", "tBlScr", "tCan24", "cBuf", "cLeaf", "cStem", "cFruit", "tCanSum", "time"]
X = pd.DataFrame(X.T, columns=columns)
print(X.head())
# X.to_csv("results/comparison/june/states-300dt.csv", index=False)