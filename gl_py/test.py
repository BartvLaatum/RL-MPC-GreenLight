import casadi as ca
import numpy as np

x = ca.MX.sym('x')
y = ca.MX.sym('y')

# Equivalent of np.pow(x, y)
result =  ca.constpow(x, y)

print(result)  # Outputs: pow(x, y)