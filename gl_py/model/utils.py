import casadi as ca
import numpy as np
import pandas as pd
# from casadi import SX, MX, vertcat, Function, integrator  # Import specific classes/functions

from model.ode import ODE

def define_model(nx: int, nu: int, nd: int, n_params: int, dt: float):
    # Define the symbolic variables for CasADi
    x = ca.SX.sym("x", nx)
    u = ca.SX.sym("u", nu)
    d = ca.SX.sym("d", nd)
    p = ca.SX.sym("p", n_params)

    dxdt = ODE(x, u, d, p)
    input_args_sym = ca.vertcat(d, p)

    int_opts = {"abstol": 1e-6, "reltol": 1e-6}
    # int_opts = {}
    F = ca.integrator(
        "F", "cvodes",
        {"x": x, "u": u, "p": input_args_sym, "ode": dxdt},
        0.0, dt, int_opts
    )

    return F

def satVp_cpp(temp):
    """Calculate saturation vapor pressure"""
    a = 610.78
    b = 17.2694
    c = 238.3
    return a * np.exp(b * temp / (temp + c))

def cond(hec, vp1, vp2):
    """Condensation function"""
    a = 6.4e-9
    return 1.0 / (1.0 + np.exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2)

def co2dens2ppm_cpp(temp, dens):
    """Convert CO2 density to CO2 concentration [ppm]"""
    R = 8.3144598        # Molar gas constant [J mol^{-1} K^{-1}]
    C2K = 273.15         # Conversion from Celsius to Kelvin [K]
    M_CO2 = 44.01e-3     # Molar mass of CO2 [kg mol^{-1}]
    P = 101325           # Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2)

def proportional_control(processVar, setPt, pBand, minVal, maxVal):
    """Proportional control function"""
    return minVal + (maxVal - minVal) * (1. / (1. + np.exp(-2. / pBand * np.log(100.) * (processVar - setPt - pBand / 2.))))

def tau12(tau1, tau2, rho1Dn, rho2Up):
    """Transmission coefficient of a double layer [-]
    Equation 14 [1], Equation A4 [5]
    """
    return tau1 * tau2 / (1. - rho1Dn * rho2Up)

def rhoDn(tau2, rho1Dn, rho2Up, rho2Dn):
    """Reflection coefficient of the lower layer [-]
    Equation 15 [1], Equation A5 [5]
    """
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up)

def dli_check(lamp_input, dli):
    """Check daily light integral"""
    if dli > 15.:
        return 0.
    return lamp_input

def init_state(d0, rhMax, time_in_days):
    """Initialize greenhouse state vector"""
    state = np.zeros(28)
    state[0] = d0[3]        # co2Air
    state[1] = state[0]     # co2Top
    state[2] = 18.5         # tAir
    state[3] = state[2]     # tTop
    state[4] = state[2] + 4 # tCan
    state[5] = state[2]     # tCovIn
    state[6] = state[2]     # tCovE
    state[7] = state[2]     # tThScr
    state[8] = state[2]     # tFlr
    state[9] = state[2]     # tPipe
    state[10] = state[2]    # tSoil1
    state[11] = .25*(3.*state[2] + d0[6])   # tSoil2
    state[12] = .25*(2.*state[2] + 2*d0[6]) # tSoil3
    state[13] = .25*(state[2] + 3*d0[6])    # tSoil4
    state[14] = d0[6]       # tSoil5
    state[15] = rhMax / 100. * satVp_cpp(state[2])  # vpAir
    state[16] = state[15]   # vpTop
    state[17] = state[2]    # tLamp
    state[18] = state[2]    # tIntLamp
    state[19] = state[2]    # tGroPipe
    state[20] = state[2]    # tBlScr
    state[21] = state[4]    # tCan24
    state[22] = 1000.       # cBuf
    state[23] = 9.5283e4    # cLeaf
    state[24] = 2.5107e5    # cStem
    state[25] = 5.5338e4    # cFruit
    state[26] = 3.0978e3    # tCanSum
    state[27] = time_in_days # time
    return state

def load_dummy_weather(N, dt, month='june'):
    """Load weather data from CSV file
    Args:
        N: Number of timesteps to load
        month: Month to load data from (default: 'june')
    Returns:
        weather_data: List of weather data rows
    """
    try:
        df = pd.read_csv(f"weather/{month}/weather-{int(dt)}dt.csv")
        weather_data = df.head(N).values

    except FileNotFoundError:
        print(f"Error: Could not open weather data file for {month}")
        weather_data = []
    return weather_data

