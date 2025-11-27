import numpy as np

controls = np.loadtxt("results/exact/june/control-inputs-300dt.csv", delimiter=",")
weather = np.loadtxt("weather/june/weather-300dt.csv", delimiter=",")
time_steps = np.arange(0, len(controls) * 300, 300).reshape(-1, 1)
controls_with_time = np.hstack((time_steps, controls))
time_steps = np.arange(0, len(weather) * 300, 300).reshape(-1, 1)
weather_with_time = np.hstack((time_steps, weather))
np.savetxt("results/exact/june/control-inputs-with-time.csv", controls_with_time, delimiter=",")
np.savetxt("results/exact/june/weather-with-time.csv", weather_with_time, delimiter=",")
