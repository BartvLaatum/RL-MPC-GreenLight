# Model Predictive Control (MPC) for GreenLight Greenhouse Model

## Project Description
This C++ codebase implements an economic Model Predictive Control (MPC) strategy for optimizing greenhouse climate control using the GreenLight model. The primary goal is to maximize fruit growth while minimizing the costs associated with lighting, heating, and CO2 enrichment. This implementation currently does not enforce system constraints but focuses on the economic optimization of inputs.

**Key Features:**
- The optimization problem solves for 96 timesteps into the future, where each timestep is 15 minutes.
- Outputs include open-loop control and state trajectories.
- Results are saved as CSV files in the `data/month/` folder, categorized by the loaded weather data.


---

## Features
### Weather Files
- **Location**: Weather data for January and June are stored in `data/month/weather.csv`.
- **Details**: The weather files contain data for the first day of each respective month.

### Controllable Inputs
The MPC model includes six controllable inputs:
1. `uBoil` (boiler output)
2. `uCo2` (CO2 injection rate)
3. `uThScr` (thermal screen position)
4. `uVent` (ventilation position)
5. `uLamp` (lamp output)
6. `uBlScr` (blackout screen position)

**Note**: Intermediate lamps and grow pipes are excluded from this implementation.

### Model Versions
The code supports two versions of the model for state computation:
1. **Original (Discontinuous)**: Implemented in `aux_states.hpp`.
2. **Smooth (Differentiable)**: Implemented in `aux_states-diff.hpp`, using smooth approximations for functions like `min` and `max`.

**Switching Between Versions:**
- Modify `ode.hpp` by de-commenting the desired `.hpp` file.
- Recompile the code to apply changes.

**Model parameters**
- Model parameters stem from the Bleijswijk 2010 settings, calibrated by [David Katzin](https://github.com/davkat1/GreenLight).

### Visualization
- **Script**: `visualise_trajectories.py`
- **Functionality**: Plots the resulting open-loop state and control trajectories.
- **Output**: Saved as image files in `figures/month/`.

---

## Dependencies
### CasADi C++
- Used for symbolic computation.
- Official documentation: [CasADi](https://web.casadi.org/)

### IPOPT
- Optimization solver.
- **Optional Solver Support:**
  - HSL library (e.g., `ma57`): [HSL Licensing](https://licences.stfc.ac.uk/product/coin-hsl)
  - MUMPS: Suitable for smaller systems but may increase computational time.
- Documentation: [IPOPT](https://coin-or.github.io/Ipopt/)

---

## Instructions
### Compilation
Use the following command to compile the program:
```bash
g++ -o mpc mpc.cpp \
    -I/usr/local/include \
    -L/usr/local/lib \
    -lcasadi -lipopt -lpthread -ldl -lm
```

### Running the Program
Run the compiled executable:
```bash
./mpc
```

### Manual Adjustments
1. **Changing the Weather Month**:
   - Edit the `load_dummy_weather` function in `utils.cpp` to select the desired weather file.
   - Update save directories in `mpc.cpp` accordingly.

2. **Model Version Indication**:
   - The resulting CSV files include a suffix (`_diff` or ``) to specify whether the smooth (differentiable) or original (discontinuous) model version was used.

---

## Expected Outputs
### Data Files
- Saved in `data/month/`.
- Include open-loop control and state trajectories.

### Visualizations
- Generated plots saved in `figures/month/`.
---

## Additional Notes
### Replacing the Solver
- MUMPS could be considered.
- Trade-off: MUMPS may increase computation time compared to HSL solvers like `ma57`.
- Make sure all libraries are linked correctly.

---

## Contact
For questions or further assistance, please contact the me at [GitHub Issues](https://github.com/BartvLaatum).

