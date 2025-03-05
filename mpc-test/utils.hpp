// utils_casadi.hpp
#ifndef UTILS_HPP
#define UTILS_HPP

#include <casadi/casadi.hpp>
#include <cmath>
#include <vector>

inline double satVp_cpp(double temp) {
    const float a = 610.78;
    const float b = 17.2694;
    const float c = 238.3;
    return a * std::exp(b * temp / (temp + c));
}

inline double cond(double hec, double vp1, double vp2) {
    const double a = 6.4e-9; 
    return 1.0 / (1.0 + std::exp(-0.1 * (vp1 - vp2))) * a * hec * (vp1 - vp2);
}

inline double co2dens2ppm_cpp(double temp, double dens) {
    const double R = 8.3144598;        // Molar gas constant [J mol^{-1} K^{-1}]
    const double C2K = 273.15;         // Conversion from Celsius to Kelvin [K]
    const double M_CO2 = 44.01e-3;     // Molar mass of CO2 [kg mol^{-1}]
    const double P = 101325;           // Pressure (assumed to be 1 atm) [Pa]
    
    return 1e6 * R * (temp + C2K) * dens / (P * M_CO2);
}

inline double proportional_control(double processVar, double setPt, double pBand, double minVal, double maxVal) {
    return minVal + (maxVal - minVal) * (1. / (1. + std::exp(-2. / pBand * std::log(100.) * (processVar - setPt - pBand / 2.))));
}

inline double tau12(double tau1, double tau2, double rho1Dn, double rho2Up) {
    // Transmission coefficient of a double layer [-]
    // Equation 14 [1], Equation A4 [5]
    return tau1 * tau2 / (1. - rho1Dn * rho2Up);
}

inline double rhoDn(double tau2, double rho1Dn, double rho2Up, double rho2Dn) {
    // Reflection coefficient of the lower layer [-]
    // Equation 15 [1], Equation A5 [5]
    return rho2Dn + (tau2 * tau2 * rho1Dn) / (1. - rho1Dn * rho2Up);
}


double dli_check(double lamp_input, float dli)
{
    if (dli > 15.) {
        return 0.;
    } 
    return lamp_input;
}

DM init_state(const std::vector<double>& d0, float rhMax, double time_in_days) 
{
    DM state = DM::zeros(28);
    state(0) = d0[3];       // co2Air
    state(1) = state(0);    // co2Top
    state(2) = 18.5;        // tAir
    state(3) = state(2);    // tTop
    state(4) = state(2) + 2; // tCan
    state(5) = state(2);    // tCovIn
    state(6) = state(2);    // tCovE
    state(7) = state(2);    // tThScr
    state(8) = state(2);    // tFlr
    state(9) = state(2);    // tPipe
    state(10) = state(2);   // tSoil1
    state(11) = .25*(3.*state(2) + d0[6]);  // tSoil2
    state(12) = .25*(2.*state(2) + 2*d0[6]);// tSoil3
    state(13) = .25*(state(2) + 3*d0[6]);   // tSoil4
    state(14) = d0[6];      // tSoil5
    state(15) = rhMax / 100. * satVP(state(2)); // vpAir
    state(16) = state(15);  // vpTop
    state(17) = state(2);   // tLamp
    state(18) = state(2);   // tIntLamp
    state(19) = state(2);   // tGroPipe
    state(20) = state(2);   // tBlScr
    state(21) = state(4);   // tCan24
    state(22) = 1000.;      // cBuf
    state(23) = 9.5283e4;     // cLeaf
    state(24) = 2.5107e5;     // cStem
    state(25) = 5.5338e4;      // cFruit
    state(26) = 3.0978e3;         // tCanSum
    state(27) = time_in_days; // time
    return state;
}

// Function to load dummy weather data from a text file
std::vector<std::vector<double>> load_weather_dummy(int N) {
    std::vector<std::vector<double>> weather_data;
    std::ifstream file("weather.csv");

    if (!file.is_open()) {
        std::cerr << "Error opening weather data file" << std::endl;
        return weather_data;
    }

    std::string line;
    while (std::getline(file, line) && weather_data.size() < N) {
        std::vector<double> data_row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            data_row.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        weather_data.push_back(data_row);
    }

    file.close();
    return weather_data;
}

// Function to load dummy weather data from a text file
std::vector<std::vector<double>> load_dummy_controls(int N) {
    std::vector<std::vector<double>> controls;
    std::ifstream file("data/januari/optimal_controls-non-diff.csv");

    if (!file.is_open()) {
        std::cerr << "Error opening weather data file" << std::endl;
        return controls;
    }

    std::string line;
    while (std::getline(file, line) && controls.size() < N) {
        std::vector<double> data_row;
        std::stringstream ss(line);
        double value;

        while (ss >> value) {
            data_row.push_back(value);
            if (ss.peek() == ',') {
                ss.ignore();
            }
        }

        controls.push_back(data_row);
    }

    file.close();
    return controls;
}


#endif  // UTILS_HPP
