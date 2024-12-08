// Markus Buchholz, 2024
// g++ mpc_blueboat_controls.cpp -o mpc_controls -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#include <iostream>
#include <vector>
#include <tuple>
#include <eigen3/Eigen/Dense>
#include <cmath>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

// MPC parameters
int horizon = 20;  // Prediction horizon
float Q = 0.0001f; // State cost
float R = 0.01f;   // Control cost
float S = 0.001f;  // Derivative control cost

// Environmental disturbances
const float A = 3.0;                 // Wave amplitude
const float Vw = 5.0;                // Wind speed (m/s)
const float Vc = 2.5;                // Speed of the ocean current (m/s)
const float rho_water = 1000.0;      // Density of water (kg/m^3)
const float g = 9.81;                // Gravity (m/s^2)
const float L = 2.0;                 // Length of the vehicle
const float B = 2.0;                 // Breadth of the vehicle
const float draft = 0.5;             // Draft of the vehicle
const float Lambda = 25000.0;        // Wavelength
const float omega_e = 0.5;           // Wave frequency
const float phi = 0;                 // Wave phase
const float current_angle = M_PI / 6; // Angle of the ocean current (radians)
const float rho_air = 1.225;         // Density of air (kg/m^3)
const float Cx = 0.001;              // Coefficient for wind drag in x
const float Cy = 0.001;              // Coefficient for wind drag in y
const float Ck = 0.001;              // Coefficient for yaw moment due to wind
const float Aw = 5.0;                // Area for wind
const float Alw = 5.0;               // Area for yaw wind
const float Hlw = 2.0;               // Height for yaw wind
const float beta_w = M_PI / 4;       // Wind direction relative to vehicle
const float wave_beta = M_PI / 4;    // Wave direction

// Dynamic model class for ASV
class DynamicModel {
public:
    float Tstbd = 0;
    float Tport = 0;

    float delta_x = 0;
    float delta_y = 0;

    // Hydrodynamic parameters
    float Xu;
    float Yv;
    float Yr;
    float Nv;
    float Nr;
    float X_u_dot = -2.25;
    float Y_v_dot = -23.13;
    float Y_r_dot = -1.31;
    float N_v_dot = -16.41;
    float N_r_dot = -2.79;
    float Xuu;
    float Yvv = -99.99;
    float Yvr = -5.49;
    float Yrv = -5.49;
    float Yrr = -8.8;
    float Nvv = -5.49;
    float Nvr = -8.8;
    float Nrv = -8.8;
    float Nrr = -3.49;

    float m = 16.0; // mass = (14.5) boat + 2 * (0.75) batteries
    float Iz = 4.1; // moment of inertia
    float B = 0.41; // centerline-to-centerline separation
    float c = 1.0;  // thruster correction factor

    Vector3f upsilon;
    Vector3f upsilon_dot_last;
    Vector3f upsilon_dot;
    Vector3f eta;
    Vector3f eta_dot_last;
    Vector3f eta_dot;
    Matrix3f M;
    Vector3f T; // Torque and force vector
    Matrix3f CRB;
    Matrix3f CA;
    Matrix3f C;
    Matrix3f Dl;
    Matrix3f Dn;
    Matrix3f D;
    Matrix3f J;
    Vector3f Delta;

    float x;
    float y;
    float etheta;
    float u;
    float v;
    float r;

    DynamicModel() {
        upsilon << 0, 0, 0;
        upsilon_dot_last << 0, 0, 0;
        eta << 0, 0, 0;
        eta_dot_last << 0, 0, 0;

        // constant matrix M
        M << m - X_u_dot, 0, 0,
             0, m - Y_v_dot, 0 - Y_r_dot,
             0, 0 - N_v_dot, Iz - N_r_dot;

        J << cos(eta(2)), -sin(eta(2)), 0,
             sin(eta(2)), cos(eta(2)), 0,
             0, 0, 1;
    }

    Vector3f function_1(Vector3f upsilon, float time) {
        // Hydrodynamic equations and parameter conditions
        Xu = -25;
        Xuu = 0;
        if (abs(upsilon(0)) > 1.2) {
            Xu = 64.55;
            Xuu = -70.92;
        }

        // Ensure upsilon(0) and upsilon(1) are not NaN
        if (isnan(upsilon(0)) || isnan(upsilon(1))) {
            std::cerr << "Invalid upsilon values: " << upsilon.transpose()
                      << std::endl;
            exit(1);
        }

        Yv = 0.5 * (-40 * 1000 * abs(upsilon(1))) *
             (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) +
              0.016 * (pow((0.27 / 0.09), 2)));
        Yr = 6 * (-3.141592 * 1000) *
             sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01;
        Nv = 0.06 * (-3.141592 * 1000) *
             sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01;
        Nr = 0.02 * (-3.141592 * 1000) *
             sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01 *
             1.01;

        // Clamp values to avoid extreme values
        Yv = std::max(std::min(Yv, 1e8f), -1e8f);
        Yr = std::max(std::min(Yr, 1e8f), -1e8f);
        Nv = std::max(std::min(Nv, 1e8f), -1e8f);
        Nr = std::max(std::min(Nr, 1e8f), -1e8f);

        // Vector of NED disturbances
        Delta << delta_x, delta_y, 0;

        // Vector of body disturbances
        Delta = J.inverse() * Delta;

        // Vector tau of torques
        T << Tport + c * Tstbd, 0, 0.5 * B * (Tport - c * Tstbd);

        // Coriolis matrix - rigid body
        CRB << 0, 0, 0 - m * upsilon(1),
               0, 0, m * upsilon(0),
               m * upsilon(1), 0 - m * upsilon(0), 0;

        // Coriolis matrix - added mass
        CA << 0, 0,
            2 * ((Y_v_dot * upsilon(1)) + ((Y_r_dot + N_v_dot) / 2) * upsilon(2)),
            0, 0, 0 - X_u_dot * m * upsilon(0),
            2 * (((0 - Y_v_dot) * upsilon(1)) -
                 ((Y_r_dot + N_v_dot) / 2) * upsilon(2)),
            X_u_dot * m * upsilon(0), 0;

        // Coriolis matrix
        C = CRB + CA;

        // Drag matrix - linear
        Dl << 0 - Xu, 0, 0,
              0, 0 - Yv, 0 - Yr,
              0, 0 - Nv, 0 - Nr;

        // Drag matrix - nonlinear
        Dn << Xuu * abs(upsilon(0)), 0, 0,
            0, Yvv * abs(upsilon(1)) + Yvr * abs(upsilon(2)),
            Yrv * abs(upsilon(1)) + Yrr * abs(upsilon(2)), 0,
            Nvv * abs(upsilon(1)) + Nvr * abs(upsilon(2)),
            Nrv * abs(upsilon(1)) + Nrr * abs(upsilon(2));

        // Drag matrix
        D = Dl - Dn;

        // Ensure M is invertible
        if (M.determinant() == 0) {
            std::cerr << "Matrix M is not invertible: " << M << std::endl;
            exit(1);
        }

        // Environmental disturbances
        float si = sin(omega_e * time + phi) * (2 * M_PI / Lambda) * A;
        float F_wave_x = rho_water * g * B * L * draft * cos(wave_beta) * si;
        float F_wave_y = -rho_water * g * B * L * draft * sin(wave_beta) * si;

        // Wind effects
        float uw = Vw * cos(beta_w - eta(2));
        float vw = Vw * sin(beta_w - eta(2));
        float Vrw = sqrt(uw * uw + vw * vw);
        float F_wind_x = 0.5 * rho_air * Vrw * Vrw * Cx * Aw;
        float F_wind_y = 0.5 * rho_air * Vrw * Vrw * Cy * Alw;

        // Ocean current effects
        float current_velocity_x = Vc * cos(current_angle);
        float current_velocity_y = Vc * sin(current_angle);

        // Total disturbances
        float Fx = T(0) - (Xu + Xuu * abs(upsilon(0))) + F_wave_x + F_wind_x +
                   current_velocity_x;
        float Fy = T(1) - (Yv * upsilon(1) + Yr * upsilon(2)) + F_wave_y +
                   F_wind_y + current_velocity_y;

        // Acceleration vector [u' v' r']
        upsilon_dot = M.inverse() *
                      (Vector3f(Fx, Fy, T(2)) - (C * upsilon) - (D * upsilon) +
                       Delta);

        // Transformation matrix
        J << cos(eta(2)), -sin(eta(2)), 0, 
             sin(eta(2)), cos(eta(2)), 0, 
             0, 0, 1;

        eta_dot = J * upsilon_dot; // transformation into local reference frame
                                   // [x' y' psi']

        // Normalize the angle to [-pi, pi]
        eta_dot(2) = fmod(eta_dot(2) + M_PI, 2 * M_PI);
        if (eta_dot(2) < 0)
            eta_dot(2) += 2 * M_PI;
        eta_dot(2) -= M_PI;

        // Check for invalid eta_dot values
        if (isnan(eta_dot(0)) || isnan(eta_dot(1)) || isnan(eta_dot(2))) {
            std::cerr << "Invalid eta_dot values: " << eta_dot.transpose()
                      << std::endl;
            std::cerr << "upsilon values: " << upsilon.transpose() << std::endl;
            std::cerr << "upsilon_dot values: " << upsilon_dot.transpose()
                      << std::endl;
            std::cerr << "Matrix J: " << J << std::endl;
            std::cerr << "Matrix M: " << M << std::endl;
            std::cerr << "Matrix C: " << C << std::endl;
            std::cerr << "Matrix D: " << D << std::endl;
            std::cerr << "Vector T: " << T.transpose() << std::endl;
            std::cerr << "Vector Delta: " << Delta.transpose() << std::endl;
            exit(1);
        }

        return eta_dot;
    }

    void updateForces(float force_u, float force_r) {
        // Compute total thrust and thrust difference
        float T_total = force_u;   // Total thrust from velocity control
        float T_diff = force_r * B; // Thrust difference from yaw control (considering separation)

        // Allocate thrusts to port and starboard thrusters
        Tport = (T_total + T_diff) / (2 * c);
        Tstbd = (T_total - T_diff) / (2 * c);
    }
};

//---------------------------------------------------------------------------------------------------------

// Run MPC for controlling the ASV
std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>,
           std::vector<float>, std::vector<float>, std::vector<float>>
methodRungeKuttaMPC() {
    DynamicModel dynamicModel;

    std::vector<float> diffEq1;
    std::vector<float> diffEq2;
    std::vector<float> diffEq3;
    std::vector<float> diffEq4;
    std::vector<float> diffEq5;
    std::vector<float> diffEq6;
    std::vector<float> thrustPort;
    std::vector<float> thrustStarboard;
    std::vector<float> time;

    // Init values
    Vector3f upsilon(0.0, 0.0, 0.0);
    Vector3f eta_dot(0.0, 0.0, 0.0);
    float dt = 0.01;
    float t = 0.0;

    diffEq1.push_back(eta_dot(0));
    diffEq2.push_back(eta_dot(1));
    diffEq3.push_back(eta_dot(2));
    diffEq4.push_back(0);
    thrustPort.push_back(0);
    thrustStarboard.push_back(0);
    time.push_back(t);

    float setpoint_u = 0.2;   // target speed (m/s)
    float setpoint_psi = 0.0; // target heading (radians)

    float prev_control_u = 0.0;
    float prev_control_psi = 0.0;

    for (int ii = 0; ii < 1000; ii++) {
        t = t + dt;

        // yaw sinusoidal function
        setpoint_psi = 0.25 * sin(0.025 * ii);

        // state prediction using MPC
        VectorXd predicted_states = VectorXd::Zero(horizon + 1);
        predicted_states[0] = upsilon(0); // Starting with current velocity

        VectorXd predicted_psi_states = VectorXd::Zero(horizon + 1);
        predicted_psi_states[0] = upsilon(2); // Starting with current yaw rate

        for (int jj = 1; jj <= horizon; jj++) {
            predicted_states[jj] =
                predicted_states[jj - 1] +
                0.1 * setpoint_u; // predicting future velocity states
            predicted_psi_states[jj] =
                predicted_psi_states[jj - 1] +
                0.1 * setpoint_psi; // predicting future yaw states
        }

        VectorXd state_errors = VectorXd::Zero(horizon);
        VectorXd psi_state_errors = VectorXd::Zero(horizon);

        for (int jj = 0; jj < horizon; jj++) {
            state_errors[jj] = predicted_states[jj + 1] - setpoint_u;
            psi_state_errors[jj] = predicted_psi_states[jj + 1] - setpoint_psi;
        }

        MatrixXd Q_matrix = MatrixXd::Identity(horizon, horizon) * Q;
        MatrixXd R_matrix = MatrixXd::Identity(horizon, horizon) * R;
        MatrixXd S_matrix = MatrixXd::Identity(horizon, horizon) * S;

        MatrixXd H = Q_matrix + R_matrix;
        VectorXd g = state_errors;
        VectorXd psi_g = psi_state_errors;

        // solve H * control_input_delta = -g
        VectorXd control_input_delta = H.ldlt().solve(-g);
        VectorXd psi_control_input_delta = H.ldlt().solve(-psi_g);

        // Compute control derivatives
        VectorXd control_derivatives = VectorXd::Zero(horizon);
        VectorXd psi_control_derivatives = VectorXd::Zero(horizon);

        control_derivatives[0] = control_input_delta[0] - prev_control_u;
        psi_control_derivatives[0] = psi_control_input_delta[0] - prev_control_psi;

        for (int jj = 1; jj < horizon; jj++) {
            control_derivatives[jj] = control_input_delta[jj] - control_input_delta[jj - 1];
            psi_control_derivatives[jj] = psi_control_input_delta[jj] - psi_control_input_delta[jj - 1];
        }

        // incorporate derivative costs
        H += S_matrix;
        g += S_matrix * control_derivatives;
        psi_g += S_matrix * psi_control_derivatives;

        // re-solve with updated cost function
        control_input_delta = H.ldlt().solve(-g);
        psi_control_input_delta = H.ldlt().solve(-psi_g);

        // control inputs from MPC
        float control_u = control_input_delta[0]; // Only first affect the decision
        float control_psi = psi_control_input_delta[0];

        // update control inputs
        prev_control_u = control_u;
        prev_control_psi = control_psi;

        // update
        dynamicModel.updateForces(control_u, control_psi);

        // Store thrust values
        thrustPort.push_back(dynamicModel.Tport);
        thrustStarboard.push_back(dynamicModel.Tstbd);

        Vector3f k1 = dynamicModel.function_1(upsilon, t);
        Vector3f k2 = dynamicModel.function_1(upsilon + dt / 2 * k1, t + dt / 2);
        Vector3f k3 = dynamicModel.function_1(upsilon + dt / 2 * k2, t + dt / 2);
        Vector3f k4 = dynamicModel.function_1(upsilon + dt * k3, t + dt);

        upsilon = upsilon + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
        eta_dot = dynamicModel.function_1(upsilon, t);

        std::cout << "Iteration " << ii << ": eta_dot = " << eta_dot.transpose()
                  << std::endl;
        diffEq1.push_back(eta_dot(0));
        diffEq2.push_back(eta_dot(1));
        diffEq3.push_back(eta_dot(2));
        diffEq4.push_back(setpoint_psi);
        diffEq5.push_back(control_u);
        diffEq6.push_back(control_psi);

        time.push_back(t);
    }

    return std::make_tuple(diffEq1, diffEq2, diffEq3, diffEq4,
                           thrustPort, thrustStarboard, time);
}

//---------------------------------------------------------------------------------------------------------

void plot2D(std::vector<float> xX, std::vector<float> yY,
            const std::string &title, const std::string &xlabel,
            const std::string &ylabel) {
    plt::title(title);
    plt::named_plot("solution", xX, yY);
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::legend();
    plt::show();
}

//---------------------------------------------------------------------------------------------------------
void plot2D2D(std::vector<float> xX, std::vector<float> yY1,
              std::vector<float> yY2, const std::string &title,
              const std::string &xlabel, const std::string &ylabel) {
    plt::title(title);
    plt::named_plot("thrust_port", xX, yY1);
    plt::named_plot("thrust_stb", xX, yY2);
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::legend();
    plt::show();
}

//---------------------------------------------------------------------------------------------------------

int main() {
    auto vdp = methodRungeKuttaMPC();
    auto xX = std::get<0>(vdp);
    auto yY = std::get<1>(vdp);
    auto psi = std::get<2>(vdp);
    auto psi_desired = std::get<3>(vdp);;
    auto thrustPort = std::get<4>(vdp);
    auto thrustStarboard = std::get<5>(vdp);
    auto time = std::get<6>(vdp);

    plot2D2D(time, psi, psi_desired,
             "MPC ASV with disturbances (waves:3m, wind 5m/s, currents:2.5m/s): "
             "Heading (Psi) vs Time",
             "Time (s)", "Heading (rad)");

    plot2D2D(time, thrustPort, thrustStarboard,
             "Thrust MPC ASV with disturbances (waves:3m, wind 5m/s, "
             "currents:2.5m/s): Thrust vs Time",
             "Time (s)", "Thrust");
}
