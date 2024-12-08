// Markus Buchholz, 2024
// g++ blue_dyn.cpp -o t -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#include <iostream>
#include <vector>
#include <tuple>
#include <eigen3/Eigen/Dense>
#include "matplotlibcpp.h"

namespace plt = matplotlibcpp;
using namespace std;
using namespace Eigen;

class DynamicModel
{
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

    float m = 16.0;   // mass = (14.5) boat + 2 * (0.75) batteries
    float Iz = 4.1; // moment of inertia
    float B = 0.41; // centerline-to-centerline separation
    float c = 0.78; // thruster correction factor

    Vector3f upsilon;
    Vector3f upsilon_dot_last;
    Vector3f upsilon_dot;
    Vector3f eta;
    Vector3f eta_dot_last;
    Vector3f eta_dot;
    Matrix3f M;
    Vector3f T;
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

    DynamicModel()
    {
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

    Vector3f function_1(Vector3f upsilon)
    {
        // hydrodynamic equations and parameter conditions
        Xu = -25;
        Xuu = 0;
        if (abs(upsilon(0)) > 1.2)
        {
            Xu = 64.55;
            Xuu = -70.92;
        }

        // Ensure upsilon(0) and upsilon(1) are not NaN
        if (isnan(upsilon(0)) || isnan(upsilon(1)))
        {
            std::cerr << "Invalid upsilon values: " << upsilon.transpose() << std::endl;
            exit(1);
        }

        Yv = 0.5 * (-40 * 1000 * abs(upsilon(1))) * (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) + 0.016 * (pow((0.27 / 0.09), 2)));
        Yr = 6 * (-3.141592 * 1000) * sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01;
        Nv = 0.06 * (-3.141592 * 1000) * sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01;
        Nr = 0.02 * (-3.141592 * 1000) * sqrt(pow(upsilon(0), 2) + pow(upsilon(1), 2)) * 0.09 * 0.09 * 1.01 * 1.01;

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
        CA << 0, 0, 2 * ((Y_v_dot * upsilon(1)) + ((Y_r_dot + N_v_dot) / 2) * upsilon(2)),
            0, 0, 0 - X_u_dot * m * upsilon(0),
            2 * (((0 - Y_v_dot) * upsilon(1)) - ((Y_r_dot + N_v_dot) / 2) * upsilon(2)), X_u_dot * m * upsilon(0), 0;

        // Coriolis matrix
        C = CRB + CA;

        // Drag matrix - linear
        Dl << 0 - Xu, 0, 0,
            0, 0 - Yv, 0 - Yr,
            0, 0 - Nv, 0 - Nr;

        // Drag matrix - nonlinear
        Dn << Xuu * abs(upsilon(0)), 0, 0,
            0, Yvv * abs(upsilon(1)) + Yvr * abs(upsilon(2)), Yrv * abs(upsilon(1)) + Yrr * abs(upsilon(2)),
            0, Nvv * abs(upsilon(1)) + Nvr * abs(upsilon(2)), Nrv * abs(upsilon(1)) + Nrr * abs(upsilon(2));

        // Drag matrix
        D = Dl - Dn;

        // Ensure M is invertible
        if (M.determinant() == 0)
        {
            std::cerr << "Matrix M is not invertible: " << M << std::endl;
            exit(1);
        }

        upsilon_dot = M.inverse() * (T - (C * upsilon) - (D * upsilon) + Delta); // acceleration vector [u' v' r']

        // Transformation matrix
        J << cos(eta(2)), -sin(eta(2)), 0,
            sin(eta(2)), cos(eta(2)), 0,
            0, 0, 1;

        eta_dot = J * upsilon_dot; // transformation into local reference frame [x' y' psi']

        // Normalize the angle to [-pi, pi]
        eta_dot(2) = fmod(eta_dot(2) + M_PI, 2 * M_PI);
        if (eta_dot(2) < 0)
            eta_dot(2) += 2 * M_PI;
        eta_dot(2) -= M_PI;

        // Check for invalid eta_dot values
        if (isnan(eta_dot(0)) || isnan(eta_dot(1)) || isnan(eta_dot(2)))
        {
            std::cerr << "Invalid eta_dot values: " << eta_dot.transpose() << std::endl;
            std::cerr << "upsilon values: " << upsilon.transpose() << std::endl;
            std::cerr << "upsilon_dot values: " << upsilon_dot.transpose() << std::endl;
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
};

std::tuple<std::vector<float>, std::vector<float>, std::vector<float>, std::vector<float>> methodRungeKuttaDiff()
{
    DynamicModel dynamicModel;
    std::vector<float> diffEq1;
    std::vector<float> diffEq2;
    std::vector<float> diffEq3;
    std::vector<float> time;

    // init values
    Vector3f upsilon(0.1, 0.1, 0.1); // Initial values for upsilon
    Vector3f eta_dot(0.1, 0.0, 0.1); // Initial values for eta_dot
    float dt = 0.01; // Time step
    float t = 0.0;  // init time

    diffEq1.push_back(eta_dot(0));
    diffEq2.push_back(eta_dot(1));
    diffEq3.push_back(eta_dot(2));
    time.push_back(t);

    for (int ii = 0; ii < 1000; ii++)
    {
        t = t + dt;
        
        Vector3f k1 = dynamicModel.function_1(upsilon);
        Vector3f k2 = dynamicModel.function_1(upsilon + dt / 2 * k1);
        Vector3f k3 = dynamicModel.function_1(upsilon + dt / 2 * k2);
        Vector3f k4 = dynamicModel.function_1(upsilon + dt * k3);

        upsilon = upsilon + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4);
        
        eta_dot = dynamicModel.function_1(upsilon);

        std::cout << "Iteration " << ii << ": eta_dot = " << eta_dot.transpose() << std::endl;
        diffEq1.push_back(eta_dot(0));
        diffEq2.push_back(eta_dot(1));
        diffEq3.push_back(eta_dot(2));
        time.push_back(t);
    }

    return std::make_tuple(diffEq1, diffEq2, diffEq3, time);
}

//---------------------------------------------------------------------------------------------------------

void plot2D(std::vector<float> xX, std::vector<float> yY, const std::string &title, const std::string &xlabel, const std::string &ylabel)
{
    plt::title(title);
    plt::named_plot("solution", xX, yY);
    plt::xlabel(xlabel);
    plt::ylabel(ylabel);
    plt::legend();
    plt::show();
}

//---------------------------------------------------------------------------------------------------------

int main()
{
    auto vdp = methodRungeKuttaDiff();
    auto xX = std::get<0>(vdp);
    auto yY = std::get<1>(vdp);
    auto psi = std::get<2>(vdp);
    auto time = std::get<3>(vdp);
    plot2D(time, xX, "X Position vs Time", "Time (s)", "X Position (m)");
    plot2D(time, yY, "Y Position vs Time", "Time (s)", "Y Position (m)");
    plot2D(time, psi, "Heading (Psi) vs Time", "Time (s)", "Heading (rad)");
}
