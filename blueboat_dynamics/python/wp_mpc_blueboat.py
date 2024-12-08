# Markus Buchholz, 2024

import numpy as np
import matplotlib.pyplot as plt

# MPC parameters
horizon = 20  # Prediction horizon
Q = 0.0001  # State cost
R = 0.01  # Control cost
S = 0.001  # Derivative control cost

# Environmental disturbances
A = 3.0  # Wave amplitude
Vw = 5.0  # Wind speed (m/s)
Vc = 2.5  # Speed of the ocean current (m/s)
rho_water = 1000.0  # Density of water (kg/m^3)
g = 9.81  # Gravity (m/s^2)
L = 2.0  # Length of the vehicle
B = 2.0  # Breadth of the vehicle
draft = 0.5  # Draft of the vehicle
Lambda = 25000.0  # Wavelength
omega_e = 0.5  # Wave frequency
phi = 0  # Wave phase
current_angle = np.pi / 6  # Angle of the ocean current (radians)
rho_air = 1.225  # Density of air (kg/m^3)
Cx = 0.001  # Coefficient for wind drag in x
Cy = 0.001  # Coefficient for wind drag in y
Ck = 0.001  # Coefficient for yaw moment due to wind
Aw = 5.0  # Area for wind
Alw = 5.0  # Area for yaw wind
Hlw = 2.0  # Height for yaw wind
beta_w = np.pi / 4  # Wind direction relative to vehicle
wave_beta = np.pi / 4  # Wave direction

# Saturation limits
max_thrust = 100.0  #  maximum thrust limit
min_thrust = -100.0  # minimum thrust limit

# Normalize an angle to [-pi, pi]
def normalize_angle(angle):
    angle = np.fmod(angle + np.pi, 2 * np.pi)
    if angle < 0:
        angle += 2 * np.pi
    return angle - np.pi

# Saturation function
def apply_saturation(value, min_value, max_value):
    return max(min(value, max_value), min_value)

# Calculate spline coefficients
def compute_spline_coefficients(x, y):
    n = len(x) - 1  # Number of segments
    h = np.diff(x)
    a = np.array(y)

    alpha = np.zeros(n)
    l = np.ones(n + 1)
    mu = np.zeros(n)
    z = np.zeros(n + 1)

    for i in range(1, n):
        alpha[i] = (3 / h[i]) * (a[i + 1] - a[i]) - (3 / h[i - 1]) * (a[i] - a[i - 1])

    for i in range(1, n):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    b = np.zeros(n)
    c = np.zeros(n + 1)
    d = np.zeros(n)

    for j in range(n - 1, -1, -1):
        c[j] = z[j] - mu[j] * c[j + 1]
        b[j] = (a[j + 1] - a[j]) / h[j] - h[j] * (c[j + 1] + 2 * c[j]) / 3
        d[j] = (c[j + 1] - c[j]) / (3 * h[j])

    return np.vstack((a[:-1], b, c[:-1], d)).T

# Evaluate the spline at a given point t
def evaluate_spline(coeffs, x, t):
    return coeffs[0] + coeffs[1] * (t - x) + coeffs[2] * (t - x) ** 2 + coeffs[3] * (t - x) ** 3

# Calculate the spline path and yaw
def calculate_spline_path(waypoints, num_points=1000):
    x = np.array([wp[0] for wp in waypoints])
    y = np.array([wp[1] for wp in waypoints])

    x_coeffs = compute_spline_coefficients(x, x)
    y_coeffs = compute_spline_coefficients(x, y)

    path = []
    yaws = []

    for i in range(num_points):
        t = i / (num_points - 1) * (x[-1] - x[0]) + x[0]
        seg = np.searchsorted(x, t) - 1
        px = evaluate_spline(x_coeffs[seg], x[seg], t)
        py = evaluate_spline(y_coeffs[seg], x[seg], t)
        path.append([px, py])

        if i > 0:
            dx = px - path[i - 1][0]
            dy = py - path[i - 1][1]
            yaw = np.arctan2(dy, dx)
            yaws.append(yaw)

    return np.array(path), np.array(yaws)

# Dynamic model class for ASV
class DynamicModel:
    def __init__(self, initial_upsilon=None, initial_eta=None):
        self.Tstbd = 0
        self.Tport = 0

        self.delta_x = 0
        self.delta_y = 0

        self.Xu = -25
        self.Yv = 0
        self.Yr = 0
        self.Nv = 0
        self.Nr = 0
        self.X_u_dot = -2.25
        self.Y_v_dot = -23.13
        self.Y_r_dot = -1.31
        self.N_v_dot = -16.41
        self.N_r_dot = -2.79
        self.Xuu = 0
        self.Yvv = -99.99
        self.Yvr = -5.49
        self.Yrv = -5.49
        self.Yrr = -8.8
        self.Nvv = -5.49
        self.Nvr = -8.8
        self.Nrv = -8.8
        self.Nrr = -3.49

        self.m = 16.0  # mass = (14.5) boat + 2 * (0.75) batteries
        self.Iz = 4.1  # moment of inertia
        self.B = 0.41  # centerline-to-centerline separation
        self.c = 1.0  # thruster correction factor

        # Initialize state vectors
        self.upsilon = initial_upsilon if initial_upsilon is not None else np.zeros(3)
        self.eta = initial_eta if initial_eta is not None else np.zeros(3)
        self.upsilon_dot_last = np.zeros(3)
        self.upsilon_dot = np.zeros(3)
        self.eta_dot_last = np.zeros(3)
        self.eta_dot = np.zeros(3)

        self.M = np.array([[self.m - self.X_u_dot, 0, 0],
                           [0, self.m - self.Y_v_dot, -self.Y_r_dot],
                           [0, -self.N_v_dot, self.Iz - self.N_r_dot]])

        self.J = np.array([[np.cos(self.eta[2]), -np.sin(self.eta[2]), 0],
                           [np.sin(self.eta[2]), np.cos(self.eta[2]), 0],
                           [0, 0, 1]])

    def function_1(self, upsilon, time):
        Xu = -25
        Xuu = 0
        if np.abs(upsilon[0]) > 1.2:
            Xu = 64.55
            Xuu = -70.92

        Yv = 0.5 * (-40 * 1000 * np.abs(upsilon[1])) * \
            (1.1 + 0.0045 * (1.01 / 0.09) - 0.1 * (0.27 / 0.09) +
            0.016 * (np.power((0.27 / 0.09), 2)))
        Yr = 6 * (-3.141592 * 1000) * \
            np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
        Nv = 0.06 * (-3.141592 * 1000) * \
            np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01
        Nr = 0.02 * (-3.141592 * 1000) * \
            np.sqrt(np.power(upsilon[0], 2) + np.power(upsilon[1], 2)) * 0.09 * 0.09 * 1.01 * 1.01

        Delta = np.array([self.delta_x, self.delta_y, 0])
        Delta = np.linalg.inv(self.J) @ Delta

        T = np.array([self.Tport + self.c * self.Tstbd, 0, 0.5 * self.B * (self.Tport - self.c * self.Tstbd)])

        CRB = np.array([[0, 0, -self.m * upsilon[1]],
                        [0, 0, self.m * upsilon[0]],
                        [self.m * upsilon[1], -self.m * upsilon[0], 0]])

        CA = np.array([
            [0, 0, 2 * ((self.Y_v_dot * upsilon[1]) + ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2])],
            [0, 0, -self.X_u_dot * self.m * upsilon[0]],
            [2 * ((-self.Y_v_dot * upsilon[1]) - ((self.Y_r_dot + self.N_v_dot) / 2) * upsilon[2]), self.X_u_dot * self.m * upsilon[0], 0]
        ])

        C = CRB + CA

        Dl = np.array([[-Xu, 0, 0],
                    [0, -Yv, -Yr],
                    [0, -Nv, -Nr]])

        Dn = np.array([
            [Xuu * np.abs(upsilon[0]), 0, 0],
            [0, self.Yvv * np.abs(upsilon[1]) + self.Yvr * np.abs(upsilon[2]), self.Yrv * np.abs(upsilon[1]) + self.Yrr * np.abs(upsilon[2])],
            [0, self.Nvv * np.abs(upsilon[1]) + self.Nvr * np.abs(upsilon[2]), self.Nrv * np.abs(upsilon[1]) + self.Nrr * np.abs(upsilon[2])]
        ])

        D = Dl - Dn
        
        si = np.sin(omega_e * time + phi) * (2 * np.pi / Lambda) * A
        F_wave_x = rho_water * g * B * L * draft * np.cos(wave_beta) * si
        F_wave_y = -rho_water * g * B * L * draft * np.sin(wave_beta) * si

        uw = Vw * np.cos(beta_w - self.eta[2])
        vw = Vw * np.sin(beta_w - self.eta[2])
        Vrw = np.sqrt(uw ** 2 + vw ** 2)
        F_wind_x = 0.5 * rho_air * Vrw ** 2 * Cx * Aw
        F_wind_y = 0.5 * rho_air * Vrw ** 2 * Cy * Alw

        current_velocity_x = Vc * np.cos(current_angle)
        current_velocity_y = Vc * np.sin(current_angle)

        Fx = T[0] - (Xu + Xuu * np.abs(upsilon[0])) + F_wave_x + F_wind_x + current_velocity_x
        Fy = T[1] - (Yv * upsilon[1] + Yr * upsilon[2]) + F_wave_y + F_wind_y + current_velocity_y

        self.upsilon_dot = np.linalg.inv(self.M) @ (np.array([Fx, Fy, T[2]]) - (C @ upsilon) - (D @ upsilon) + Delta)

        self.J = np.array([[np.cos(self.eta[2]), -np.sin(self.eta[2]), 0],
                           [np.sin(self.eta[2]), np.cos(self.eta[2]), 0],
                           [0, 0, 1]])

        self.eta_dot = self.J @ self.upsilon_dot

        self.eta_dot[2] = normalize_angle(self.eta_dot[2])

        return self.eta_dot

    def update_forces(self, force_u, force_r):
        T_total = force_u
        T_diff = force_r * self.B

        self.Tport = (T_total + T_diff) / (2 * self.c)
        self.Tstbd = (T_total - T_diff) / (2 * self.c)

# run MPC
def run_mpc(waypoints, setpoint_u, initial_surge_speed, initial_sway_speed, initial_yaw_rate):
    # Set initial conditions
    initial_upsilon = np.array([initial_surge_speed, initial_sway_speed, initial_yaw_rate])
    initial_eta = np.array([0.0, 0.0, 0.0])  # Assuming initial position and heading at the origin

    dynamic_model = DynamicModel(initial_upsilon, initial_eta)

    diffEq1, diffEq2, diffEq3 = [], [], []
    diffEq4, diffEq5, diffEq6 = [], [], []
    thrustPort, thrustStarboard = [], []
    time = []

    path, yaw_path = calculate_spline_path(waypoints)

    upsilon = np.zeros(3)
    eta_dot = np.zeros(3)
    dt = 0.01
    t = 0.0

    diffEq1.append(eta_dot[0])
    diffEq2.append(eta_dot[1])
    diffEq3.append(eta_dot[2])
    diffEq4.append(0)
    thrustPort.append(0)
    thrustStarboard.append(0)
    time.append(t)

    prev_control_u = 0.0
    prev_control_psi = 0.0

    for ii in range(len(path) - 1):
        t += dt

        # introduce a smooth start to reduce spikes
        ramp_up_factor = min(1.0, t / 1.0)  # ramp up over the first second

        setpoint_psi = yaw_path[ii]

        predicted_states = np.zeros(horizon + 1)
        predicted_states[0] = upsilon[0]

        predicted_psi_states = np.zeros(horizon + 1)
        predicted_psi_states[0] = upsilon[2]

        for jj in range(1, horizon + 1):
            predicted_states[jj] = predicted_states[jj - 1] + 0.1 * setpoint_u
            predicted_psi_states[jj] = predicted_psi_states[jj - 1] + 0.1 * setpoint_psi

        state_errors = predicted_states[1:] - setpoint_u
        psi_state_errors = predicted_psi_states[1:] - setpoint_psi

        Q_matrix = np.eye(horizon) * Q
        R_matrix = np.eye(horizon) * R
        S_matrix = np.eye(horizon) * S

        H = Q_matrix + R_matrix
        g = state_errors
        psi_g = psi_state_errors

        control_input_delta = np.linalg.solve(H, -g)
        psi_control_input_delta = np.linalg.solve(H, -psi_g)

        control_derivatives = np.zeros(horizon)
        psi_control_derivatives = np.zeros(horizon)

        control_derivatives[0] = control_input_delta[0] - prev_control_u
        psi_control_derivatives[0] = psi_control_input_delta[0] - prev_control_psi

        for jj in range(1, horizon):
            control_derivatives[jj] = control_input_delta[jj] - control_input_delta[jj - 1]
            psi_control_derivatives[jj] = psi_control_input_delta[jj] - psi_control_input_delta[jj - 1]

        H += S_matrix
        g += S_matrix @ control_derivatives
        psi_g += S_matrix @ psi_control_derivatives

        control_input_delta = np.linalg.solve(H, -g)
        psi_control_input_delta = np.linalg.solve(H, -psi_g)

        control_u = ramp_up_factor * control_input_delta[0]
        control_psi = ramp_up_factor * psi_control_input_delta[0]

        prev_control_u = control_u
        prev_control_psi = control_psi

        dynamic_model.update_forces(control_u, control_psi)

        # apply saturation to the thrusts
        dynamic_model.Tport = apply_saturation(dynamic_model.Tport, min_thrust, max_thrust)
        dynamic_model.Tstbd = apply_saturation(dynamic_model.Tstbd, min_thrust, max_thrust)

        thrustPort.append(dynamic_model.Tport)
        thrustStarboard.append(dynamic_model.Tstbd)

        k1 = dynamic_model.function_1(upsilon, t)
        k2 = dynamic_model.function_1(upsilon + dt / 2 * k1, t + dt / 2)
        k3 = dynamic_model.function_1(upsilon + dt / 2 * k2, t + dt / 2)
        k4 = dynamic_model.function_1(upsilon + dt * k3, t + dt)

        upsilon += dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        eta_dot = dynamic_model.function_1(upsilon, t)

        diffEq1.append(eta_dot[0])
        diffEq2.append(eta_dot[1])
        diffEq3.append(eta_dot[2])
        diffEq4.append(setpoint_psi)
        diffEq5.append(control_u)
        diffEq6.append(control_psi)

        time.append(t)

    return diffEq1, diffEq2, diffEq3, diffEq4, thrustPort, thrustStarboard, time

# plot data
def plot_with_matplotlib(time, y1, y2, title, xlabel, ylabel, y1_label, y2_label):
    plt.figure()
    plt.plot(time, y1, label=y1_label)
    plt.plot(time, y2, label=y2_label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True)
    plt.show()

# Main function
def main():
    setpoint_u = 0.2  # target speed (m/s)

    waypoints = [
        [0.0, 0.0],
        [5.0, 0.5],
        [10.0, -1.3],
        [18.0, 1.8],
        [20.0, 0.3],
        [25.0, 1.3],
        [30.0, -0.3]
    ]

    # Initial conditions
    initial_surge_speed = 0.5  # example initial surge speed
    initial_sway_speed = 0.0
    initial_yaw_rate = 0.0

    diffEq1, diffEq2, diffEq3, diffEq4, thrustPort, thrustStarboard, time = run_mpc(waypoints, setpoint_u, 
                                                                                    initial_surge_speed, 
                                                                                    initial_sway_speed, 
                                                                                    initial_yaw_rate)

    plot_with_matplotlib(time, diffEq3, diffEq4, "MPC ASV with disturbances (waves:3m, wind 5m/s, currents:2.5m/s): Heading (Psi) vs Time", "Time (s)", "Heading (rad)", "solution yaw", "desired yaw")
    plot_with_matplotlib(time, thrustPort, thrustStarboard, "Thrust MPC ASV with disturbances (waves:3m, wind 5m/s, currents:2.5m/s): Thrust vs Time", "Time (s)", "Thrust", "thrustPort", "thrustStb")

if __name__ == "__main__":
    main()
