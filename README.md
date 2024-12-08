# BlueBoat Dynamic Model with Disturbances
![image](https://github.com/user-attachments/assets/11c43ef3-3e72-4a61-9c18-ec4e0eac3aa4)



## Introduction

This repository provides a comprehensive dynamic model of BlueBoat implemented in C++. <br>
The model with environmental disturbances (waves, wind, and ocean currents) uses the Runge-Kutta method to solve the differential equations governing the ASV's motion.<br>
Additionally, the repository includes a Simulink model equipped with PID controllers, which allows for parameter autotuning to optimize the ASV's performance.<br> 
The C++ program also includes **MPC** and **PID** controller, allowing for consistent control strategies across different platforms.<br> 


## Prerequisites
 - Eigen for C++ ```sudo apt install libeigen3-dev```
 - [Matlab/Simulink](https://uk.mathworks.com/products/simulink-online.html) (use Online version)
 - [MATLAB Drive](https://uk.mathworks.com/products/matlab-drive.html)
 - [GnuPlot](https://gnuplot.sourceforge.net/demo/) for Plotting in C++.

Install Gnuplot and Boost:

```bash
sudo apt-get install gnuplot-qt

sudo apt-get install libboost-iostreams-dev

sudo apt-get install libboost-all-dev
```


## Run C++ (MPC, PID)

### ASV dynamics with disturbances and MPC controller 

The C++ program simulates a BlueBoat using MPC while accounting for environmental disturbances such as wind, waves, and ocean currents.
The simulation models the ASV's dynamic response to these factors, demonstrating the BlueBoat ability to maintain a desired trajectory and heading.

The nonlinear dynamic model of the ASV captures hydrodynamic forces, environmental disturbances (waves, wind, and ocean currents), and actuators (thrust from port and starboard motors).
We use the  [Runge-Kutta method](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods) for solving ordinary differential equations (ODE). The Runge-Kutta method is used to predict the future states of the ASV, which is necessary for the MPC.

The core of MPC is to solve an optimization problem over a prediction horizon. The goal is to minimize a cost function that penalizes deviations from a desired trajectory. 

The optimization is solved using [Quadratic Programming](https://en.wikipedia.org/wiki/Quadratic_programming) (QP). The control input at each time step is obtained by solving a system of linear equations (Hessian matrix H and gradient vector g), with an additional derivative control cost matrix to ensure smoother inputs.

The system solves the optimization problem at each time step to determine the optimal control inputs (thrust for port and starboard). 


```bash
#compile
g++ mpc_blue_dyn_with_disturbances -o mpc_dyn_dist -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#run
./mpc_dyn_dist
```


#### Expected results

![image](https://github.com/user-attachments/assets/3851cbaa-acb8-4ca2-822f-ab4a5c3e35ab)


### Simulate ASV dynamics with disturbances and MPC controller feeding waypoints 

It is possible to simulate the BlueBoat's motion dynamics by feeding the waypoint for the path. The program computes the splines, which are consumed by the MPC controller.

Note. Plots can be inaccurate at the extremes due to issues with the ```mathplotlib.h``` library.


```bash
#compile
g++ wp_mpc_blueboat_controls.cpp -o wp_mpc_controls -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

or

g++ wp_mpc_blueboat_controls_gnuplot.cpp -o wp_mpc_controls -I/usr/include/eigen3 -I/usr/include/boost -lboost_system -lboost_filesystem -lboost_iostreams
#run
sudo ./wp_mpc_controls
```

#### Expected results

![image](https://github.com/user-attachments/assets/e83f27a6-771d-4661-9654-adc8f0df039f)


![image](https://github.com/user-attachments/assets/bd971812-9599-461e-a376-8a52609d51c3)



### Simulate Dynamics with control outputs (Thuster Port, Thruster StarBoard)


```bash
#compile
g++ mpc_blueboat_controls.cpp -o mpc_controls -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#run
./mpc_controls
```

#### Expected results

![image](https://github.com/user-attachments/assets/4be60a36-dfed-4d0a-8f96-04a6e4f4f8ff)


![image](https://github.com/user-attachments/assets/18645588-60ab-4ae7-98a2-d043827f3765)


### Compare MPC performance for different disturbances feeding waypoints 

```bash
#compile
g++ wp_mpc_blueboat_controls_gnuplot_params.cpp -o mpc_compare -I/usr/include/eigen3 -I/usr/include/boost -lboost_system -lboost_filesystem -lboost_iostreams

#run
sudo /mpc_controls
```
#### Expected results

![image](https://github.com/user-attachments/assets/e13954c8-12e1-4c3d-a7a8-ae57c69e81e1)


### ASV dynamics with MPC

The following program presents a Model Predictive Control (MPC) framework for controlling a BlueBoat. The objective is to optimize the vehicle's forward velocity and yaw control.

#### Dynamic Modeling

- State-Space Representation: The ASV dynamics are captured using a state-space model that considers hydrodynamic forces and moments, including added mass effects and damping forces.

- Hydrodynamic Parameters: The model incorporates specific hydrodynamic coefficients such as ```X_u, Y_v , N_r```.

#### Model Predictive Control (MPC)

- Extended Prediction Horizon: The MPC utilizes an extended prediction horizon to anticipate future states and optimize control inputs. 

- Quadratic Cost Function: The MPC algorithm minimizes a quadratic cost function that penalizes deviations from the desired state and excessive control effort. The cost function is defined as:

  $$J = \sum_{i=0}^{N-1} (x_i - x_{\text{desired}})^T Q (x_i - x_{\text{desired}}) + u_i^T R u_i$$

  where ```Q``` and ```R``` are weighting matrices that balance state tracking accuracy and control effort. The parameters ```Q``` and ```R``` are carefully tuned to achieve optimal performance.

- Optimization Process: The MPC solves a quadratic programming problem at each timestep to determine the optimal control inputs. This involves solving the equation:

  $$H \Delta u = -g$$

  where ```H``` is the Hessian matrix, ```Delta u``` is the change in control input, and ```g``` is the gradient vector derived from the state errors.

#### Thrust Allocation

- Differential Thrust Control: The control system computes the required thrust for the port and starboard thrusters based on the total thrust ```T_total``` and thrust differential ```T_diff```. The allocation equations are:

  $$T_{\text{port}} = \frac{T_{\text{total}} + T_{\text{diff}}}{2}$$

  $$T_{\text{starboard}} = \frac{T_{\text{total}} - T_{\text{diff}}}{2}$$

  This allocation ensures the ASV can effectively achieve both forward motion and yaw control.


```bash
#compile
g++ mpc_blue_dyn.cpp -o mpc_dyn -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#run
./mpc_dyn
```


#### Expected results

![image](https://github.com/user-attachments/assets/c2cff605-e0ce-4e4b-8261-1bd8d535c1f3)




#### ASV dynamics with PID
```bash
#compile
g++ pid_blue_dyn.cpp -o pid_dyn -I/usr/include/eigen3 -I/usr/include/python3.12 -I/usr/lib/python3/dist-packages/numpy/core/include -lpython3.12

#run
./pid_dyn
```
#### Expected results 

The PID values have been computed by running the Simulink application and incorporated into C++ prior to simulation.

![image](https://github.com/user-attachments/assets/f0ef11d0-267f-48cc-872f-174a5426dd97)



## Simulate BlueBoat in Simulink

Launch Simulink and open ```USVModel_with_controler.slx```. 

![image](https://github.com/user-attachments/assets/562352e5-418a-4a0f-b8c5-937eee132fb2)

### Expected results 

![image](https://github.com/user-attachments/assets/e6f4460b-febf-4adb-8570-41838d9469bf)


## Control System Design

The BlueBoat's control system (two thrusters) aims to manage both the forward velocity and yaw angle. Controllers independently control these two aspects.

### Control for Velocity and Yaw

##### Velocity Control:

The velocity controller adjusts the total thrust to achieve the desired speed:
```math
$$ e_{\text{velocity}} = v_{\text{desired}} - u $$
```
```math

$$ T_{\text{total}} = K_p^v \cdot e_{\text{velocity}} + K_i^v \cdot \int e_{\text{velocity}} \, dt + K_d^v \cdot \frac{d e_{\text{velocity}}}{dt} $$
```

##### Yaw Control:

The yaw controller adjusts the thrust difference to achieve the desired heading:
```math
 $$ e_{\text{yaw}} = \psi_{\text{desired}} - \psi $$
```
```math
$$ T_{\text{diff}} = K_p^\psi \cdot e_{\text{yaw}} + K_i^\psi \cdot \int e_{\text{yaw}} \, dt + K_d^\psi \cdot \frac{d e_{\text{yaw}}}{dt} $$
```

#### Thrust Allocation

The thrusts for the port and starboard thrusters are calculated as follows:

##### Thrust Port:
```math
$$ T_{\text{port}} = \frac{T_{\text{total}}}{2} + \frac{T_{\text{diff}}}{2} $$
```
##### Thrust Starboard:

```math

$$ T_{\text{starboard}} = \frac{T_{\text{total}}}{2} - \frac{T_{\text{diff}}}{2} $$
```
#### Explanation of Thrust Allocation

- Total Thrust ```T_total```: Ensures the ASV maintains the desired forward velocity by equally distributing thrust between both thrusters.
- Thrust Difference ```T_diff```: Adjusts the relative thrust between the two thrusters to create a turning moment for yaw control.


## Acknowledgment

- [Marine Systems Simulator](https://www.fossen.biz/MSS/)
- [Mathworks](https://blogs.mathworks.com/student-lounge/2019/03/18/modeling-robotic-boats-in-simulink/?s_tid=blogs_rc_2)
- [Unmanned surface vehicle robust tracking control using an adaptive super-twisting controller](https://www.sciencedirect.com/science/article/pii/S096706612400145X)
- [ASV design-modeling](https://uk.mathworks.com/videos/design-modeling-and-simulation-of-autonomous-underwater-vehicles-1619636864529.html)
- [Simulate Navigation Algorithms of An ASV](https://uk.mathworks.com/matlabcentral/fileexchange/118968-simulate-navigation-algorithms-of-an-asv)
- [ASV Simulate - Webinar](https://uk.mathworks.com/videos/matlab-day-for-marine-robotics-autonomous-systems-part-4-design-and-simulation-of-autonomous-surface-vessels-1661323350784.html)
- [Designing an MPC controller with Simulink](https://uk.mathworks.com/matlabcentral/fileexchange/68992-designing-an-mpc-controller-with-simulink?s_tid=srchtitle_support_results_2_MPC)
- [fileexchange](https://uk.mathworks.com/matlabcentral/fileexchange/)
- [matplotlib-cpp](https://github.com/lava/matplotlib-cpp)
