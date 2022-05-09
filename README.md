# ROAR_MPC
Scripts used to develop MPC for the ROAR program at UC Berkeley

A video of the MPC module running on the easy map can be found here: https://www.youtube.com/watch?v=NKT-Q7Uw9yQ

And a video of the MPC module running on the Berkeley minor map can be found here: https://www.youtube.com/watch?v=1iVn_CoDxvE

## Required Packages
Pyomo was used for every optimization problem here and needs to be installed. The IPOPT solver also needs to be installed.

## Longitudinal Dynamics
The Longitudinal Dynamics folder holds scripts and the data used to find the parameters b, F_friction and C_D. More information can be found in section 3.2.1 of the Capstone Report.

## Lateral Dynamics
The Lateral Dynamics folder holds scripts and the data used to find the parameters mu, B and C in the Pacekja tire model. More information can be found in section 3.2.2 of the Capstone Report.

## Global Planner
The Global Planner folder holds a non-linear global planner script and the "Easy Map" track model. More information can be found in section 4.2 of the Capstone Report.