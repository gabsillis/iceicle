# Explicit Convergence Study
Convergence studies are an essential part of code verification processes. In this study we investigate convergence of the unsteady heat equation with explicit time integration.

## Governing Equation 
Consider in 1D the transient heat equation.
$$
\frac{\partial u}{\partial t} = \mu \frac{\partial^2 u}{\partial x^2}
$$
with boundary conditions:
$$
u(0, t) = 0 \\
u(\pi, t) = 0
$$
and initial condition:
$$
u(x, t=0) = \sin(x)
$$
The analytical solution for this problem is:
$$
u(x, t) = e^{-\mu t}\sin(x)
$$

## Problem Setup
We will investigate this using the 2D heat equation miniapp on uniform quad meshes.
The `examples/heat_equation/explicit.lua` script file holds the setup for this problem.
The left and right sides have Dirichlet boundary conditions and the top and bottom of the domain are essential (Neumann with 0 normal gradient) boundary conditions.
We will investigate meshes with 4, 8, 16, and 32 elements in each direction and compare to the exact solution at time $t=1$ with $\mu = 0.1$. 
We use a conservative CFL of 0.01 with RK3-SSP time integration.

## Data
Record the L2 error norm for each run:
```
nx dgp1            dgp2           dgp3           dgp4           
2  0.183909981     0.0279165763   0.00356375458  0.000307587443 
4  0.0466076481    0.00493615789  0.000281345818 1.92586206e-05 
8  0.0116306351    0.000694407443 1.69448418e-05 4.28855482e-06 
16 0.00290668955   9.12901929e-05 1.7149757e-06  1.07544178e-06  
32 0.000726611786  1.16985429e-05 3.52151086e-07 

```
