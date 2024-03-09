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
We will investigate meshes with 4, 8, 16, and 32 elements in each direction and compare to the exact solution at time $t=1$ with $\mu = 1$. 
We use a conservative CFL of 0.01 to make up for the explicit Euler time integration scheme.

## Data
Record the L2 error norm for each run:
```
nx dgp1            dgp2           dgp3           dgp4           dgp5
4  0.0291410261    0.00419562734 
8  
16 
32 

```
