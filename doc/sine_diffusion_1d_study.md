
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
We run the `sine_diffusion_1d` executable with arguments `--nelem` to control the number of elements and `--order` to control the polynomial order of the basis functions (note this is one less than the expected order of convergence). `--ivis` Can also be used to limit the amount of output.
Time integration is done with RK3-SSP using a Fourier number starting at 0.0001 through option `--fo` and decreasing until converged.
We use:
$$
\mu = 1.0 \\
t_{f} = 1.0
$$
where $t_f$ is the final time
## Data
### Standard DDG
#### ddg_standard.dat
```
nx dgp1          dgp2           dgp3           dgp4
2  0.39253572    0.0342310385   0.0295085698   0.000618673124
4  0.0865962955  0.0162980418   0.00146665915  0.000107503922
8  0.022298003   0.00215772759  9.16912292e-05 3.42449894e-06
16 0.00561251423 0.000275937015 7.77050204e-06 1.05995419e-07
32 0.00140545422 3.50002585e-05 3.60395859e-07 1.13590162e-08  
```

### DDGIC
```
nx dgp1          dgp2           dgp3           dgp4
2  0.40382591    0.0360444902   0.0294514095   0.000580089877 
4  0.125660094   0.0168739011   0.00142564585  0.000108381037 
8  0.047332525   0.0022157682   8.40767787e-05 3.42216429e-06 
16 0.014611397   0.000280198664 5.19002789e-06 1.059878e-07 
32 0.00399429849 3.52918274e-05 3.38055008e-07 5.4590666e-09 
```

### Interior Penalty
```
nx dgp1          dgp2           dgp3           dgp4
2  0.39253572    0.0451453519   0.0332393394   0.000618673124 
4  0.0865962955  0.0192983987   0.00143829301  0.000107503922 
8  0.022298003   0.00404716997  8.54828275e-05 3.42449894e-06 
16 0.00561251423 0.000955739058 5.31485229e-06 4.70040364e-07 
```

## Plot
![Convergence Study](./study1d/convergence-study.png)


