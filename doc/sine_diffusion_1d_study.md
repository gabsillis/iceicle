
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
Time integration is done with RK3-TVD using a Fourier number of 0.0001 through option `--fo`.
We use:
$$
\mu = 1.0 \\
t_{f} = 1.0
$$
where $t_f$ is the final time.
The basis functions used are Legendre polynomials.
## Data
### Standard DDG
```
nx dgp1          dgp2           dgp3           dgp4           
2  0.262893571   0.0412573649   0.0243959323   0.000824445723 
4  0.0636081844  0.0142770782   0.00121340947  9.13430722e-05  
8  0.0165579431  0.00190644295  6.954314e-05   3.05927449e-06  
16 0.00417741741 0.000243693789 4.21576248e-06 9.76500519e-08  
32 0.00104667189 3.07065654e-05 2.6077164e-07  3.07044764e-09 
```

### DDGIC
DDGIC is run by using `--ddgic_mult=0.5`. The multiplier ($\sigma$) is from the 2023 Danis and Yan paper which gives a general form for DDGIC, 
but in a previous paper this coefficient was 0.5 which gives slightly better results in DGP1
```
nx dgp1          dgp2           dgp3           dgp4
2  0.265591462   0.0419605773   0.024368788    0.000803681355 
4  0.0824449545  0.0145741899   0.00120722602  9.18430908e-05  
8  0.027170573   0.00193508108  6.88325795e-05 3.07358828e-06 
16 0.00765793611 0.000245728865 4.17464082e-06 9.79118744e-08 
32 0.00201135591 3.08407736e-05 2.58739186e-07 3.0748137e-09 
```

### Interior Penalty
Interior Penalty is run by using `--interior_penalty`.
Note we use the $\beta_0$ DDG coefficient as the interior penalty coefficient, so DG(P1) is equivalent to standard DDG
```
nx dgp1          dgp2           dgp3           dgp4
2  0.262893571   0.0475606773   0.0290106132   0.000952407674  
4  0.0636081844  0.0175930309   0.00140523714  0.000111911005  
8  0.0165579431  0.00390484158  8.63017364e-05 6.16326363e-06 
16 0.00417741741 0.000942627192 5.38785297e-06 3.73127053e-07 
32 0.00104667189 0.000233492261 3.36699076e-07 2.31358939e-08 
```

## Plot
![Convergence Study](./study1d/convergence-study.png)


