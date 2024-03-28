# ICEicle Build and Installation Guide
## Quickstart
A bare-bones version of the library (with unit tests build by default can be built with)
```bash
mkdir build 
cd build 
cmake .. 
make -j
```

## Third Party Library Options
### PETSc
`ICEICLE_USE_PETSC` : Interface ICEicle with PETSc - a library for linear and nonlinear sparse solvers on distributed memory platforms. Currently integrated features include:
- Finite Difference construction of Jacobians in a PETSc `Mat`
- Newton solver that uses `KSP` linear solvers

This library is located through the `PETSC_DIR` and `PETSC_ARCH` environment variables using pkgconfig.

### Lua 
`ICEICLE_USE_LUA` : Interface with the Lua C api through sol2
Note: this requires Lua development libraries


On Fedora
```
dnf install lua-devel
```

The sol2 dependency is resolved through a FetchContent call

## Miniapps
Similar to MFEM several mini-apps provide standalone functionality and are built upon the ICEicle library. These mini-apps are built when all of their third party library dependencies are met.

### Heat Equation Miniapp
This mini-app solves the heat equation in 2D.
$$
\begin{split}
\Delta u = 0 \quad \text{in } \Omega  \\
\nabla u \cdot \mathbf{n} = g_N \quad \text{on } \Gamma_N \\
u = g_D \quad \text{on } \Gamma_D
\end{split}
$$
The mesh, initial conditions, and boundary conditions are configurable in a lua input_deck. The default input deck is `iceicle.lua` in the directory that the executable is run from. This filename can be customized by the command line argument `--scriptfile=<filename>` or `--scriptfile <filename>`.
#### Dependencies 
- `ICEICLE_USE_LUA` : Required
- `ICEICLE_USE_PETSC` : Optional (needed for implicit solver)

#### Instructions
Build ICEicle with Lua and Petsc 
```bash
cd build 
cmake -DICEICLE_USE_LUA -DICEICLE_USE_PETSC ..
make -j
```
Navigate to the heat equation example directory and run the 2D example case.
```bash
cd <project top directory>
cd examples/heat_equation
../../build/bin/heat_eqn_miniapp
```
VTU output is generated in the `iceicle_data` directory and can  be viewed with Paraview or other visualization software. 
Petsc arguments such as the preconditioner (`-pc_type`) can be passed on the command line as usual.

### Sine-Diffusion Miniapp 
This standalone program has no third party library dependencies. 
This solves a sine diffusion problem in 1D on a domain from $0$ to $2\pi$ (see [sine diffusion study](./sine_diffusion_1d_study.md)). 
#### Arguments 
Below is an abbreviated list of command line arguments which can be set by `--arg=<value>` or `--arg <value>`. Flag arguments don't need a second argument and just need `--arg`
- `help` - print the help message and list all the arguments with their descriptions
- `nelem` - (real number) number of elements
- `order` - (real number) polynomial order of basis functions 
- `fo` - (real number) the Fourier number for timesteps
- `dt` - (real number) the timestep (if not using Fourier number)
