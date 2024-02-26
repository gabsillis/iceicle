### ICEicle Build and Installation
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
The mesh, initial conditions, and boundary conditions are configurable in a lua input_deck.
#### Dependencies 
- `ICEICLE_USE_LUA` : Required
- `ICEICLE_USE_PETSC` : Optional (needed for implicit solver)
