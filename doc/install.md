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

