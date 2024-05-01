# ICEicle
Finite Element research code for Moving Discontinous Galerkin with Interface Conservation Enforcement (MDG-ICE)

Main repository is at https://gitlab.com/giannigroup/iceicle (mirrored to GitHub)

Checkout additional doccumentation at https://iceicle.readthedocs.io/en/latest/

## Required Libraries/Components
- C++ 23 (Maintain builds for clang and gcc)
- NumericalToolbox (included as a submodule)

## Optional Libraries/Components 
- MPI (Automatically Detected)
- PETSc (Searches for `PETSC_DIR` and `PETSC_ARCH`) `-DICEICLE_USE_PETSC`
    - Petsc requires MPI
- Lua and sol3 (`-DICEICLE_USE_LUA`)
    - For more complex input decks such as in `examples/heat_equation`
- MFEM (Default: OFF) `-DICEICLE_USE_MFEM`
    - mfem is detected as a system library or when installed in cmake in `../iceicle_tpl/mfem/build`
- OpenGL+GLEW+Imgui (Default: ON) `-DICEICLE_USE_OPENGL`
    - OpenGL and GLEW are detected from system installation through cmake
    - Imgui is included as a submodule
- Matplotlibcpp (Default: ON) `-DENABLE_MATPLOTLIB_CPP`
    - Automatically downloaded through `FetchContent_Declare`
- GTest `-DENABLE_TESTING_ICEICLE`
    - Automatically downloaded through `FetchContent_Declare`

## Installation / QuickStart
See the [Installation Guide](./doc/install.md).

## Current Features
### Geometric Elements
##### Arbitrary n-dimensional p-order Hypercube Elements 
Nodes are defined in tensor product order (highest dimension first)

Support for Hypercube Face Types and corresponding transformations

### Basis Functions
##### Arbitrary order Lagrange Polynomials on Hypercube domains
Q-Type Tensor Product
##### Legendre Polynomials on Hypercube Domains 
Q-Type Tensor product, up to 10th order 
### Quadrature Rules 
##### Gauss-Legendre Quadrature
Defined for Hypercube Elements 

Arbitrary number of quadrature points

##### Grundmann-Moller Simplex Quadrature 
Defined for Simplices

Arbitrary number of quadrature points

### Meshing
##### Uniform Hypercube Mesh Generation
##### Mesh Perturbation by Function Callback

### Explicit Time Integration
##### Explicit Euler
##### RK3-SSP
3-stage strong stability preserving Runge-Kutta scheme
##### RK3-TVD
3-stage total variation diminishing Runge-Kutta scheme with reduced storage requirements.

### Implicit solver
##### Newton Linesearch
Uses Petsc for the linear solver - generates sparse Jacobian

