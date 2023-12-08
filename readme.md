# ICEicle
Finite Element research code for Moving Discontinous Galerkin with Interface Conservation Enforcement (MDG-ICE)

## Required Libraries/Components
- C++ 20
- MPI
- NumericalToolbox (included as a submodule)

## Optional Libraries/Components 
- MFEM (Default: OFF) `-DICEICLE_USE_TPL`
    - mfem is detected as a system library or when installed in cmake in `../iceicle_tpl/mfem/build`
- OpenGL+GLEW+Imgui (Default: ON) `-DICEICLE_USE_OPENGL`
    - OpenGL and GLEW are detected from system installation through cmake
    - Imgui is included as a submodule
- Matplotlibcpp (Default: ON) `-DENABLE_MATPLOTLIB_CPP`
    - Automatically downloaded through `FetchContent_Declare`
- GTest `-DENABLE_TESTING_ICEICLE`
    - Automatically downloaded through `FetchContent_Declare`

## Current Features
### Geometric Elements
##### Arbitrary n-dimensional p-order Hypercube Elements 
Nodes are defined in tensor product order (highest dimension first)

Support for Hypercube Face Types and corresponding transformations

### Basis Functions
##### Arbitrary order Lagrange Polynomials on Hypercube domains

### Quadrature Rules 
##### Gauss-Legendre Quadrature
Defined for Hypercube Elements 

Arbitrary number of quadrature points

##### Grundmann-Moller Simplex Quadrature 
Defined for Simplices

Arbitrary number of quadrature points
