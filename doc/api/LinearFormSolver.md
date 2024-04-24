# Linear Form Solver
This solves equations with a weak form of $\int uv \;dx = \int fv \;dx$. Alternatively notated by L2 inner products with $\langle u, v \rangle = \langle f, v \rangle$

Construction of a `LinearFormSolver` requires an `FESpace` and a discretization templated by `disc_type`.

`disc_type` has the SFINAE requirement that it must have a function
```c++
domain_integral(
    const FiniteElement<T, IDX, ndim> &,
    NodeArray<T, ndim>&,
    elspan auto res
) -> void;
```
This function represents $\int fv\;dx$ and should store the result of this expression in res.

`disc_type` must also specifiy the `constexpr` number of vector components for $f$
```c++ 
static constexpr int disc_type::nv_comp 
```

### Usage
Construct the linear form solver with an `FESpace` and `disc_type`
```c++
FESpace<T, IDX, ndim> fespace;
disc_type discretization;
LinearFormSolver solver{fespace, discretization};
```

Solve for $u$ that satisfies the weak form by:
```c++
fespan u;
solver.solve(u);
```
