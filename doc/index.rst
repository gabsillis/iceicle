.. ICEicle documentation master file, created by
   sphinx-quickstart on Thu Apr 25 12:04:09 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. role:: cpp(code)
   :language: cpp

Welcome to ICEicle's documentation!
===================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`search`

Introduction
============
ICEicle is a research finite element code aimed at implementing MDG-ICE.

Lua Interface
=============
The :code:`conservation_law` miniapp exposes functionality via input decks written in Lua to allow for dynamic setup of a variety of problems. 
The :code:`heat_eqn_miniapp` miniapp will follow some of these conventions but uses an outdated interface so may be different. 
Check the :code:`examples` directory for some example input files. 
To enable this functionality make sure cmake can access Lua development libraries and turn on the cmake option :code:`ICEICLE_USE_LUA`.
This will download the `sol2 <https://github.com/ThePhD/sol2>`_ library through the :code:`FetchContent` mechanism.

Lua input files should return a single table with all of the configuration options. 
These configuration options are split into modules for each overarching table (see below)
For example - the following input file solves a diffusion equation in 1d:

.. code-block:: lua
   :linenos:

   local fourier_nr = 0.0001

   local nelem_arg = 8;

   return {
      -- specify the number of dimensions (REQUIRED)
      ndim = 1,

      -- create a uniform mesh
      uniform_mesh = {
         nelem = { nelem_arg },
         bounding_box = {
               min = { 0.0 },
               max = { 2 * math.pi }
         },
         -- set boundary conditions
         boundary_conditions = {
               -- the boundary condition types
               -- in order of direction and side
               types = {
                  "dirichlet", -- left side
                  "dirichlet", -- right side
               },

               -- the boundary condition flags
               -- used to identify user defined state
               flags = {
                  0, -- left
                  0, -- right
               },
         },
         geometry_order = 1,
      },

      -- define the finite element domain
      fespace = {
         -- the basis function type (optional: default = lagrange)
         basis = "legendre",

         -- the quadrature type (optional: default = gauss)
         quadrature = "gauss",

         -- the basis function order
         order = 2,
      },

      -- describe the conservation law
      conservation_law = {
         -- the name of the conservation law being solved
         name = "burgers",
         mu = 1.0,

      },

      -- initial condition
      initial_condition = function(x)
         return math.sin(x)
      end,

      -- boundary conditions
      boundary_conditions = {
         dirichlet = {
               0.0,
         },
      },

      -- solver
      solver = {
         type = "rk3-tvd",
         dt = fourier_nr * (2 * math.pi / nelem_arg) ^ 2,
         tfinal = 1.0,
         ivis = 1000
      },

      -- output
      output = {
         writer = "dat"
      }
   }

==============
Dimensionality
==============

:code:`ndim` : specifies the number of dimensions.

This will affect the sizes of many input arrays and internally sets the ``ndim`` template parameter. 
For spacetime simulations this is the number of spatial dimensions +1 for the time dimension.

====
Mesh
====

------------
Uniform Mesh
------------

All simulations need a mesh to define and partition the geometric domain. Currently only uniform hyper-cube mesh generation is supported.

:code:`uniform_mesh` : creates a uniform mesh 

**Required Members**

* :code:`nelem` : the number of elements in each direction

  This is a table of size ``ndim`` and is ordered in axis order (x, y, z, ...)

* :code:`bounding_box` : the bounding box of the mesh 

   * :code:`min` : the minimal corner of the mesh - if the mesh is the bi-unit hypercube, this is the :math:`\begin{pmatrix} -1 & -1 & ... \end{pmatrix}^T` corner of the mesh

   * :code:`max` : the maximal corner of the mesh - if the mesh is the bi-unit hypercube, this is the :math:`\begin{pmatrix} 1 & 1 & ... \end{pmatrix}^T` corner of the mesh

* :code:`boundary_conditions` : table to define the boundary condition identifiers for the domain faces 

   The tables in this table follow the face number ordering for Hypercube-type elements. 
   Considering a domain :math:`[-1, 1]^n` - this ordering is first the :math:`-1` side for each dimension in order (x, y, z, ...), 
   then the :math:`+1` sides. 

   In 2D this is: -x (left), -y (bottom), +x (right), +y (top).

   * :code:`types` : the :cpp:enum:`iceicle::BOUNDARY_CONDITIONS` type (check for documentation entries with "Lua name" for the case insensitive name)

   * :code:`flags` : an integer flag to pass to the boundary condition. 
     For Dirichlet and Neumann boundary conditions, this is the (0-indexed) index of the value or function 
     in the table where these boundary conditions are configured later.

**Optional Members**

* ``geometry_order`` : the polynomial order of the basis functions defining the geometry 

   (defaults to 1)

=======
FESpace
=======

:code:`fespace` : this table defines the finite element space 

**Optional Members**

* ``basis`` the basis function type. Current options are:

   * :cpp:`"lagrange"` Nodal Lagrange basis functions (default)

   * :cpp:`"legendre"` Modal Legendre basis functions 

      uses a Q-type tensor product on hypercube elements

* ``quadrature`` the quadrature function type. Current options are:

   * :cpp:`"gauss"` Gauss Legendre Quadrature (default)

* ``order`` the polynomial order of the basis functions (defaults to 0)

.. note::
   This order must be between 0 and MAX_POLYNOMIAL_ORDER (see ``build_config.hpp``) 
   because the polynomials use integer templates for the order provide optimization opportunities

================
Conservation Law
================

The ``conservation_law`` module is specific to the Conservation Law Miniapp and specifies the conservation law being solved 
and some physical parameters.

**Required Members**

* ``name`` the name of the conservation law, current options are: 
   * :cpp:`burgers` : The viscous burgers equation 

      :math:`\frac{\partial u}{\partial t} + \frac{\partial (a_j u + b_j u^2)}{\partial x_j} = \mu\frac{\partial^2 u}{\partial x^2}`

   * :cpp:`spacetime-burgers` : The viscous burgers equation in spacetime (see :ref:`Spacetime DG`)

      :math:`\frac{\partial u}{\partial t} + \frac{\partial (a_j u + b_j u^2)}{\partial x_j} = \mu\frac{\partial^2 u}{\partial x^2}`

* ''mu'' The viscosity coefficient for burgers equation

* ``a_adv`` a table the size of the number of **spatial** dimensions for the linear advection term :math:`a_j` in burgers equation

* ``b_adv`` a table the size of the number of **spatial** dimensions for the nonlinear advection term :math:`b_j` in burgers equation

.. note::
   The Spacetime burgers equation will have ``ndim-1`` elements because of the one time dimension.

=================
Initial Condition
=================

``initial_condition`` (optional) should be a function that takes an ``ndim`` number of arguments
which represent the coordinates in the physical domain and return the value of the solution(s) at that point.

For single equation conservation laws, this will just be a single value.
For multiple equation conservation laws, this should return a table of values of the required size.

If not specified, this initializes the entire domain to 0.

===================
Boundary Conditions
===================

``boundary_conditions`` configures different boundary conditions. 

* ``dirichlet`` set Neumann boundary condition for each flag in order 

   These can be real values, or functions that take ``ndim`` arguments (to represent physical space coordinates) 
   and returns the perscribed value at this location


* ``neumann`` set Neumann boundary condition for each flag in order 

   These can be real values, or functions that take ``ndim`` arguments (to represent physical space coordinates) 
   and returns the perscribed value at this location

======
Solver
======

``solver`` specifies which solver to use when solving the pde. 
Implicit methods assume no method of lines for time, so are either steady state, or spacetime solutions.

* ``type`` the type of solver to use; current named solvers are:

   * :cpp:`"newton", "newtonls"` : Newtons method with optional linesearch (Implicit)

   * :cpp:`"gauss-newton"` : Regularized Gauss-Newton method :ref:`GaussNewtonPetsc`

   * :cpp:`"explicit_euler"` : Explicit Euler's method 

   * :cpp:`"rk3-ssp", "rk3-tvd"` : Three stage Runge-Kutta explicit time integration. Strong Stability Preserving (SSP) or Total Variation Diminishing (TVD) versions

``ivis`` The visualization (output) is run every ``ivis`` iterations of the solver

--------------------------
Explicit Solver Parameters
--------------------------

* ``dt`` set the timestep for explicit schemes

* ``cfl`` the CFL number to use to determine the timestep from the conservation law 

.. note::

   ``dt`` and ``cfl`` are mutually exclusive

* ``tfinal`` the final time value to terminate the explicit method

* ``ntime`` the number of timesteps to take 

.. note::
   ``tfinal`` and ``ntime`` are mutually exclusive


--------------------------
Implicit Solver Parameters
--------------------------

* ``tau_abs`` the absolute tolerance for the residual norm to terminate the solve 

* ``tau_rel`` the relative amount to the initial residual norm at which to terminate the solve

The solve stops when the residual is less than :math:`\tau_{abs} + \tau_{rel} ||r_0||`

* ``kmax`` the maximum number of nonlinear iterations to take 

* ``linesearch`` perform a linesearch along the direction calculated by the implicit solver
   * ``type`` the linesearch type. Currently only :cpp:`"wolfe"` or :cpp:`"cubic"` for cubic interpolation linesearch is supported.

   * ``kmax`` (optional) the maximum number of linesearch iterations (defaults to 5)

   * ``alpha_initial`` (optional) the initial multiplier in the direction (1.0 is the full newton step for Newton solver) (defaults to 1)

   * ``alpha_max`` (optional) the maximum multiplier for the direction (defaults to 10)

   * ``c1`` and ``c2`` (optional) linesearch coefficients (defaults to 1e-4 and 0.9 respectively)

----------------------
Solver Specific Params
----------------------

* ``regularization`` The regularization parameter :math:`\lambda` for regularized nonlinear solvers 
  such as :cpp:`"gauss-newton"` type. This can be a real value or a function 
  :code:`function f(k, res)` that takes an integer iteration number `k` and a real valued residual norm `res` 
  and returns a real value as the regularization parameter.

* ``form_subproblem_mat`` set to true if you want to explicitly form the subproblem matrix for :cpp:`"guass_newton"` type

======
Output
======

``output`` specifies the output format for solutions (saves in the ``iceicle_data`` directory)

* ``vtu`` Paraview vtu file (2D and 3D only)

* ``dat`` Space separated values along the solution in 1D (1D only)

API
===

===============
Transformations
===============
Transformations are responsible for taking coordinates from a reference domain to a physical domain.
We denote the Physical domain with :math:`\Omega` and the reference domain with :math:`\hat\Omega`. 
Similarly the physical domain of a given element is notated :math:`\mathcal{K}` 
and the reference element domain with :math:`\hat{\mathcal{K}}`

==================
Geometric Entities
==================
There are two primary abstractions iceicle defines for geometric entities used in finite element computations:
:cpp:class:`iceicle::GeometricElement`, which represents the physical domain in :math:`\mathbb{R}^d`, and 
:cpp:class:`iceicle::Face`, which represents the physical domain of the intersection of two 
:cpp:class:`iceicle::GeometricElement` s. 

---------------------------------
Element Coordinate Transformation
---------------------------------
:cpp:class:`iceicle::GeometricElement` implementations must implement the transformation :math:`T:\mathbf{\xi}\mapsto\mathbf{x}`
from a reference domain :math:`\hat{\mathcal{K}} \subset \mathbb{R}^d` 
to the physical domain :math:`\mathcal{K} \subset \mathbb{R}^d` 
where :math:`\mathbf{\xi}\in\hat{\mathcal{K}}, \mathbf{x}\in\mathcal{K}`. 
The degrees of freedom that define the physical domain are termed "nodes".
A :cpp:class:`iceicle::GeometricElement` will just store the indices to the coordinate degrees of freedom 
which are stored in an :cpp:type:`iceicle::NodeArray`.

:cpp:func:`iceicle::GeometricElement::transform` represents the transformation :math:`T` provided the :cpp:type:`iceicle::NodeArray`.

:cpp:func:`iceicle::GeometricElement::Jacobian` represents the Jacobian of the transformation :math:`\mathbf{J} = \frac{\partial T}{\partial \mathbf{\xi}}`, 
alternatively :math:`\mathbf{J}` can be written as :math:`\frac{\partial \mathbf{x}}{\partial \mathbf{\xi}}`.

:cpp:func:`iceicle::GeometricElement::Hessian` represents the hessian of the transformation 
:math:`\mathbf{H} =\frac{\partial^2 T}{\partial \mathbf{\xi}\partial \mathbf{\xi}}` or :math:`\frac{\partial \mathbf{x}}{\partial \mathbf{\xi}\partial \mathbf{\xi}}`.


.. tikz:: Element Transformation
   :libs: arrows

   \path[shape=circle]
   (-1,-1) node[draw,scale=0.5](a1){}
   ( 1,-1) node[draw,scale=0.5](a2){}
   ( 1, 1) node[draw,scale=0.5](a3){}
   (-1, 1) node[draw,scale=0.5](a4){};
   \draw[thick] (a1) -- (a2) -- (a3) -- (a4) -- (a1);

   \draw[->] (-2,-1.5) -- (-1.5,-1.5) node[anchor=west, scale=0.7]{$\xi$};
   \draw[->] (-2,-1.5) -- (-2.0,-1.0) node[anchor=south, scale=0.7]{$\eta$};

   \draw[->, thick] (1.5, 0.0) -- (3.0, 0.0);

   \path[shape=circle]
   (4.0,-1) node[draw,scale=0.5](b1){}
   ( 5.8,-0.7) node[draw,scale=0.5](b2){}
   ( 6.0, 0.9) node[draw,scale=0.5](b3){}
   (4.2, 1.1) node[draw,scale=0.5](b4){};
   \draw[thick] (b1) -- (b2) -- (b3) -- (b4) -- (b1);

   \draw[->] (3.0,-1.5) -- (3.5,-1.5) node[anchor=west, scale=0.7]{$x$};
   \draw[->] (3.0,-1.5) -- (3.0,-1.0) node[anchor=south, scale=0.7]{$y$};

-------------------
Element Node Access
-------------------
Access to the indices of the nodes is provided in the following interfaces:

:cpp:func:`iceicle::GeometricElement::nodes` gives a pointer to the start of the array of indices.

:cpp:func:`iceicle::GeometricElement::nodes_span` gives the array of indices as a :cpp:class`std::span`

:cpp:func:`iceicle::GeometricElement::n_nodes` gives the size of the array of indices (the number of nodes)

------------------
Domain Definitions
------------------

Domains are specified by the domain type and polynomial order of basis functions for the nodes, accesible through 
:cpp:func:`iceicle::GeometricElement::domain_type` and :cpp:func:`iceicle::GeometricElement::geometry_order` respectively.

---------------
Face Generation
---------------
:code:`face_utils.hpp` contains a utility :cpp:func:`make_face` 
to generate faces by finding the intersection between two elements.

.. code-block:: cpp 
   :linenos:

   HypercubeElement<double, int, 2, 1> el0{{0, 1, 2, 3}};
   HypercubeElement<double, int, 2, 1> el2{{2, 3, 4, 5}};

   // find and make the face with vertices {2, 3}
   auto face_opt = make_face(0, 2, el0, el2);

   // unique_ptr to face is stored in the value() of optional
   std::unique_ptr<Face<double, int, 2>> face{std::move(face_opt.value())}; 

This can detect elements with different geometric polynomial orders because this operates on vertices.
The polynomial order of the face geometry is the minimum of the two element polynomial orders

==============
Finite Element
==============

==========
Data Views
==========

There are several view types for multidimensional data - inspired by :cpp:`std::mdspan`. 
The multidimensional data is indexed by finite-element specific terms.
A **degree of freedom** in ICEicle represents a basis function in the finite element space. 
These functions may take vector valued inputs and have vector valued outputs which are indexed by **vector component**.
For example, a node in 3D can be viewed as a geometric degree of freedom, with vector components for the x, y, and z positions.
Often coefficients are defined for each vector component - some other implementations may refer to this as a degree of freedom.

-----------
geo_dof_map
-----------

The struct :cpp:struct:`iceicle::geo_dof_map` represents a mapping to a subset of geometric (nodal) degrees of freedom


--------------
component_span
--------------
:cpp:class:`iceicle::component_span` represents a view over a subset of vector components.
Each degree of freedom (:cpp:`idof`) and vector component (:cpp:`idof`) index maps to an index in a one dimensional array.
This mapping is controlled by the :cpp:`LayoutPolicy`.


===============
Discretizations
===============

----------
Projection
----------
The Projection discretization is a linear form to project a function onto the space. 
The strong form of the equation is:

.. math::

   \mathbf{U} = \mathbf{f}(\mathbf{x})

with a weak form 

.. math::
   
   \int_\mathcal{K} \mathbf{U} v \; d\mathcal{K} = \int_\mathcal{K} \mathbf{f}(\mathbf{x}) v \; d\mathcal{K}

The right hand side of this equation is represented by :cpp:func:`iceicle::Projection::domain_integral()`.
Projections take a :cpp:type:`iceicle::ProjectionFunction` to represent the function :math:`\mathbf{f}`.

============
Spacetime DG
============

In ICEicle, Space-time DG is ostensibly treated as any other DG problem, with the last dimension representing the time dimension.
Discretizations take this into account and implement fluxes such that the last dimension is the time dimension. 
Consider a general conservation law as generally described with space and time derivatives separate:

.. math:: 

   \frac{\partial \mathbf{U}}{\partial t} + \frac{\partial \mathbf{F}_j(\mathbf{U})}{\partial x_j} = 0

Let :math:`\Omega_x \subset \mathbb{R}^d` represent the spatial domain, and :math:`\Omega_t \subset \mathbb{R}` represent the time domain. 
:math:`d` is the number of spatial dimensions
This conseration law can be rewritten in the combined space-time domain :math:`\Omega = \Omega_x \times \Omega_t`.

.. math::

   \nabla\cdot\mathcal{F} = 0

Where:

.. math:: 

   \nabla = \begin{pmatrix} \frac{\partial}{\partial x_1} & ... & \frac{\partial}{\partial x_d} & \frac{\partial}{\partial t} \end{pmatrix}^T

   \mathcal{F} = \begin{pmatrix} \mathbf{F}_1 & ... &\mathbf{F}_d & \mathbf{U} \end{pmatrix}^T

Interior fluxes and domain integrals just need to take this formulation into account in their implemenation. 
For the boundary, this results in two new boundary conditions.


The :cpp:enumerator:`iceicle::BOUNDARY_CONDITIONS::SPACETIME_FUTURE` represents the boundary with the future of the solution. Due to the nature of time 
this flux has to be purely upwind, so this is effectively equivalent to the :cpp:enumerator:`iceicle::BOUNDARY_CONDITIONS::EXTRAPOLATION` boundary condition.

The :cpp:enumerator:`iceicle::BOUNDARY_CONDITIONS::SPACETIME_PAST` represents the boundary with the past of the solution. Some implementations choose to do this 
as a Dirichlet boundary condition. However, since we want to process slabs of spacetime this is implemented by referencing 
another solution and the :code`fespace` it is defined on (which can be different). 

The first step is to find the node connectivity with :cpp:func:`iceicle::compute_st_node_connectivity`.
This takes two meshes in spacetime, ignores the time dimension and matches up nodes based on the boundary conditions 
and physical domain location. The resulting map when given a key of a node index in the current mesh,
will return the corresponding key in the "past" (connected) mesh.

.. code-block:: cpp 
   :linenos:

   using namespace iceicle;
   AbstractMesh<double, int, ndim> past_mesh{/* ... */};
   AbstractMesh<double, int, ndim> current_mesh{/* ... */};
   std::map<int, int> st_conn = compute_st_node_connectivity(past_mesh, current_mesh);

   int inode_past = st_conn[inode_curr]; 

The next step is to 

This also allows setting an initial condition by projecting the desired initial condition and using that as the past solution.

For example, using a sine wave as an initial condition.

.. code-block:: cpp
   :linenos:

   using namespace iceicle;

   // 1 dim space + 1 dim time
   static constexpr int ndim = 2;

   // lambda for the sine function
   ProjectionFunction<double, ndim, 1> func = [](const double* x, const double* out){
      out[0] = std::sin(x[0]);
   }

   // create the FESpace
   FESpace<double, int, ndim> fespace{/* ... */};

   std::map<int, int> node_connectivity = compute_st_node_connectivity(fespace.meshptr, fespace.meshptr);

Note that since the :cpp:`x` argument is just a pointer, initial conditions designed for just the spatial dimensions 
wil have no issue being used to initilize a time-slab.

=======
Solvers
=======

----------------
GaussNewtonPetsc
----------------

:cpp:class:`iceicle::solvers::GaussNewtonPetsc` is a nonlinear optimization solver. 
This uses a regularized version of the Gauss-Newton method (can be seen as a hybrid between Gauss-Newton and Levenberg-Marquardt) 
with linesearch.
In each nonlinear iteration, it solves the following subproblem:

.. math::

   \Big(\mathbf{J}^T \mathbf{J} + \lambda \mathbf{I}\Big)\pmb{p} = -\mathbf{J}^T \pmb{r}

and then performs a linesearch in the direction :math:`\pmb{p}` to minimize :math:`\pmb{r}`. 
This can be viewed as Newton's method on the least squares problem using :math:`\mathbf{J}^T\mathbf{J}` as the Hessian approximation.

Petsc is used for matrix operations.


API References
==============

.. doxygenclass:: iceicle::GeometricElement
   :members:

.. doxygenclass:: iceicle::Face
   :members:

.. doxygentypedef:: iceicle::NodeArray

.. doxygenenum:: iceicle::BOUNDARY_CONDITIONS

.. doxygenclass:: iceicle::Projection
   :members:

.. doxygentypedef:: iceicle::ProjectionFunction

.. doxygenfunction:: iceicle::compute_st_node_connectivity

.. doxygenclass:: iceicle::solvers::GaussNewtonPetsc
   :members:
