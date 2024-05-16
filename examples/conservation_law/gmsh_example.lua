-- return the configuration table
return {

    -- specify the number of dimensions (REQUIRED)
    ndim = 2,

    -- read in a mesh from gmsh
    gmsh = {
        file = "../gmsh_meshes/naca.msh",
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
        a_adv = { 0.0, 0.0 },
        b_adv = { 0.0, 0.0 },
    },

    -- initial condition
    initial_condition = function(x, t)
        return 0
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            0.0,
            1.0,
        },
    },

    -- solver
    solver = {
        type = "newton",
        ivis = 1,
        tau_abs = 1e-8,
        tau_rel = 0,
        kmax = 5,
    },

    -- output
    output = {
        writer = "vtu"
    }
}
