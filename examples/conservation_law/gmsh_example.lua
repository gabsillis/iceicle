local gamma = 1.4
-- return the configuration table
return {

    -- specify the number of dimensions (REQUIRED)
    ndim = 2,

    -- read in a mesh from gmsh
    gmsh = {
        file = "../gmsh_meshes/naca.msh",
        bc_definitions = {
            -- 1: airfoil boundary
            { "slip wall",     0 },

            -- 2: top and bottom walls
            { "slip wall",     0 },

            -- 3: inlet
            { "dirichlet",     0 },

            -- 4: outlet
            { "extrapolation", 0 },

        },
    },

    -- define the finite element domain
    fespace = {
        -- the basis function type (optional: default = lagrange)
        basis = "legendre",

        -- the quadrature type (optional: default = gauss)
        quadrature = "gauss",

        -- the basis function order
        order = 1,
    },

    -- describe the conservation law
    conservation_law = {
        -- the name of the conservation law being solved
        name = "euler",
    },

    -- initial condition
    initial_condition = function(x, t)
        return { 1.0, 0.0, 0.0, 1.0 / gamma - 1 }
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            { 1.0, 0.0, 0.0, 1.0 / gamma - 1 }
        },
    },

    -- solver
    solver = {
        type = "rk3-tvd",
        cfl = 0.1,
        ntime = 40
    },

    -- output
    output = {
        writer = "vtu",
    }
}
