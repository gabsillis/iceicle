return {
    -- specify the number of dimensions (REQUIRED)
    ndim = 2,

    -- create a uniform mesh
    uniform_mesh = {
        nelem = { 2, 1 },
        bounding_box = {
            min = { 0.0, 0.0 },
            max = { 1.0, 1.0 }
        },
        -- set boundary conditions
        boundary_conditions = {
            -- the boundary condition types
            -- in order of direction and side
            types = {
                "dirichlet", -- left side
                "dirichlet", -- bottom side
                "dirichlet", -- right side
                "dirichlet", -- top side
            },

            -- the boundary condition flags
            -- used to identify user defined state
            flags = {
                1,  -- left
                1,  -- bottom
                -1, -- right
                1,  -- top
            },
        },
        geometry_order = 1,
    },

    -- define the finite element domain
    fespace = {
        -- the basis function type (optional: default = lagrange)
        basis = "lagrange",

        -- the quadrature type (optional: default = gauss)
        quadrature = "gauss",

        -- the basis function order
        order = 2,
    },

    -- describe the conservation law
    conservation_law = {
        -- the name of the conservation law being solved
        name = "burgers"
    },

    -- initial condition
    initial_condition = function(x, y)
        print("x: ", x, " | y: ", y)
        return x + y
    end
}
