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
