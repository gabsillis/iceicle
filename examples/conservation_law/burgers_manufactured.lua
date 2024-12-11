-- Example: Explicit Burgers Equation
-- Author: Gianni Absillis
--
-- Modeled after the test problem in Kercher et al. for Moving Discontinuous Galerkin with Interface Condition Enforcement for Viscous Flows
-- This solves the given burgers equation with dirichlet and extrapolation boundary conditions using a semi-discrete form on a uniform mesh

local fourier_nr = 0.0001

local nelem_arg = 10

local t_s = 0.5

local y_inf = 0.2

return {
    -- specify the number of dimensions (REQUIRED)
    ndim = 1,

    -- create a uniform mesh
    uniform_mesh = {
        nelem = { nelem_arg },
        bounding_box = {
            min = { 0.0 },
            max = { 1.0 }
        },
        -- set boundary conditions
        boundary_conditions = {
            -- the boundary condition types
            -- in order of direction and side
            types = {
                "extrapolation", -- left side
                "extrapolation", -- right side
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
        basis = "lagrange",

        -- the quadrature type (optional: default = gauss)
        quadrature = "gauss",

        -- the basis function order
        order = 5,
    },

    -- describe the conservation law
    conservation_law = {
        -- the name of the conservation law being solved
        name = "burgers",
        mu = 1e-3,
        a_adv = { 0.0 },
        b_adv = { 1.0 },

        source = function(x)
            return -2 * math.pi * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * x)
                - 1e-3 * 4 * math.pi ^ 2 * math.sin(2 * math.pi * x)
        end,
    },

    -- initial condition
    initial_condition = function(x)
        return math.sin(2 * math.pi * x)
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            0
        },
    },

    -- solver
    solver = {
        type = "rk3-tvd",
        dt = 1e-5,
        ntime = 5000,
        ivis = 1000
    },

    -- output
    output = {
        writer = "dat"
    }
}
