local fourier_nr = 0.0001


local t_s = 0.5

local y_inf = 0.2

return {
    -- specify the number of dimensions (REQUIRED)
    ndim = 1,

    -- create a uniform mesh
    uniform_mesh = {
        nelem = { 10 },
        bounding_box = {
            min = { 0.0 },
            max = { 1.0 }
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
                1, -- right
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
        name = "burgers",
        mu = 0.01,
        a_adv = { 1.0 },
        b_adv = { 0.0 },
    },

    -- initial condition
    initial_condition = function(x)
        return x
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            0.0,
            1.0
        },
    },

    -- solver
    solver = {
        type = "gauss-newton",
        ivis = 1,
        tau_abs = 1e-8,
        tau_rel = 0,
        kmax = 60,
        regularization = function(k, res)
            return 1
        end,
        form_subproblem_mat = false,
        verbosity = 0,
        linesearch = {
            kmax = 8,
            type = "none",
            alpha_initial = 1.0,
            alpha_max = 2.0,
        },
        mdg = {
            ncycles = 2,
            ic_selection_threshold = function(icycle)
                if (icycle % 2 == 0) then
                    return 0.0
                else
                    return 1e8
                end
            end,
        },
    },

    -- output
    output = {
        writer = "dat"
    }
}
