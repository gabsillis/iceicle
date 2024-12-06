return {

    uniform_mesh = {
        nelem = { 5, 5 },
        bounding_box = {
            min = { 0.0, 0.0 },
            max = { 1.0, 1.0 },
        },

        -- quad_ratio = { 0.0, 0.0 },

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
                0, -- left
                0, -- bottom
                0, -- right
                0, -- top
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


    -- solver
    solver = {
        type = "gauss-newton",
        form_subproblem_mat = true,
        lambda_u = 1e-8,
        kmax = 100,
        tau_abs = 1e-10,
        ivis = 1,
        idiag = 1,
    },

    -- output
    output = {
        writer = "vtu",
    },

    -- post-processing
    post = {
        exact_solution = function(x, y)
            --- local rho = math.sin(2 * (x + y)) + 4
            local rho = 1 + x
            return { rho, rho * 1, rho * 1, rho ^ 2 }
        end,

        tasks = {
            "l2_error",
            "plot_exact_projection"
        },
    },
}
