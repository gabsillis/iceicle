-- NOTE: require libraries must be done on the c++ side

-- define the mesh as a uniform quad mesh
uniform_mesh = {
    -- specify the number of elements in each direction
    nelem = { 100, 100 },

    -- specify the bounding box of the uniform mesh domain
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
            "dirichlet"  -- top side
        },

        -- the boundary condition flags
        -- used to identify user defined state
        flags = {
            1,  -- left
            -2, -- bottom
            -1, -- right
            1   -- top
        },
    }
}

-- define the finite element domain
fespace = {
    -- the basis function type (optional: default = lagrange)
    basis = "lagrange",

    -- the quadrature type (optional: deefault = gauss)
    quadrature = "gauss",

    -- the basis function order
    order = 1

}

-- initial condition
-- initial_condition = "zero"
initial_condition = function(x, y) return 1.0 + 0.05 * x * x + y * y end

local bc2 = function(x, y)
    return 1 + 0.1 * math.sin(math.pi * x)
end
-- boundary condition state to be used by the
-- types and flags set in the mesh
boundary_conditions = {
    dirichlet = {
        -- constant values (integ)
        values = {
            1.0, -- flag 1
            2.0, -- flag 2
        },

        -- callback functions
        callbacks = {
            -- flag -1
            function(x, y)
                return 1 + 0.1 * math.sin(math.pi * y)
            end,
            -- flag -2
            bc2
        }
    }
}
