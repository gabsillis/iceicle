-- Example: flatplate
-- Author: Gianni Absillis (gabsill@ncsu.edu)
--
-- This example reads a mesh for the 2D flat plate from gmsh
-- solves the navier-stokes equations on the flatplate
-- to form a boundary layer

local gamma = 1.4
local rho = 1.0
local pressure = 1
local mach = 0.1

local csound = math.sqrt(gamma * pressure / rho)
print(csound)
local umag = csound * mach
local uadv = umag
local vadv = 0
local rhoe = pressure / (gamma - 1) - 0.5 * rho * umag * umag

-- return the configuration table
return {
    -- specify the number of dimensions (REQUIRED)
    ndim = 2,

    -- read in a mesh from gmsh
    gmsh = {
        file = "../gmsh_meshes/flatplate.msh",

        bc_definitions = {
            -- 1: slip wall
            { "slip wall",  0 },

            -- 2: no slip wall (isothermal)
            { "isothermal", 0 },

            -- 3: free stream
            { "dirichlet",  0 },
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
        name = "navier-stokes",

        isothermal_temperatures = { 273.15 },

        free_stream = { rho, rho * uadv, rho * vadv, rhoe },

    },

    -- initial condition
    initial_condition = function(x, y)
        return { rho, rho * uadv, rho * vadv, rhoe }
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            { rho, rho * uadv, rho * vadv, rhoe },
        },
    },

    -- solver
    solver = {
        type = "rk3-tvd",
        cfl = 0.001,
        ntime = 100,
        ivis = 1,
        idiag = 1,
    },

    -- solver
    --    solver = {
    --        type = "gauss-newton",
    --        form_subproblem_mat = true,
    --        linesearch = {
    --            type = "cubic",
    --            alpha_initial = 1.0,
    --            max_it = 10,
    --        },
    --        lambda_u = 1e-8,
    --        ivis = 1,
    --        tau_abs = 1e-10,
    --        tau_rel = 0,
    --        kmax = 1000,
    --    },

    -- output
    output = {
        writer = "vtu",
    }
}
