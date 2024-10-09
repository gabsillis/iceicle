local gamma = 1.4
local rho = 1.0
local pressure = 1
-- local mach = 0.38
local mach = 0.7
local aoa = .01;

local csound = math.sqrt(gamma * pressure / rho)
local umag = csound * mach
local uadv = umag * math.cos(aoa)
local vadv = umag * math.sin(aoa)
local rhoe = pressure / (gamma - 1) - 0.5 * rho * umag * umag

-- return the configuration table
return {

    -- specify the number of dimensions (REQUIRED)
    ndim = 2,

    -- read in a mesh from gmsh
    gmsh = {
        file = "../gmsh_meshes/naca.msh",

        -- WARNING: right now this is in order of definition of 2D entities (curves/lines)
        -- TODO: look into actually matching up physical tags
        bc_definitions = {
            -- 1: airfoil boundary
            { "slip wall", 0 },

            -- 2: bottom wall
            { "dirichlet", 0 },

            -- 3: outlet
            { "dirichlet", 0 },

            -- 4: top wall
            { "dirichlet", 0 },

            -- 5: inlet
            { "dirichlet", 0 },
        },
    },

    -- define the finite element domain
    fespace = {
        -- the basis function type (optional: default = lagrange)
        basis = "legendre",

        -- the quadrature type (optional: default = gauss)
        quadrature = "gauss",

        -- the basis function order
        order = 0,
    },

    -- describe the conservation law
    conservation_law = {
        -- the name of the conservation law being solved
        name = "euler",
    },

    -- initial condition
    initial_condition = function(x, t)
        return { rho, rho * uadv, rho * vadv, rhoe }
    end,

    -- boundary conditions
    boundary_conditions = {
        dirichlet = {
            { rho,     rho * uadv, rho * vadv, rhoe },
            { 2 * rho, rho * uadv, rho * vadv, rhoe },
            { 3 * rho, rho * uadv, rho * vadv, rhoe },
            { 4 * rho, rho * uadv, rho * vadv, rhoe },
            { 5 * rho, rho * uadv, rho * vadv, rhoe },
        },
    },

    -- solver
    --    solver = {
    --        type = "rk3-tvd",
    --        cfl = 0.5,
    --        ntime = 50000,
    --        ivis = 100
    --    },

    -- solver
    solver = {
        type = "gauss-newton",
        form_subproblem_mat = true,
        linesearch = {
            type = "cubic",
            alpha_initial = 1.0,
            max_it = 10,
        },
        lambda_u = 1e-8,
        ivis = 1,
        tau_abs = 1e-10,
        tau_rel = 0,
        kmax = 1000,
    },

    -- output
    output = {
        writer = "vtu",
    }
}
