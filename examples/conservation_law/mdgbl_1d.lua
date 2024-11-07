-- Example: MDG 1D Boundary Layer
-- Author: Gianni Absillis (gabsill@ncsu.edu)
--
-- Steady state advection diffusion equation with dirichlet boundary conditions to form a boundary layer type problem
-- This is solved with MDG.

local mu_arg = 0.01
local v_arg = 1.0
local l = 1.0
local Pe = v_arg / mu_arg / l

return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 1,

	-- create a uniform mesh
	uniform_mesh = {
		nelem = { 10 },
		bounding_box = {
			min = { 0.0 },
			max = { l },
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
		return 1
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			0.0,
			1.0,
		},
	},

	-- MDG
	mdg = {
		ncycles = 1,
		ic_selection_threshold = function(icycle)
			return 0.0
		end,
	},

	-- solver
	solver = {
		type = "gauss-newton",
		form_subproblem_mat = true,
		linesearch = {
			type = "none",
		},
		lambda_b = 0.0,
		lambda_u = 0.0,
		lambda_lag = 0.01,
		ivis = 1,
		idiag = 1,
		tau_abs = 1e-8,
		tau_rel = 1e-8,
		kmax = 1000,
	},

	-- output
	output = {
		writer = "dat",
	},
}
