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
	user_mesh = {
		coord = {
			{ 0.0 },
			{ 0.9 },
			{ 1.0 },
		},

		elements = {
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 0, 1 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 1, 2 },
			},
		},

		boundary_faces = {
			{
				bc_type = "dirichlet",
				bc_flag = 0,
				nodes = { 0 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 2 },
			},

		},
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
		return (1 - math.exp(x * Pe)) / (1 - math.exp(Pe))
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
			return 0.0;
		end,
	},

	-- solver
	solver = {
		type = "gauss-newton",
		form_subproblem_mat = true,
		linesearch = {
			type = "none",
		},
		lambda_b = 1.0,
		lambda_u = 1e-6,
		ivis = 1,
		tau_abs = 1e-10,
		tau_rel = 0,
		kmax = 10,
	},

	-- output
	output = {
		writer = "dat",
	},
}
