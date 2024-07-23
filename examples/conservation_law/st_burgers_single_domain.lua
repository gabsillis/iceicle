local fourier_nr = 0.0001

local nelem_arg = 8

local t_s = 0.5

local y_inf = 0.2

return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 2,

	-- create a uniform mesh
	uniform_mesh = {
		nelem = { nelem_arg, 6 },
		bounding_box = {
			min = { 0.0, 0.0 },
			max = { 1.0, 1.0 },
		},
		-- set boundary conditions
		boundary_conditions = {
			-- the boundary condition types
			-- in order of direction and side
			types = {
				"dirichlet", -- left side
				"dirichlet", -- bottom side
				"extrapolation", -- right side
				"spacetime-future", -- top side
			},

			-- the boundary condition flags
			-- used to identify user defined state
			flags = {
				0, -- left
				1, -- bottom
				0, -- right
				0, -- top
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
		name = "spacetime-burgers",
		mu = 1e-3,
		a_adv = { 0.0 },
		b_adv = { 1.0 },
	},

	-- initial condition
	initial_condition = function(x, t)
		-- return y_inf
		return 1.0 / (2 * math.pi * t_s) * math.sin(2 * math.pi * x) + y_inf
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			y_inf,
			function(x, t)
				-- return y_inf
				return 1.0 / (2 * math.pi * t_s) * math.sin(2 * math.pi * x) + y_inf
				-- return math.exp(-0.5 * ((x - 0.5) / 0.1) ^ 2) / 0.1;
			end,
		},
	},

	-- solver
	solver = {
		type = "gauss-newton",
		ivis = 1,
		tau_abs = 1e-15,
		tau_rel = 0,
		kmax = 1000,
		regularization = function(k, res)
			return 0.1
		end,
		form_subproblem_mat = false,
		verbosity = 0,
		linesearch = {
			kmax = 6,
			type = "none",
			alpha_initial = 1.0,
			alpha_max = 5,
		},
		mdg = {
			ncycles = 1,
			ic_selection_threshold = function(icycle)
				return 0.0
			end,
		},
	},

	-- output
	output = {
		writer = "vtu",
	},
}
