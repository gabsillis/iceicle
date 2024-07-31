return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 2,

	-- create a uniform mesh
	uniform_mesh = {
		nelem = { 2, 2 },
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
				"dirichlet", -- right side
				"spacetime-future", -- top side
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
		basis = "legendre",

		-- the quadrature type (optional: default = gauss)
		quadrature = "gauss",

		-- the basis function order
		order = 1,
	},

	-- describe the conservation law
	conservation_law = {
		-- the name of the conservation law being solved
		name = "spacetime-burgers",
		mu = 0.0,
		a_adv = { 0.20 },
		b_adv = { 0.0 },
	},

	-- initial condition
	initial_condition = function(x, t)
		-- return 0.0
		if t > 5 * (x - 0.5) then
			return 0.0
		else
			return 1.0
		end
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			function(x, t)
				if x < 0.5 then
					return 0.0
				else
					return 1.0
				end
			end,
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
		lambda_b = 1e-3,
		ivis = 1,
		tau_abs = 1e-15,
		tau_rel = 0,
		kmax = 50,
	},

	-- output
	output = {
		writer = "vtu",
	},
}
