return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 1,

	-- create a uniform mesh
	uniform_mesh = {
		nelem = { 50 },
		bounding_box = {
			min = { 0.0 },
			max = { 1.0 },
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
	initial_condition = function(x)
		if x < 0.5 then
			return { 1.0, 3.5, 1.0 }
		else
			return { 1.0, 3.5, 1.0 }
		end
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			{ 1.0, 3.5, 1.0 },
			{ 1.0, 3.5, 1.0 },
		},
	},

	-- solver
	solver = {
		type = "rk3-tvd",
		dt = 0.00001,
		tfinal = 0.2,
		ivis = 10,
	},

	-- output
	output = {
		writer = "dat",
	},
}
