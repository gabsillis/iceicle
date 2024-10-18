-- Example: Time-explicit solution of Sod's Shocktube Problem
-- Author: Gianni Absillis
--
-- Works in parallel (just mpirun the executable with this input file and mesh partitioning happens automagically)

local gamma = 1.4

return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 1,

	-- create a uniform mesh
	uniform_mesh = {
		nelem = { 1000 },
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
		order = 1,
	},

	-- describe the conservation law
	conservation_law = {
		-- the name of the conservation law being solved
		name = "euler",
	},

	-- initial condition
	initial_condition = function(x)
		if x < 0.5 then
			return { 1.0, 0.0, 1.0 / (gamma - 1) }
		else
			return { 0.125, 0.0, 0.1 / (gamma - 1) }
		end
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			{ 1.0,   0.0, 1.0 / (gamma - 1) },
			{ 0.125, 0.0, 0.1 / (gamma - 1) },
		},
	},

	-- solver
	solver = {
		type = "rk3-tvd",
		-- dt = 0.0001,
		cfl = 0.3,
		tfinal = 0.2,
		ivis = 50,
	},

	-- output
	output = {
		writer = "dat",
	},
}
