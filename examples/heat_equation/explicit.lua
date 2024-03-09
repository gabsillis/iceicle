-- NOTE: require libraries must be done on the c++ side

-- define the mesh as a uniform quad mesh
uniform_mesh = {
	-- specify the number of elements in each direction
	nelem = { 2, 2 },

	-- specify the bounding box of the uniform mesh domain
	bounding_box = {
		min = { 0.0, 0.0 },
		max = { math.pi, math.pi },
	},

	-- set boundary conditions
	boundary_conditions = {
		-- the boundary condition types
		-- in order of direction and side
		types = {
			"dirichlet", -- left side
			"neumann", -- bottom side
			"dirichlet", -- right side
			"neumann", -- top side
		},

		-- the boundary condition flags
		-- used to identify user defined state
		flags = {
			1, -- left
			1, -- bottom
			1, -- right
			1, -- top
		},
	},
	geometry_order = 1
}

-- define the finite element domain
fespace = {
	-- the basis function type (optional: default = lagrange)
	basis = "lagrange",

	-- the quadrature type (optional: default = gauss)
	quadrature = "gauss",

	-- the basis function order
	order = 4,
}

mu = 0.1;

-- initial condition
-- initial_condition = "zero"
initial_condition = function(x, y)
	return math.sin(x)
end

-- exact solution (transient)
exact_sol = function(x, y, t)
	return math.sin(x) * math.exp(-0.1);
end

-- boundary condition state to be used by the
-- types and flags set in the mesh
boundary_conditions = {
	dirichlet = {
		-- constant values (integ)
		values = {
			0.0, -- flag 1
		},
	},

	neumann = {
		-- constant values (integ)
		values = {
			0.0, -- flag 1
		},
	}
}

solver = {
	type = "rk3-ssp",
	cfl = 0.01,
	tfinal = 1
}
