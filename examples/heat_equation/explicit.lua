-- NOTE: require libraries must be done on the c++ side

-- define the mesh as a uniform quad mesh
uniform_mesh = {
	-- specify the number of elements in each direction
	nelem = { 8, 1 },

	-- specify the bounding box of the uniform mesh domain
	bounding_box = {
		min = { 0.0, 0.0 },
		max = { 1.0, math.pi },
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
	order = 2,
}

mu = 1e-4;
a_adv = { 1.0, 0.0 };

-- initial condition
-- initial_condition = "zero"
initial_condition = function(x, y)
	-- return math.sin(x)
	return math.exp(-0.5 * ((x - 0.5) / 0.1) ^ 2) / 0.1;
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
	type = "rk3-tvd",
	dt = 1e-5,
	ntime = 10000,
	ivis = 1000
}
