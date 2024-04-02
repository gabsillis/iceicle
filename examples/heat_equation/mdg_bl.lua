-- NOTE: require libraries must be done on the c++ side

-- define the mesh as a uniform quad mesh
uniform_mesh = {
	-- specify the number of elements in each direction
	nelem = { 10, 3 },

	-- specify the bounding box of the uniform mesh domain
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
			"neumann", -- bottom side
			"dirichlet", -- right side
			"neumann", -- top side
		},

		-- the boundary condition flags
		-- used to identify user defined state
		flags = {
			1, -- left
			1, -- bottom
			2, -- right
			1, -- top
		},
	},
	geometry_order = 1,
}
mu = 0.01
a_adv = { 1.0, 0.0 }

-- mesh_perturbation = "zig-zag"

-- define the finite element domain
fespace = {
	-- the basis function type (optional: default = lagrange)
	basis = "lagrange",

	-- the quadrature type (optional: default = gauss)
	quadrature = "gauss",

	-- the basis function order
	order = 2,
}

-- boundary condition state to be used by the
-- types and flags set in the mesh
boundary_conditions = {
	dirichlet = {
		-- constant values
		values = {
			1.0, -- flag 1
			2.0, -- flag 2
			0.9, -- flag 3
		},

		-- callback functions
		callbacks = {
			-- flag -1
			function(x, y)
				return x * x - y * y
			end,
		},
	},
	neumann = {
		values = {
			0.0, -- flag 1
		}
	}
}

solver = {
	type = "newton",
	mdg = {
		-- number of times to repeat the node selection + nonlinear solve process
		ncycles = 2,
		ic_selection_threshold = 1e-8,
	},
	verbosity = 4,
	kmax = 2,
	ivis = 1,
	linesearch = {
		type = "cubic",
		alpha_initial = 0.1,
		alpha_max = 1.0,
	}
}

-- initial condition
initial_condition = function(x, y)
	return 1 + x;
end
