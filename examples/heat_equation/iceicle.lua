-- NOTE: require libraries must be done on the c++ side

-- define the mesh as a uniform quad mesh
uniform_mesh = {
	-- specify the number of elements in each direction
	nelem = { 2, 1 },

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
			"dirichlet", -- bottom side
			"dirichlet", -- right side
			"dirichlet", -- top side
		},

		-- the boundary condition flags
		-- used to identify user defined state
		flags = {
			1, -- left
			1, -- bottom
			-1, -- right
			1, -- top
		},
	},
	geometry_order = 1,
}

-- mesh_perturbation = "taylor-green"

-- define the finite element domain
fespace = {
	-- the basis function type (optional: default = lagrange)
	basis = "lagrange",

	-- the quadrature type (optional: default = gauss)
	quadrature = "gauss",

	-- the basis function order
	order = 2,
}


local bc2 = function(x, y)
	return 1 + 0.1 * math.sin(math.pi * x)
end
local bc3 = function(x, y)
	return 1 + 0.1 * math.sin(math.pi * 2 * x)
end
local bc4 = function(x, y)
	return 0.9 + 0.1 * (x - y)
end

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
				return 1 + 0.1 * math.sin(math.pi * y)
			end,
			-- flag -2
			bc2,
			-- flag -3
			bc3,
			-- flag -4
			bc4,
		},
	},
	neumann = {
		values = {
			0.0, -- flag 1
		}
	}
}
function sinh(x)
	if x == 0 then
		return 0.0
	end
	local neg = false
	if x < 0 then
		x = -x
		neg = true
	end
	if x < 1.0 then
		local y = x * x
		x = x
			+ x
			* y
			*
			(((-0.78966127417357099479e0 * y + -0.16375798202630751372e3) * y + -0.11563521196851768270e5) * y + -0.35181283430177117881e6)
			/
			(((0.10000000000000000000e1 * y + -0.27773523119650701667e3) * y + 0.36162723109421836460e5) * y + -0.21108770058106271242e7)
	else
		x = math.exp(x)
		x = x / 2.0 - 0.5 / x
	end
	if neg then
		x = -x
	end
	return x
end

exact_sol = function(x, y)
	return 0.1 * sinh(math.pi * x) / sinh(math.pi) * math.sin(math.pi * y) + 1.0
end

-- initial condition
-- initial_condition = "zero"
initial_condition = function(x, y)
	return x + y
end

solver = {
	type = "newton",
	linesearch = {
		type = "cubic",
		alpha_initial = 0.1,
	},
	verbosity = 1
}
