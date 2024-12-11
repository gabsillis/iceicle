-- Example: Steady state heat equation in 2D
-- Author: Gianni Absillis
--
-- Solves the academic problem of a 2D heat equation with dirichlet boundary conditions.
-- With error analysis

local function sinh(x)
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

local nelem_arg = 10
local mu = 1.0
local tfinal = 1.0

local non_uniform_coord = function(xmin, xmax, nelem, element_ratio)
	local dx = xmax / (nelem // 2 + element_ratio * (nelem // 2));

	-- Make the nodes
	local coord = { xmin };
	for inode = 1, nelem do
		if inode % 2 == 0 then
			coord[inode + 1] = coord[inode] + dx
		else
			coord[inode + 1] = coord[inode] + element_ratio * dx
		end
	end

	return coord
end

return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 2,

	-- create a uniform mesh
	uniform_mesh = {
		--		nelem = { nelem_arg, nelem_arg },
		--		bounding_box = {
		--			min = { 0.0, 0.0 },
		--			max = { 1.0, 1.0 },
		--		},

		directional_nodes = {
			non_uniform_coord(0, 1, 20, 3),
			non_uniform_coord(0, 1, 30, 1)
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
				0, -- left
				0, -- bottom
				1, -- right
				0, -- top
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
		mu = mu,
	},

	-- initial condition
	initial_condition = function(x)
		return math.sin(x)
	end,

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			1.0,
			function(x, y)
				return 1 + 0.1 * math.sin(math.pi * y)
			end,
		},
	},

	-- solver
	solver = {
		ivis = 1,
		type = "newton",
	},

	-- output
	output = {
		writer = "vtu",
	},

	-- post-processing
	post = {
		exact_solution = function(x, y)
			return 0.1 * sinh(math.pi * x) / sinh(math.pi) * math.sin(math.pi * y) + 1.0
		end,

		tasks = {
			"l2_error",
		},
	},
}
