-- Example: Steady state heat equation in 2D on a mixed uniform mesh
-- of quads and tris
-- Author: Gianni Absillis
--
--
-- Solves the academic problem of a 2D heat equation with dirichlet boundary conditions.
-- On a mesh of mixed quad and triangular elements generated uniformly
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

local nelem_arg = 20
local mu = 1.0

return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 2,

	-- create a uniform mesh
	mixed_uniform_mesh = {
		nelem = { nelem_arg, nelem_arg },
		bounding_box = {
			min = { 0.0, 0.0 },
			max = { 1.0, 1.0 },
		},

		quad_ratio = { 0.0, 0.0 },

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
	},

	-- manually do some edge flips
	mesh_management = {
		edge_flips = { 0 },
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
		sigma_ic = 0.0
	},

	-- initial condition
	initial_condition = function(x, y)
		return 0.1 * sinh(math.pi * x) / sinh(math.pi) * math.sin(math.pi * y) + 1.0
		-- return math.sin(x)
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
		idiag = 1,
		type = "newton",
		kmax = 5,
	},

	--	solver = {
	--		ivis = 1,
	--		idiag = 1,
	--		type = "rk3-tvd",
	--		ntime = 100,
	--		dt = 0.00001
	--	},

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
