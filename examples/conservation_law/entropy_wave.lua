local u_inf = 2
local v_inf = 2
local gamma = 1.4
local p_inf = 1.0 / gamma
local rho_inf = 1.0
local umag_inf = math.sqrt(u_inf ^ 2 + v_inf ^ 2)
local wave_size = 0.5

local ic = function(x, y)
	return {
		rho_inf + wave_size * math.sin(math.pi * (x + y)),
		rho_inf * u_inf,
		rho_inf * v_inf,
		p_inf / (gamma - 1) + 0.5 * rho_inf * umag_inf ^ 2
	}
end

return {
	ndim = 2,
	uniform_mesh = {
		nelem = { 20, 20 },
		bounding_box = { min = { -1, -1 }, max = { 1, 1 } },
		boundary_conditions = {
			types = {
				"extrapolation",
				"extrapolation",
				"extrapolation",
				"extrapolation"
			},
			flags = { 0, 0, 0, 0 },
			geometry_order = 1,
		},
	},

	-- define the finite element domain
	fespace = {
		-- the basis function type (optional: default = lagrange)
		basis = "lagrange",

		-- the quadrature type (optional: default = gauss)
		quadrature = "gauss",

		-- the basis function order
		order = 0,
	},

	initial_condition = ic,

	conservation_law = {
		name = "euler",
	},

	solver = {
		type = "rk3-tvd",
		cfl = 0.9,
		tfinal = 1.0,
		ivis = 1,
		idiag = 1,
	},

	output = {
		writer = "vtu"
	}
}
