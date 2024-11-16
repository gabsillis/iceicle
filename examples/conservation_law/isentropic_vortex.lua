-- Example: Isentropic Vortex Transport
-- Author: Gianni Absillis (gabsill@ncsu.edu)
--
-- This example is from I do like CFD, Vol. 1 Second edition
-- Section 7.13.3
-- (see Figure 7.13.1 for coefficients)
-- The exact solution is in Section 6.3
--
-- Another reference for this example is
-- "A Survey of the Isentropic Euler Vortex Problem using High-Order Methods"
-- by Seth C. Spiegel et al.

-- Gas Properties
local gamma = 1.4
local Rgas = 287.052874

-- Problem coefficients
local alpha = 1
local K = 5.0
local x0 = -10
local y0 = -10
local tfinal = 5

-- Free stream properties
local rho_inf = 1.0
local p_inf = 1.0 / gamma
local u_inf = 2
local v_inf = 2

-- derived free stream properties
local umag2_inf = u_inf * u_inf + v_inf * v_inf
local rhoE_inf = p_inf / (gamma - 1) + 0.5 * rho_inf * umag2_inf
local csound_inf = math.sqrt(gamma * p_inf / rho_inf)
local T_inf = p_inf / (rho_inf * Rgas)


-- gives a function in space of the exact solution at time t
local exact = function(t)
	return function(x, y)
		local xbar = x - x0 - u_inf * t;
		local ybar = y - y0 - v_inf * t;
		local rbar = math.sqrt(xbar ^ 2 + ybar ^ 2)

		local u = u_inf - K / (2 * math.pi) * ybar * math.exp(alpha * (1 - rbar ^ 2) / 2)
		local v = v_inf + K / (2 * math.pi) * xbar * math.exp(alpha * (1 - rbar ^ 2) / 2)
		-- T / T_inf
		local T_hat = 1 -
			K ^ 2 * (gamma - 1) / (8 * alpha * math.pi ^ 2 * csound_inf ^ 2) * math.exp(alpha * (1 - rbar ^ 2))

		local rho = rho_inf * (T_hat) ^ (1 / (gamma - 1))
		local p = p_inf * (T_hat) ^ (gamma / (gamma - 1))

		local umag2 = u * u + v * v
		local rhoE = p / (gamma - 1) + 0.5 * rho * umag2
		return { rho, rho * u, rho * v, rhoE }
	end
end

return {
	ndim = 2,
	uniform_mesh = {
		nelem = { 40, 40 },
		bounding_box = { min = { -20, -20 }, max = { 20, 20 } },
		boundary_conditions = {
			types = {
				"riemann",
				"riemann",
				"riemann",
				"riemann"
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
		order = 1,

	},

	initial_condition = exact(0),

	conservation_law = {
		name = "euler",
		free_stream = { rho_inf, rho_inf * u_inf, rho_inf * v_inf, rhoE_inf },
	},

	solver = {
		type = "rk3-ssp",
		cfl = 0.1,
		tfinal = tfinal,
		ivis = 100,
		idiag = 100,
	},

	output = {
		writer = "vtu"
	},

	post = {
		exact_solution = exact(tfinal),
		tasks = {
			"l2_error",
			"l1_error",
			"plot_exact_projection"
		},
	}


}
