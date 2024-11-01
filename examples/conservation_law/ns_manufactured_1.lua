local gamma = 1.4;
local Rgas = 0.287;      -- kJ/(kg*K)
local mu_inf = 1.716e-5; --kg/ms
local T_ref = 273.15;    --K
local T_s = 110.4;       --K
local Pr = 0.72;

function manufactured_sol(x, y)
	return { 1, x, y, 1 }
end

function source(x, y)
	return
		-2,
		-(3 * x - x * (gamma - 1) - (2 * x * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (3 * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) + (x * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas))),
		-(3 * y - y * (gamma - 1) - (2 * y * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (3 * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) + (y * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas))),
		-((2 * gamma * (T_ref + T_s) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (Pr * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) - x ^ 2 * (gamma - 1) - y ^ 2 * (gamma - 1) - (4 * (T_ref + T_s) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (3 * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) - 2 * (gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1) - (2 * x ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (3 * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) - (2 * y ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (3 * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) + (x ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) + (y ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) + (gamma * x ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (Pr * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) + (gamma * y ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (3 / 2)) / (Pr * Rgas * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas) ^ 2) - (3 * gamma * x ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (2 * Pr * Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) - (3 * gamma * y ^ 2 * (T_ref + T_s) * (gamma - 1) * (-((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / (Rgas * T_ref)) ^ (1 / 2)) / (2 * Pr * Rgas * T_ref * (T_s - ((gamma - 1) * (x ^ 2 / 2 + y ^ 2 / 2 - 1)) / Rgas)) + 2)
	--print(ret)
end

return {
	ndim = 2,
	uniform_mesh = {
		nelem = { 50, 50 },
		bounding_box = { min = { 0.0, 0.0 }, max = { 1.0, 1.0 } },
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
		basis = "legendre",

		-- the quadrature type (optional: default = gauss)
		quadrature = "gauss",

		-- the basis function order
		order = 1,
	},

	conservation_law = {
		name = "navier-stokes",
		source = source
	},

	initial_condition = manufactured_sol,

	solver = {
		type = "rk3-tvd",
		dt = 1e-6,
		ntime = 100,
		ivis = 1
	},

	output = {
		writer = "dat"
	}
}
