---
--- Example: MDG Inviscid Burgers
--- Author: Gianni Absillis (gabsill@ncsu.edu)
---
--- Inviscid Burgers Equation in spacetime with a piecewise linear initial condition.
--- This creates a triangular shaped characteristics pattern.
---
--- NOTE: this is a numerically challenging problem for MDG and is a work in progress

-- Initial Condition extruded into time
-- local pde_ic = function(x, t)
-- 	if x < 0.25 then
-- 		return 4
-- 	elseif x < 0.75 then
-- 		return 8 - 12 * x
-- 	else
-- 		return -2
-- 	end
-- end
--
-- -- Exact solution to the problem
-- local exact = function(x, t)
-- 	local x_plus = math.min(0.25 + 4 * t, 7.0 / 12.0)
-- 	local x_minus = math.max(0.75 - 2 * t, 7.0 / 12.0)
-- 	if x < x_plus then
-- 		return 4
-- 	elseif x < x_minus then
-- 		return (12 * x - 6) / (12 * t - 1)
-- 	else
-- 		return -2
-- 	end
-- end

local pde_ic = function(x, t)
	if x < 0.25 then
		return 2
	elseif x < 0.75 then
		return 4 - 8 * x
	else
		return -2
	end
end

-- Exact solution to the problem
local exact = function(x, t)
	local x_plus = math.min(0.25 + 2 * t, 0.5)
	local x_minus = math.max(0.75 - 2 * t, 0.5)
	if x < x_plus then
		return 2
	elseif x < x_minus then
		return (8 * x - 4) / (8 * t - 1)
	else
		return -2
	end
end

-- configuration
return {
	-- specify the number of dimensions (REQUIRED)
	ndim = 2,

	-- create a uniform mesh
	--	uniform_mesh = {
	--		nelem = { 4, 2 },
	--
	--		bounding_box = {
	--			min = { 0.0, 0.0 },
	--			max = { 1.0, 0.5 },
	--		},
	--
	--		boundary_conditions = {
	--			types = {
	--				"dirichlet", -- left side
	--				"dirichlet", -- bottom side
	--				"dirichlet", -- right side
	--				"extrapolation", -- top side
	--			},
	--
	--			flags = { 0, 0, 0, 1 }
	--		}
	--	},

	--	user_mesh = {
	--		coord = {
	--			{ 0.0,  0.0 }, -- 0
	--			{ 0.25, 0.0 }, -- 1
	--			{ 0.75, 0.0 }, -- 2
	--			{ 1.0,  0.0 }, -- 3
	--			{ 0.0,  0.25 }, -- 4
	--			{ 0.25, 0.25 }, -- 5
	--			{ 0.75, 0.25 }, -- 6
	--			{ 1.0,  0.25 }, -- 7
	--			{ 0.0,  0.5 }, -- 8
	--			{ 0.25, 0.5 }, -- 9
	--			{ 0.75, 0.5 }, -- 10
	--			{ 1.0,  0.5 }, -- 11
	--		},
	--		elements = {
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 0, 4, 1, 5 },
	--			},
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 1, 5, 2, 6 },
	--			},
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 2, 6, 3, 7 },
	--			},
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 4, 8, 5, 9 },
	--			},
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 5, 9, 6, 10 },
	--			},
	--			{
	--				domain_type = "hypercube",
	--				order = 1,
	--				nodes = { 6, 10, 7, 11 },
	--			},
	--		},
	--
	--		boundary_faces = {
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 0, 1 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 1, 2 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 2, 3 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 3, 7 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 7, 11 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 8, 9 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 9, 10 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 10, 11 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 0, 4 },
	--			},
	--			{
	--				bc_type = "dirichlet",
	--				bc_flag = 1,
	--				nodes = { 4, 8 },
	--			},
	--		},
	--	},

	user_mesh = {
		coord = {
			{ 0.0,  0.0 }, -- 0
			{ 0.25, 0.0 }, -- 1
			{ 0.5,  0.0 }, -- 2
			{ 0.75, 0.0 }, -- 3
			{ 1.0,  0.0 }, -- 4
			{ 0.0,  0.125 }, -- 5
			{ 0.4,  0.12 }, -- 6
			{ 0.5,  0.125 }, -- 7
			{ 0.6,  0.12 }, -- 8
			{ 1.0,  0.125 }, -- 9
			{ 0.0,  0.5 }, -- 10
			{ 0.25, 0.5 }, -- 11
			{ 0.5,  0.5 }, -- 12
			{ 0.75, 0.5 }, -- 13
			{ 1.0,  0.5 }, -- 14
		},

		elements = {
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 0, 5, 1, 6 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 1, 6, 2, 7 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 2, 7, 3, 8 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 3, 8, 4, 9 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 5, 10, 6, 11 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 6, 11, 7, 12 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 7, 12, 8, 13 },
			},
			{
				domain_type = "hypercube",
				order = 1,
				nodes = { 8, 13, 9, 14 },
			},
		},

		boundary_faces = {
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 0, 1 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 1, 2 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 2, 3 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 3, 4 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 4, 9 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 9, 14 },
			},
			{
				bc_type = "extrapolation",
				bc_flag = 1,
				nodes = { 10, 11 },
			},
			{
				bc_type = "extrapolation",
				bc_flag = 1,
				nodes = { 11, 12 },
			},
			{
				bc_type = "extrapolation",
				bc_flag = 1,
				nodes = { 12, 13 },
			},
			{
				bc_type = "extrapolation",
				bc_flag = 1,
				nodes = { 13, 14 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 0, 5 },
			},
			{
				bc_type = "dirichlet",
				bc_flag = 1,
				nodes = { 5, 10 },
			},
		},

	},

	initial_condition = pde_ic,
	-- initial_condition = exact,

	conservation_law = {
		name = "spacetime-burgers",
		mu = 0.0,
		a_adv = { 0.0 },
		b_adv = { 1.0 },
	},

	-- describe the finite element space
	fespace = {
		basis = "lagrange",
		quadrature = "gauss",
		order = 2
	},

	-- boundary conditions
	boundary_conditions = {
		dirichlet = {
			pde_ic,
			exact,
		},
	},

	-- MDG
	mdg = {
		ncycles = 1,
		ic_selection_threshold = function(icycle)
			return 0
		end,
	},

	solver = {
		type = "gauss-newton",
		form_subproblem_mat = true,
		linesearch = {
			type = "corrigan"
		},
		lambda_b = 0.001,
		lambda_lag = 0.0,
		lambda_u = 1e-12,
		ivis = 1,
		idiag = 1,
		tau_rel = 1e-8,
		kmax = 1000,
	},

	output = {
		writer = "vtu"
	},
}
