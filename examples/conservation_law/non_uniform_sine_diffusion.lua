local generate_mesh_table = function(xmin, xmax, nelem, element_ratio)
	local dx = xmax / (nelem // 2 + element_ratio * (nelem // 2));

	-- Make the nodes
	local coord = { { xmin } };
	for inode = 1, nelem do
		if inode % 2 == 0 then
			coord[inode + 1] = { coord[inode][1] + dx }
		else
			coord[inode + 1] = { coord[inode][1] + element_ratio * dx }
		end
	end

	-- make element connectivity
	local elements = {}
	for ielem = 1, nelem do
		elements[ielem] = {
			domain_type = "hypercube",
			order = 1,
			nodes = { ielem - 1, ielem },
		}
	end

	-- make boundary faces
	local boundary_faces = {
		{
			bc_type = "dirichlet",
			bc_flag = 0,
			nodes = { 0 }
		},
		{
			bc_type = "dirichlet",
			bc_flag = 1,
			nodes = { nelem }
		},
	}

	return { coord = coord, elements = elements, boundary_faces = boundary_faces }
end

local fourier_nr = 0.0001

local nelem_arg = 8
local mu = 1.0
local tfinal = 1.0

return {
	ndim = 1,

	user_mesh = generate_mesh_table(0.0, 2 * math.pi, nelem_arg, 3.0),

	-- define the finite element space
	fespace = {
		-- the basis function type (optional: default = lagrange)
		basis = "legendre",

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
			0.0,
			0.0,
		},
	},

	-- solver
	solver = {
		type = "rk3-tvd",
		dt = fourier_nr * (2 * math.pi / nelem_arg) ^ 2,
		tfinal = tfinal,
		ivis = 100000,
	},

	-- output
	output = {
		writer = "dat",
	},

	-- post-processing
	post = {
		exact_solution = function(x)
			return math.sin(x) * math.exp(-mu * tfinal)
		end,

		tasks = {
			"l2_error",
			"linf_error",
			"l1_error",
			"ic_residual",
		},
	},
}
