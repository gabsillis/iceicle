local gamma = 1.4;
local Rgas = 0.287;      -- kJ/(kg*K)
local mu_inf = 1.716e-5; --kg/ms
local T_ref = 273.15;    --K
local T_s = 110.4;       --K
local Pr = 0.72;
local pi = math.pi;

local cos = math.cos;
local sin = math.sin;

local function manufactured_sol(x, y)
	local rho = 1 + sin(pi * x) * 0.1;
	local rhou = rho * sin(x) * cos(y);
	local rhov = -rho * cos(x) * sin(y);
	local rhoE = 1 + cos(pi * x) * 0.1;
	return { rho, rhou, rhov, rhoE }
end

local function source(x, y)
	return
		-((pi * cos(pi * x) * cos(y) * sin(x)) / 10),
		-(cos(x) * cos(y) ^ 2 * sin(x) * (sin(pi * x) / 10 + 1) -
			(gamma - 1) *
			((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20) +
			(pi * cos(pi * x) * cos(y) ^ 2 * sin(x) ^ 2) / 10 + cos(x) * sin(x) * sin(y) ^ 2 * (sin(pi * x) / 10 + 1) +
			(2 * mu_inf * cos(y) * sin(x) * (T_ref + T_s) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) +
			(3 * mu_inf * cos(x) * cos(y) * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * T_ref * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) -
			(2 * mu_inf * cos(x) * cos(y) * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^
			2),
		-(cos(x) ^ 2 * cos(y) * sin(y) * (sin(pi * x) / 10 + 1) -
			(2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) * (sin(pi * x) / 20 + 1 / 2) * (gamma - 1) +
			cos(y) * sin(x) ^ 2 * sin(y) * (sin(pi * x) / 10 + 1) -
			(2 * mu_inf * cos(x) * sin(y) * (T_ref + T_s) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) -
			(pi * cos(pi * x) * cos(x) * cos(y) * sin(x) * sin(y)) / 10 +
			(2 * mu_inf * cos(x) * cos(y) * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) * (sin(pi * x) / 20 + 1 / 2) * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Rgas * (sin(pi * x) / 10 + 1) * (T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^ 2) -
			(3 * mu_inf * cos(x) * cos(y) * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) * (sin(pi * x) / 20 + 1 / 2) * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(Rgas * T_ref * (sin(pi * x) / 10 + 1) * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))))),
		-(cos(x) * sin(y) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) *
			(sin(pi * x) / 20 + 1 / 2) *
			(gamma - 1) -
			cos(y) * sin(x) *
			((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20) + (pi * sin(pi * x)) / 10) +
			(2 * mu_inf * cos(x) ^ 2 * sin(y) ^ 2 * (T_ref + T_s) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) +
			(2 * mu_inf * cos(y) ^ 2 * sin(x) ^ 2 * (T_ref + T_s) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) -
			(4 * mu_inf * cos(x) ^ 2 * cos(y) ^ 2 * (T_ref + T_s) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) -
			(2 * mu_inf * cos(x) * cos(y) ^ 2 * sin(x) * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^
			2 +
			(3 * mu_inf * cos(x) * cos(y) ^ 2 * sin(x) * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * T_ref * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) +
			(gamma * mu_inf * (T_ref + T_s) * (sin(pi * x) / 20 + 1 / 2) * (2 * cos(x) ^ 2 * cos(y) ^ 2 - 2 * cos(x) ^ 2 * sin(y) ^ 2 - 2 * cos(y) ^ 2 * sin(x) ^ 2 + 2 * sin(x) ^ 2 * sin(y) ^ 2) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Pr * (sin(pi * x) / 10 + 1) * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1)))) -
			(Rgas * gamma * mu_inf * (T_ref + T_s) * ((pi ^ 2 * sin(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * (sin(x * pi) / 10 + 1) ^ 2) - ((gamma - 1) * ((pi ^ 2 * cos(pi * x)) / 10 + (sin(pi * x) / 20 + 1 / 2) * (2 * cos(x) ^ 2 * cos(y) ^ 2 - 2 * cos(x) ^ 2 * sin(y) ^ 2 - 2 * cos(y) ^ 2 * sin(x) ^ 2 + 2 * sin(x) ^ 2 * sin(y) ^ 2) + (pi * cos(pi * x) * (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2)) / 10 - (pi ^ 2 * sin(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * (sin(pi * x) / 10 + 1)) + (pi ^ 2 * cos(pi * x) ^ 2 * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (50 * Rgas * (sin(x * pi) / 10 + 1) ^ 3) + (pi * cos(pi * x) * (gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (5 * Rgas * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Pr * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) * (gamma - 1)) +
			(Rgas * gamma * mu_inf * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * (sin(x * pi) / 10 + 1) ^ 2)) ^ 2 * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Pr * (T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^ 2 * (gamma - 1)) -
			(3 * Rgas * gamma * mu_inf * (T_ref + T_s) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * ((pi * sin(pi * x)) / 10 + (2 * cos(x) * cos(y) ^ 2 * sin(x) - 2 * cos(x) * sin(x) * sin(y) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + (pi * cos(pi * x) * (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2)) / 20)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1)) + (pi * cos(pi * x) * (gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (10 * Rgas * T_ref * (sin(x * pi) / 10 + 1) ^ 2)) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(2 * Pr * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1))) * (gamma - 1)) +
			(gamma * mu_inf * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) ^ 2 * (sin(pi * x) / 20 + 1 / 2) ^ 2 * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Pr * Rgas * (sin(x * pi) / 10 + 1) ^ 2 * (T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^ 2) -
			(2 * mu_inf * cos(x) ^ 2 * cos(y) * sin(y) * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) * (sin(pi * x) / 20 + 1 / 2) * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (3 / 2)) /
			(Rgas * (sin(pi * x) / 10 + 1) * (T_s + ((gamma - 1) * (cos(x * pi) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(x * pi) / 20 + 1 / 2) + 1)) / (Rgas * (sin(x * pi) / 10 + 1))) ^ 2) -
			(3 * gamma * mu_inf * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) ^ 2 * (sin(pi * x) / 20 + 1 / 2) ^ 2 * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(2 * Pr * Rgas * T_ref * (sin(x * pi) / 10 + 1) ^ 2 * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1)))) +
			(3 * mu_inf * cos(x) ^ 2 * cos(y) * sin(y) * (T_ref + T_s) * (2 * cos(x) ^ 2 * cos(y) * sin(y) - 2 * cos(y) * sin(x) ^ 2 * sin(y)) * (sin(pi * x) / 20 + 1 / 2) * (gamma - 1) * (((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * T_ref * (sin(pi * x) / 10 + 1))) ^ (1 / 2)) /
			(Rgas * T_ref * (sin(pi * x) / 10 + 1) * (T_s + ((gamma - 1) * (cos(pi * x) / 10 - (cos(x) ^ 2 * sin(y) ^ 2 + cos(y) ^ 2 * sin(x) ^ 2) * (sin(pi * x) / 20 + 1 / 2) + 1)) / (Rgas * (sin(pi * x) / 10 + 1)))))
end

return {
	ndim = 2,
	uniform_mesh = {
		nelem = { 10, 10 },
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
		-- name = "navier-stokes",
		name = "euler",
		source = source
	},

	initial_condition = manufactured_sol,

	solver = {
		type = "rk3-tvd",
		cfl = 0.1,
		ntime = 100,
		ivis = 1,
		idiag = 1
	},

	output = {
		writer = "vtu"
	}
}
