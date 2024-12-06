#include <fenv.h>
#include <cmath>
#include <iceicle/program_args.hpp>
#include <sol/sol.hpp>
#include <iceicle/mesh/mesh_lua_interface.hpp>
#include <iceicle/mesh/mesh_partition.hpp>
#include <iceicle/solvers_lua_interface.hpp>
#include <iceicle/fespace/fespace_lua_interface.hpp>
#include <iceicle/disc/navier_stokes.hpp>
#include <iceicle/disc/conservation_law.hpp>
#include <iceicle/tmp_utils.hpp>
#include <iceicle/initialization.hpp>
using namespace iceicle;
using namespace iceicle::util;
using namespace iceicle::util::program_args;
using namespace iceicle::tmp;
using namespace std;
using T = double;
using IDX = int;

double mu_arg = 0.1;

void exact(const double *xvec, double *uvec)
{
  double x = xvec[0];
  double y = xvec[1];

  uvec[0] = sin(2 * (x + y)) + 4; 
  uvec[1] = sin(2 * (x + y)) + 4; 
  uvec[2] = sin(2 * (x + y)) + 4; 
  uvec[3] = pow(sin(2 * (x + y)) + 4, 2); 

  uvec[0] = 1.0 + x;
  uvec[1] = 1.0 + x;
  uvec[2] = 1.0 + x;
  uvec[3] = pow(1.0 + x, 2);
}

// void exact(const double *xvec, double *uvec)
// {
//   double x = xvec[0];
//   double y = xvec[1];
//   uvec[0] = x + y + 1;
//   uvec[1] = x + y + 1;
//   uvec[2] = x + y + 1;
//   uvec[3] = pow(x + y + 1, 2);
// }

// void exact(const double *xvec, double *uvec)
// {
//   double x = xvec[0];
//   double y = xvec[1];
//   uvec[0] = 1.0;
//   uvec[1] = 1.0;
//   uvec[2] = 1.0;
//   uvec[3] = 1.0;
// }

void initial_condition(const double *xvec, double *uvec)
{
  double x = xvec[0];
  double y = xvec[1];

  uvec[0] = sin(1.5 * (x + y)) + 3;
  uvec[1] = 0.2 * sin(1.5 * (x + y)) + 3;
  uvec[2] = 0.8 * sin(1.5 * (x + y)) + 3;
  uvec[3] = pow(sin(1.5 * (x + y)) + 3, 2);

  uvec[0] = 0.5;
  uvec[1] = 0.5;
  uvec[2] = 0.5;
  uvec[3] = 0.25;
}

void source(const double *xvec, double *s){
  double gamma = 1.4;
  double Pr = 0.72;
  double mu = mu_arg;
  double x = xvec[0];
  double y = xvec[1];

  s[0] = 4 * cos(2 * (x + y));
  s[1] = cos(2 * (x + y)) * (14 * gamma - 10)
    + sin(4 * (x + y)) * (2 * gamma - 2);
  s[2] = cos(2 * (x + y)) * (14 * gamma - 10)
    + sin(4 * (x + y)) * (2 * gamma - 2);
  s[3] = cos(2 * (x + y)) * (28 * gamma + 4)
    + 4 * gamma * sin(4 * (x + y)) 
    + 8 * mu * gamma / Pr * sin(2 * (x + y));

  s[0] = 1.0;
  s[1] = (gamma - 1) * (2 * x + 1) + 1;
  s[2] = 1.0;
  s[3] = 2 * x + (2 * x + 1) * (gamma - 1) + 2;

  // negate for rhs
  for(int ieq = 0; ieq < 4; ++ieq)
    s[ieq] = -s[ieq];
}

// void source(const double *xvec, double *s){
//   double gamma = 1.4;
//   double Pr = 0.72;
//   double mu = mu_arg;
//   double x = xvec[0];
//   double y = xvec[1];
// 
//   double u = (x + y + 1);
//   double v = (x + y + 1);
//   double rho_ui_uj = pow(x + y + 1, 2);
//   double diff_rho_ui_uj = 2 * (x + y + 1);
//   double diff_rho = 2;
//   double drhoedx = 2 * (x + y + 1);
//   double drhoedy = 2 * (x + y + 1);
//   double drhoudx = 0;
//   double dudy = 0;
//   double dvdx = 0;
//   double dvdy = 0;
//   double dpdx = (gamma - 1) * (drhoedx + diff_rho);
//   double dpdy = dpdx;
//   double diff_E = 2;
// 
//   s[0] = 2 * diff_rho;
//   s[1] = 2 * diff_rho_ui_uj + dpdx;
//   s[2] = 2 * diff_rho_ui_uj + dpdy;
//   s[3] = u * drhoedx + v * drhoedy + u * dpdx + v * dpdy 
//     - mu * gamma / Pr * (diff_E + diff_E);
// 
//   // negate for rhs
//   for(int ieq = 0; ieq < 4; ++ieq)
//     s[ieq] = -s[ieq];
// }

//void source(const double *xvec, double *s){
//  s[0] = 0;
//  s[1] = 0;
//  s[2] = 0;
//  s[3] = 0;
//}

int main(int argc, char *argv[]) {
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // ===============================
  // = Command line argument setup =
  // ===============================
  cli_parser cli_args{argc, argv};

  cli_args.add_options(
      cli_flag{"help", "print the help text and quit."},
      cli_flag{"enable_fp_except",
               "enable floating point exceptions (ignoring FE_INEXACT)"},
      cli_option{"scriptfile", "The file name for the lua script to run",
                 parse_type<std::string_view>{}},
      cli_flag{"debug1", "internal debug flag"},
      cli_flag{"debug2", "internal debug flag"});
  if (cli_args["help"]) {
    cli_args.print_options(std::cout);
    return 0;
  }
  if (cli_args["enable_fp_except"]) {
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
  }

  // =============
  // = Lua Setup =
  // =============

  // The default file to read as input deck
  const char *default_input_deck_filename = "iceicle.lua";

  // Parse the input deck
  sol::state lua_state;
  lua_state.open_libraries(sol::lib::base);
  lua_state.open_libraries(sol::lib::package);
  lua_state.open_libraries(sol::lib::math);

  sol::optional<sol::table> script_config_opt;
  if (cli_args["scriptfile"]) {
    script_config_opt =
        lua_state.script_file(cli_args["scriptfile"].as<std::string>());
  } else {
    script_config_opt = lua_state.script_file(default_input_deck_filename);
  }

  sol::table script_config;
  if (script_config_opt) {
    script_config = script_config_opt.value();
  } else {
    std::cerr << "Error loading script configuration" << std::endl;
    return 1;
  }

  static constexpr int ndim = 2;

  // ==============
  // = Setup Mesh =
  // ==============
  auto mesh_opt = construct_mesh_from_config<T, IDX, ndim>(script_config);
  if (!mesh_opt) {
    std::cerr << "Mesh construction failed..." << std::endl;
    AnomalyLog::handle_anomalies();
    return 1; // exit if we have no valid mesh
  }
  AbstractMesh<T, IDX, ndim> mesh = mesh_opt.value();
  std::vector<IDX> invalid_faces;
  if (!validate_normals(mesh, invalid_faces)) {
    std::cout << "invalid normals on the following faces: ";
    for (IDX ifac : invalid_faces)
      std::cout << ifac << ", ";
    std::cout << "\n";
    return 1;
  }
  perturb_mesh(script_config, mesh);
  manual_mesh_management(script_config, mesh);

  AbstractMesh<T, IDX, ndim> pmesh{partition_mesh(mesh)};

  // ===================================
  // = create the finite element space =
  // ===================================
  sol::table fespace_tbl = script_config["fespace"];
  auto fespace = lua_fespace(&pmesh, fespace_tbl);
  fespace.print_info(std::cout);

  // ========================
  // = Set up Navier Stokes =
  // ========================
  navier_stokes::ReferenceParameters<T> ref{};
  navier_stokes::CaloricallyPerfectEoS<T, ndim> eos{};
  navier_stokes::constant_viscosity<T> mu{0.1};
  navier_stokes::Physics physics{ref, eos, mu};

  navier_stokes::Flux flux{physics, std::true_type{}};
  navier_stokes::VanLeer numflux{physics};
  navier_stokes::DiffusionFlux diffusion_flux{physics, std::true_type{}};

  ConservationLawDDG conservation_law{std::move(flux), std::move(numflux), std::move(diffusion_flux)};
  // conservation_law.interior_penalty = true;
  conservation_law.user_source = std::function{source};
  conservation_law.field_names = std::vector<std::string>{"rho", "rhou"};
  conservation_law.residual_names = std::vector<std::string>{"density_conservation", "momentum_u_conservation"};
  if constexpr (ndim >= 2) {
    conservation_law.field_names.push_back("rhov");
    conservation_law.residual_names.push_back("momentum_v_conservation");
  }
  if constexpr (ndim >= 3) {
    conservation_law.field_names.push_back("rhow");
    conservation_law.residual_names.push_back("momentum_w_conservation");
  }
  conservation_law.field_names.push_back("rhoe");
  conservation_law.residual_names.push_back("energy_conservation");
  constexpr int neq =
      std::remove_reference_t<decltype(conservation_law)>::nv_comp;

  // ==============================
  // = Initialize Solution Vector =
  // ==============================
  fe_layout_right u_layout{fespace.dg_map, to_size<neq>{}};
  std::vector<T> u_data(u_layout.size());
  fespan u{u_data.data(), u_layout};
  projection_initialization(fespace, std::function{initial_condition}, tmp::compile_int<neq>{}, u);
  // projection_initialization(fespace, std::function{exact}, tmp::compile_int<neq>{}, u);

  io::PVDWriter<T, IDX, ndim> pvd_writer{};
  pvd_writer.register_fespace(fespace);
  pvd_writer.register_fields(u, conservation_law.field_names);
  pvd_writer.collection_name = "initial_condition";
  pvd_writer.write_vtu(0, 0.0);

  // =======================
  // = Boundary Conditions =
  // =======================
  conservation_law.dirichlet_callbacks.push_back(std::function{exact});

  // =========
  // = Solve =
  // =========
  auto geo_map = solvers::lua_select_mdg_geometry(
      script_config, fespace, conservation_law, 0, u);
  solvers::lua_solve(script_config, fespace, geo_map, conservation_law, u);

  // ============================
  // = Post-Processing/Analysis =
  // ============================
  solvers::lua_error_analysis(script_config, fespace, conservation_law, u);
  return 0;
}
