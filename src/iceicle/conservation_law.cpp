/**
 * @brief miniapp to solve conservation laws
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/disc/conservation_law.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/dat_writer.hpp"
#include "iceicle/disc/bc_lua_interface.hpp"
#include "iceicle/disc/burgers.hpp"
#include "iceicle/disc/navier_stokes.hpp"
#include "iceicle/fespace/fespace_lua_interface.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/initialization.hpp"
#include "iceicle/mesh/mesh_lua_interface.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/pvd_writer.hpp"
#include "iceicle/solvers_lua_interface.hpp"
#include "iceicle/string_utils.hpp"
#include "iceicle/mesh/mesh_partition.hpp"
#ifdef ICEICLE_USE_PETSC
#elifdef ICEICLE_USE_MPI
#include "mpi.h"
#endif
#include <fenv.h>
#include <sol/sol.hpp>

// using declarations
using namespace NUMTOOL::TENSOR::FIXED_SIZE;
using namespace iceicle;
using namespace iceicle::util;
using namespace iceicle::util::program_args;
using namespace iceicle::tmp;

// Get the floating point and index types from
// cmake configuration
using T = build_config::T;
using IDX = build_config::IDX;

/// @brief class for testing dirichlet implementation
/// TODO: move to a unit test
template <class T, int ndim, int neq> class test_dirichlet {
public:
  using value_type = T;
  static constexpr int nv_comp = neq;
  static constexpr int dimensionality = ndim;

  std::vector<std::function<void(const T *, T *)>> dirichlet_callbacks;

  auto operator()() -> void {
    std::array<T, ndim> x{};
    std::cout << "x: [ ";
    for (int idim = 0; idim < ndim; ++idim) {
      x[idim] = (0.125) * (idim + 1);
      std::cout << x[idim] << " ";
    }
    std::cout << "]" << std::endl;

    for (auto f : dirichlet_callbacks) {
      std::cout << "f: [ ";
      std::array<T, neq> fval{};
      f(x.data(), fval.data());
      for (int ieq = 0; ieq < neq; ++ieq) {
        std::cout << fval[ieq] << " ";
      }
      std::cout << " ]" << std::endl;
    }
    std::cout << std::endl;
  }
};

template <class T, class IDX, int ndim, class pflux, class cflux, class dflux>
void initialize_and_solve(
    sol::table config_tbl, FESpace<T, IDX, ndim> &fespace,
    ConservationLawDDG<T, ndim, pflux, cflux, dflux> &conservation_law) {
  // ==================================
  // = Initialize the solution vector =
  // ==================================
  constexpr int neq =
      std::remove_reference_t<decltype(conservation_law)>::nv_comp;
  fe_layout_right u_layout{fespace.dg_map, to_size<neq>{}};
  std::vector<T> u_data(u_layout.size());
  fespan u{u_data.data(), u_layout};
  initialize_solution_lua(config_tbl, fespace, u);

  // ===============================
  // = Output the Initial Solution =
  // ===============================
  if constexpr(ndim == 1){
    io::DatWriter<T, IDX, ndim> dat_writer{fespace};
    dat_writer.register_fields(u, conservation_law.field_names);
    dat_writer.collection_name = "initial_condition";
    dat_writer.write_dat(0, 0.0);
  }
  if constexpr (ndim == 2 || ndim == 3) {
    io::PVDWriter<T, IDX, ndim> pvd_writer{};
    pvd_writer.register_fespace(fespace);
    pvd_writer.register_fields(u, conservation_law.field_names);
    pvd_writer.collection_name = "initial_condition";
    pvd_writer.write_vtu(0, 0.0);
  }

  // ==================================
  // = Set up the Boundary Conditions =
  // ==================================
  sol::optional<sol::table> bc_desc_opt = config_tbl["boundary_conditions"];
  if (bc_desc_opt) {
    sol::table bc_tbl = bc_desc_opt.value();
    add_dirichlet_callbacks(conservation_law, bc_tbl);
    add_neumann_callbacks(conservation_law, bc_tbl);
  }

  // ========================
  // = Add User Source Term =
  // ========================
  solvers::add_source_term_callback(conservation_law, config_tbl);

  // =========
  // = Solve =
  // =========
  sol::optional<sol::table> mdg_tbl = config_tbl["mdg"];
  IDX ncycles = (mdg_tbl) ? mdg_tbl.value().get_or("ncycles", 1) : 1;
  for (IDX icycle = 0; icycle < ncycles; ++icycle) {

    mpi::execute_on_rank(0, [&] {
      std::cout << "==============" << std::endl;
      std::cout << "Cycle: " << icycle << std::endl;
      std::cout << "==============" << std::endl;
    });
    auto geo_map = solvers::lua_select_mdg_geometry(
        config_tbl, fespace, conservation_law, icycle, u);
    solvers::lua_solve(config_tbl, fespace, geo_map, conservation_law, u);
  }

  // ============================
  // = Post-Processing/Analysis =
  // ============================
  solvers::lua_error_analysis(config_tbl, fespace, conservation_law, u);
}

template <int ndim>
[[gnu::noinline]]
void setup(sol::table script_config, cli_parser cli_args) {
  // ==============
  // = Setup Mesh =
  // ==============
  auto mesh_opt = construct_mesh_from_config<T, IDX, ndim>(script_config);
  if (!mesh_opt){
    std::cerr << "Mesh construction failed..." << std::endl;
    AnomalyLog::handle_anomalies();
    return; // exit if we have no valid mesh
  }
  AbstractMesh<T, IDX, ndim> mesh = mesh_opt.value();
  std::vector<IDX> invalid_faces;
  if (!validate_normals(mesh, invalid_faces)) {
    std::cout << "invalid normals on the following faces: ";
    for (IDX ifac : invalid_faces)
      std::cout << ifac << ", ";
    std::cout << "\n";
    return;
  }
  perturb_mesh(script_config, mesh);
  manual_mesh_management(script_config, mesh);

  AbstractMesh<T, IDX, ndim> pmesh{partition_mesh(mesh)};

  if(AnomalyLog::size() > 0){
    std::cerr << "Errors from setting up mesh: ";
    AnomalyLog::handle_anomalies(std::cerr);
    return;
  }

  if (cli_args["debug1"]) {
    // linear advection a = [0.2, 0];
    // 2 element mesh on [0, 1]^2
    pmesh.coord[7][0] = 0.7;
    pmesh.coord[4][0] = 0.55;
  }
  if (cli_args["debug2"]) {
    pmesh.coord[4][0] = 0.69;
  }

  // ===================================
  // = create the finite element space =
  // ===================================
  sol::table fespace_tbl = script_config["fespace"];
  auto fespace = lua_fespace(&pmesh, fespace_tbl);

  // ============================
  // = Setup the Discretization =
  // ============================
  if (script_config["conservation_law"].valid()) {
    sol::table cons_law_tbl = script_config["conservation_law"];

    if (eq_icase(cons_law_tbl["name"].get<std::string>(), "burgers")) {

      // get the coefficients for burgers equation
      BurgersCoefficients<T, ndim> burgers_coeffs{};
      sol::optional<T> mu_input = cons_law_tbl["mu"];
      if (mu_input)
        burgers_coeffs.mu = mu_input.value();

      // WARNING: for some reason in release mode
      // if we create these tables from cons_law_tbl
      // the second one won't read properly
      sol::optional<sol::table> b_adv_input =
          script_config["conservation_law"]["b_adv"];
      sol::optional<sol::table> a_adv_input =
          script_config["conservation_law"]["a_adv"];
      if (a_adv_input.has_value()) {
        for (int idim = 0; idim < ndim; ++idim)
          burgers_coeffs.a[idim] =
              script_config["conservation_law"]["a_adv"][idim + 1];
      }
      if (b_adv_input.has_value()) {
        for (int idim = 0; idim < ndim; ++idim)
          burgers_coeffs.b[idim] =
              script_config["conservation_law"]["b_adv"][idim + 1];
      }

      std::cout << burgers_coeffs.mu 
                << " " << burgers_coeffs.a[0] 
                << " " << burgers_coeffs.b[0] 
            << std::endl;
      // create the discretization
      BurgersFlux physical_flux{burgers_coeffs};
      BurgersUpwind convective_flux{burgers_coeffs};
      BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
      ConservationLawDDG disc{std::move(physical_flux),
                              std::move(convective_flux),
                              std::move(diffusive_flux)};
      disc.field_names = std::vector<std::string>{"u"};
      disc.residual_names = std::vector<std::string>{"residual"};

      // discretization options
      disc.sigma_ic = cons_law_tbl.get_or("sigma_ic", disc.sigma_ic);

      initialize_and_solve(script_config, fespace, disc);

    } else if (eq_icase(cons_law_tbl["name"].get<std::string>(),
                        "spacetime-burgers")) {
      static constexpr int ndim_space = ndim - 1;
      // get the coefficients for burgers equation
      BurgersCoefficients<T, ndim_space> burgers_coeffs{};
      sol::optional<T> mu_input = cons_law_tbl["mu"];
      if (mu_input)
        burgers_coeffs.mu = mu_input.value();

      sol::optional<sol::table> b_adv_input =
          script_config["conservation_law"]["b_adv"];
      sol::optional<sol::table> a_adv_input =
          script_config["conservation_law"]["a_adv"];
      if (a_adv_input.has_value()) {
        for (int idim = 0; idim < ndim_space; ++idim)
          burgers_coeffs.a[idim] =
              script_config["conservation_law"]["a_adv"][idim + 1];
      }
      if (b_adv_input.has_value()) {
        for (int idim = 0; idim < ndim_space; ++idim)
          burgers_coeffs.b[idim] =
              script_config["conservation_law"]["b_adv"][idim + 1];
      }
      std::cout << burgers_coeffs.b[0] << std::endl;

      // create the discretization
      SpacetimeBurgersFlux physical_flux{burgers_coeffs};
      SpacetimeBurgersUpwind convective_flux{burgers_coeffs};
      SpacetimeBurgersDiffusion diffusive_flux{burgers_coeffs};
      ConservationLawDDG disc{std::move(physical_flux),
                              std::move(convective_flux),
                              std::move(diffusive_flux)};
      disc.field_names = std::vector<std::string>{"u"};
      disc.residual_names = std::vector<std::string>{"residual"};
      initialize_and_solve(script_config, fespace, disc);

    } else if (eq_icase_any(cons_law_tbl["name"].get<std::string>(),
                            "navier-stokes", "euler")) {

      T gamma = cons_law_tbl.get_or("gamma", 1.4);

      navier_stokes::Physics<T, ndim> physics{gamma};

      navier_stokes::Flux<T, ndim> physical_flux{physics};
      navier_stokes::VanLeer<T, ndim> convective_flux{physics};
      navier_stokes::DiffusionFlux<T, ndim> diffusive_flux{physics};
      ConservationLawDDG disc{std::move(physical_flux),
                              std::move(convective_flux),
                              std::move(diffusive_flux)};
      disc.field_names = std::vector<std::string>{"rho", "rhou"};
      disc.residual_names = std::vector<std::string>{"density_conservation", "momentum_u_conservation"};
      if constexpr (ndim >= 2) {
        disc.field_names.push_back("rhov");
        disc.residual_names.push_back("momentum_v_conservation");
      }
      if constexpr (ndim >= 3) {
        disc.field_names.push_back("rhow");
        disc.residual_names.push_back("momentum_w_conservation");
      }
      disc.field_names.push_back("rhoe");
      disc.residual_names.push_back("energy_conservation");
      initialize_and_solve(script_config, fespace, disc);
    } else {
      AnomalyLog::log_anomaly(
          Anomaly{"No such conservation_law implemented",
                  text_not_found_tag{cons_law_tbl["name"].get<std::string>()}});
    }

  } else {
    std::cout << "No conservation_law table specified: exiting...";
  }
}

int main(int argc, char *argv[]) {

  // Initialize
#ifdef ICEICLE_USE_PETSC
  PetscInitialize(&argc, &argv, nullptr, nullptr);
#elifdef ICEICLE_USE_MPI
  /* Initialize MPI */
  MPI_Init(&argc, &argv);
#endif

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

  //    // template specialization: ndim
  int ndim_arg = script_config["ndim"];
  std::cout << "ndim: " << ndim_arg << std::endl;
  switch (ndim_arg) {
  case 1:
    setup<1>(script_config, cli_args);
    break;
  case 2:
    setup<2>(script_config, cli_args);
    break;
  }
  //    // invoke the ndim function
  //    WHY DOES THIS BREAK STUFF >:3
  //    NUMTOOL::TMP::invoke_at_index(
  //        NUMTOOL::TMP::make_range_sequence<int, 1, 3>{},
  //        ndim_arg,
  //        ndim_func);

#ifdef ICEICLE_USE_PETSC
  // cleanup
  PetscFinalize();
#elifdef ICEICLE_USE_MPI
  // cleanup
  MPI_Finalize();
#endif
  AnomalyLog::handle_anomalies();
  return 0;
}
