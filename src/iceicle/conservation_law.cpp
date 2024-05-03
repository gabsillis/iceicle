/**
 * @brief miniapp to solve conservation laws
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/disc/conservation_law.hpp"
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/mesh/mesh_lua_interface.hpp"
#include "iceicle/fespace/fespace_lua_interface.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/solvers_lua_interface.hpp"
#include "iceicle/string_utils.hpp"
#include "iceicle/initialization.hpp"
#include "iceicle/pvd_writer.hpp"
#include "iceicle/disc/bc_lua_interface.hpp"
#ifdef ICEICLE_USE_PETSC 
#include "iceicle/petsc_newton.hpp"
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
template<class T, int ndim, int neq>
class test_dirichlet{
    public:
    using value_type = T;
    static constexpr int nv_comp = neq;
    static constexpr int dimensionality = ndim;

    std::vector< std::function<void(const T*, T*)> > dirichlet_callbacks;

    auto operator()() -> void {
        std::array<T, ndim> x{};
        std::cout << "x: [ ";
        for(int idim = 0; idim < ndim; ++idim){
            x[idim] = (0.125) * (idim + 1);
            std::cout << x[idim] << " ";
        }
        std::cout << "]" << std::endl;

        for(auto f : dirichlet_callbacks){
            std::cout << "f: [ ";
            std::array<T, neq> fval{};
            f(x.data(), fval.data());
            for(int ieq = 0; ieq < neq; ++ieq){
                std::cout << fval[ieq] << " ";
            }
            std::cout << " ]" << std::endl;
        }
        std::cout << std::endl;
    }
};

template<class T, class IDX, int ndim, class pflux, class cflux, class dflux>
void initialize_and_solve(
    sol::table config_tbl,
    FESpace<T, IDX, ndim> &fespace,
    ConservationLawDDG<T, ndim, pflux, cflux, dflux> &conservation_law
) {
    // ==================================
    // = Initialize the solution vector =
    // ==================================
    constexpr int neq = std::remove_reference_t<decltype(conservation_law)>::nv_comp;
    fe_layout_right u_layout{fespace.dg_map, to_size<neq>{}};
    std::vector<T> u_data(u_layout.size());
    fespan u{u_data.data(), u_layout};
    initialize_solution_lua(config_tbl, fespace, u);

    // ===============================
    // = Output the Initial Solution =
    // ===============================
    if constexpr(ndim == 2 || ndim == 3){
        io::PVDWriter<T, IDX, ndim> pvd_writer{};
        pvd_writer.register_fespace(fespace);
        pvd_writer.register_fields(u, "u");
        pvd_writer.collection_name = "initial_condition";
        pvd_writer.write_vtu(0, 0.0);
    }

    // ==================================
    // = Set up the Boundary Conditions =
    // ==================================
    sol::optional<sol::table> bc_desc_opt = config_tbl["boundary_conditions"];
    if(bc_desc_opt){
        sol::table bc_tbl = bc_desc_opt.value();
        add_dirichlet_callbacks(conservation_law, bc_tbl);
    }

    // =========
    // = Solve =
    // =========
    solvers::lua_solve(config_tbl, fespace, conservation_law, u);

}

int main(int argc, char* argv[]){

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
        cli_flag{"enable_fp_except", "enable floating point exceptions (ignoring FE_INEXACT)"},
        cli_option{"scriptfile", "The file name for the lua script to run", parse_type<std::string_view>{}},
        cli_flag{"debug1", "internal debug flag"}
    );
    if(cli_args["help"]){
        cli_args.print_options(std::cout);
        return 0;
    }
    if(cli_args["enable_fp_except"]){
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
    if(cli_args["scriptfile"]){
        script_config_opt = lua_state.script_file(cli_args["scriptfile"].as<std::string>());
    } else {
        script_config_opt = lua_state.script_file(default_input_deck_filename);
    }

    sol::table script_config;
    if(script_config_opt){
        script_config = script_config_opt.value();
    } else {
        std::cerr << "Error loading script configuration" << std::endl;
        return 1;
    }

    // Enter compile time ndim region
    auto ndim_func = [&]<int ndim>{

        // ==============
        // = Setup Mesh =
        // ==============
        sol::optional<sol::table> uniform_mesh_tbl = script_config["uniform_mesh"];
        AbstractMesh<T, IDX, ndim> mesh =
            lua_uniform_mesh<T, IDX, ndim>(uniform_mesh_tbl.value());
        perturb_mesh(script_config, mesh);

        if(cli_args["debug1"]){
            // linear advection a = [0.2, 0];
            // 2 element mesh on [0, 1]^2
            mesh.nodes[4][0] = 0.7;
        }

        // ===================================
        // = create the finite element space =
        // ===================================
        sol::table fespace_tbl = script_config["fespace"];
        auto fespace = lua_fespace(&mesh, fespace_tbl);

        // ============================
        // = Setup the Discretization =
        // ============================
        if(script_config["conservation_law"].valid()){
            sol::table cons_law_tbl = script_config["conservation_law"];
            if(eq_icase(cons_law_tbl["name"].get<std::string>(), "burgers")) {

                // get the coefficients for burgers equation
                BurgersCoefficients<T, ndim> burgers_coeffs{};
                sol::optional<T> mu_input = cons_law_tbl["mu"];
                if(mu_input) burgers_coeffs.mu = mu_input.value();

                sol::optional<sol::table> a_adv_input = cons_law_tbl["a_adv"];
                if(a_adv_input){
                    for(int idim = 0; idim < ndim; ++idim)
                        burgers_coeffs.a[idim] = a_adv_input.value()[idim + 1];
                }
                sol::optional<sol::table> b_adv_input = cons_law_tbl["b_adv"];
                if(b_adv_input){
                    for(int idim = 0; idim < ndim; ++idim)
                        burgers_coeffs.b[idim] = b_adv_input.value()[idim + 1];
                }

                // create the discretization
                BurgersFlux physical_flux{burgers_coeffs};
                BurgersUpwind convective_flux{burgers_coeffs};
                BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
                ConservationLawDDG disc{std::move(physical_flux), std::move(convective_flux), std::move(diffusive_flux)};
                initialize_and_solve(script_config, fespace, disc);

            } else if(eq_icase(cons_law_tbl["name"].get<std::string>(), "spacetime-burgers")) {
                static constexpr int ndim_space = ndim - 1;
                // get the coefficients for burgers equation
                BurgersCoefficients<T, ndim_space> burgers_coeffs{};
                sol::optional<T> mu_input = cons_law_tbl["mu"];
                if(mu_input) burgers_coeffs.mu = mu_input.value();

                sol::optional<sol::table> a_adv_input = cons_law_tbl["a_adv"];
                if(a_adv_input){
                    for(int idim = 0; idim < ndim_space; ++idim)
                        burgers_coeffs.a[idim] = a_adv_input.value()[idim + 1];
                }
                sol::optional<sol::table> b_adv_input = cons_law_tbl["b_adv"];
                if(b_adv_input){
                    for(int idim = 0; idim < ndim_space; ++idim)
                        burgers_coeffs.b[idim] = b_adv_input.value()[idim + 1];
                }

                // create the discretization
                SpacetimeBurgersFlux physical_flux{burgers_coeffs};
                SpacetimeBurgersUpwind convective_flux{burgers_coeffs};
                SpacetimeBurgersDiffusion diffusive_flux{burgers_coeffs};
                ConservationLawDDG disc{std::move(physical_flux), std::move(convective_flux), std::move(diffusive_flux)};
                initialize_and_solve(script_config, fespace, disc);

            } else {
                AnomalyLog::log_anomaly(Anomaly{ "No such conservation_law implemented",
                        text_not_found_tag{cons_law_tbl["name"].get<std::string>()}});
            }

        } else {
            std::cout << "No conservation_law table specified: exiting...";
        }


        return 0;
    };
    // exit compile time ndim region

    // template specialization: ndim
    int ndim_arg = script_config["ndim"];
    // invoke the ndim function
    NUMTOOL::TMP::invoke_at_index(
        NUMTOOL::TMP::make_range_sequence<int, 1, 4>{},
        ndim_arg,
        ndim_func);

#ifdef ICEICLE_USE_PETSC
    //cleanup
    PetscFinalize();
#elifdef ICEICLE_USE_MPI
    // cleanup
    MPI_Finalize();
#endif
    AnomalyLog::handle_anomalies();
    return 0;
}
