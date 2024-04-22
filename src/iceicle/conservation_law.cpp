/**
 * @brief miniapp to solve conservation laws
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/mesh/mesh_lua_interface.hpp"
#include "iceicle/fespace/fespace_lua_interface.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/anomaly_log.hpp"
#ifdef ICEICLE_USE_PETSC 
#include "iceicle/petsc_newton.hpp"
#elifdef ICEICLE_USE_MPI
#include "mpi.h"
#endif
#include <fenv.h>
#include <sol/sol.hpp>
int main(int argc, char* argv[]){

    // Initialize
#ifdef ICEICLE_USE_PETSC
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#elifdef ICEICLE_USE_MPI
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
#endif

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

    // ===============================
    // = Command line argument setup =
    // ===============================
    cli_parser cli_args{argc, argv};
    
    cli_args.add_options(
        cli_flag{"help", "print the help text and quit."},
        cli_flag{"enable_fp_except", "enable floating point exceptions (ignoring FE_INEXACT)"},
        cli_option{"scriptfile", "The file name for the lua script to run", parse_type<std::string_view>{}}
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
        AbstractMesh<T, IDX, ndim> mesh =
            lua_uniform_mesh<T, IDX, ndim>(script_config);
        perturb_mesh(script_config, mesh);

        // ===================================
        // = create the finite element space =
        // ===================================

        auto fespace = lua_fespace(&mesh, lua_state);
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
