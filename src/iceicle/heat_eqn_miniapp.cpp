/**
 * @brief miniapp to solve the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include "iceicle/disc/l2_error.hpp"
#include "iceicle/fespace/fespace_lua_interface.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/nonlinear_solver_utils.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/ssp_rk3.hpp"
#include "iceicle/tvd_rk3.hpp"
#include "iceicle/string_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include "iceicle/mdg_utils.hpp"
#include <iomanip>
#include <limits>
#ifdef ICEICLE_USE_PETSC 
#include "iceicle/petsc_newton.hpp"
#elifdef ICEICLE_USE_MPI
#include "mpi.h"
#endif

#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/explicit_euler.hpp>
#include <iceicle/solvers/element_linear_solve.hpp>
#include <iceicle/build_config.hpp>
#include <iceicle/pvd_writer.hpp>
#include <iceicle/mesh/mesh_lua_interface.hpp>
#include <fenv.h>
#include <sol/sol.hpp>

int main(int argc, char *argv[]){
#ifdef ICEICLE_USE_PETSC
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#elifdef ICEICLE_USE_MPI
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
#endif

    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace ICEICLE::UTIL;
    using namespace ICEICLE::TMP;

    // Get the floating point and index types from 
    // cmake configuration
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;
    using namespace ICEICLE::UTIL::PROGRAM_ARGS;

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
    if(cli_args["scriptfile"]){
        lua_state.script_file(cli_args["scriptfile"].as<std::string>());
    } else {
        lua_state.script_file(default_input_deck_filename);
    }

    // 2 dimensional simulation
    static constexpr int ndim = 2;

    // =========================
    // = create a uniform mesh =
    // =========================

    MESH::AbstractMesh<T, IDX, ndim> mesh = MESH::lua_uniform_mesh<T, IDX, ndim>(lua_state);

    // perturb nodes if applicable 
    sol::optional<std::string> perturb_fcn_name = lua_state["mesh_perturbation"];
    if(perturb_fcn_name){
        std::vector<bool> fixed_nodes = MESH::flag_boundary_nodes(mesh);

        std::function< void(std::span<T, ndim>, std::span<T, ndim>) > perturb_fcn;
        if(eq_icase(perturb_fcn_name.value(), "taylor-green")){
            perturb_fcn = MESH::PERTURBATION_FUNCTIONS::TaylorGreenVortex<T, ndim>{
                .v0 = 0.5,
                .xmin = {
                    lua_state["uniform_mesh"]["bounding_box"]["min"][1],
                    lua_state["uniform_mesh"]["bounding_box"]["min"][2],
                },

                .xmax = {
                    lua_state["uniform_mesh"]["bounding_box"]["max"][1],
                    lua_state["uniform_mesh"]["bounding_box"]["max"][2],
                },
                .L = 1
            };
        } else if(eq_icase(perturb_fcn_name.value(), "zig-zag")){
            perturb_fcn = MESH::PERTURBATION_FUNCTIONS::ZigZag<T, ndim>{};
        }

        MESH::perturb_nodes(mesh, perturb_fcn, fixed_nodes);
    }

    // ===================================
    // = create the finite element space =
    // ===================================

    auto fespace = FE::lua_fespace(&mesh, lua_state);

    // =============================
    // = set up the discretization =
    // =============================

    DISC::HeatEquation<T, IDX, ndim> heat_equation{}; 
    sol::optional<T> mu_input = lua_state["mu"];
    if(mu_input) heat_equation.mu = mu_input.value();

    sol::optional<sol::table> a_adv_input = lua_state["a_adv"];
    if(a_adv_input){
        for(int idim = 0; idim < ndim; ++idim)
            heat_equation.a[idim] = a_adv_input.value()[idim + 1];
    }

    sol::optional<sol::table> b_adv_input = lua_state["b_adv"];
    if(b_adv_input){
        for(int idim = 0; idim < ndim; ++idim)
            heat_equation.b[idim] = b_adv_input.value()[idim + 1];
    }

    // Get boundary condition values from input deck  TODO: get rid of 
    static constexpr int MAX_BC_ENTRIES = 100; // protect from infinite loops

    sol::optional<sol::table> dirichlet_spec_opt = lua_state["boundary_conditions"]["dirichlet"];
    if(dirichlet_spec_opt){
        sol::table dirichlet_spec = dirichlet_spec_opt.value();

        // Value arguments
        // To make indexing consistently 1-indexed 
        // push a 0 valued boundary condition at index 0
        // Now values start at 1 and callbacks start at -1 with no overlap on index 0
        heat_equation.dirichlet_values.push_back(0); 
        for(int ibc = 1; ibc < MAX_BC_ENTRIES; ++ibc){
            sol::optional<T> lua_value = dirichlet_spec["values"][ibc];
            if(lua_value) heat_equation.dirichlet_values.push_back(lua_value.value());
            else break;
        }

        // Callbacks
        // same index 0 trick
        // actually use the table iterator here
        heat_equation.dirichlet_callbacks.push_back([](const double*, double *out){ out[0] = 0.0; });
        sol::optional<sol::table> dirichlet_callbacks_opt = dirichlet_spec["callbacks"];
        if(dirichlet_callbacks_opt){
            sol::table callback_tbl = dirichlet_callbacks_opt.value();
            sol::function testf = callback_tbl[2];
            for(const auto& key_value : callback_tbl){
                int index = key_value.first.as<int>();
                auto dirichlet_func = key_value.second.as<std::function<double(double, double)>>();

                // indexes could be in non-sequential order so manually work with vector capacity
                if(index >= heat_equation.dirichlet_callbacks.size())
                    heat_equation.dirichlet_callbacks.resize(2 * heat_equation.dirichlet_callbacks.size());

                // add the callback function at the given index
    //            heat_equation.dirichlet_callbacks[index] =
    //                [dirichlet_func](const double *xarr, double *out){
    //                    out[0] = dirichlet_func(xarr[0], xarr[1]);
    //                };
                heat_equation.dirichlet_callbacks[index] =
                    [&lua_state, index](const double *xarr, double *out){
                        sol::optional<sol::table> ftbl = lua_state["boundary_conditions"]["dirichlet"]["callbacks"];
                        sol::function f = ftbl.value()[index];
                        out[0] = f(xarr[0], xarr[1]);
                    };
            }
        }
    }


    sol::optional<sol::table> neumann_spec_opt = lua_state["boundary_conditions"]["neumann"];
    if(neumann_spec_opt){
        sol::table neumann_spec = neumann_spec_opt.value();

        // Value arguments
        // To make indexing consistently 1-indexed 
        // push a 0 valued boundary condition at index 0
        // Now values start at 1 and callbacks start at -1 with no overlap on index 0
        heat_equation.neumann_values.push_back(0); 
        for(int ibc = 1; ibc < MAX_BC_ENTRIES; ++ibc){
            sol::optional<T> lua_value = neumann_spec["values"][ibc];
            if(lua_value) heat_equation.neumann_values.push_back(lua_value.value());
            else break;
        }

    }

    // ============================
    // = set up a solution vector =
    // ============================
    constexpr int neq = decltype(heat_equation)::nv_comp;
    FE::fe_layout_right u_layout{fespace.dg_map, to_size<neq>{}};
    std::vector<T> u_data(u_layout.size());
    FE::fespan u{u_data.data(), u_layout};

    // ===========================
    // = initialize the solution =
    // ===========================

    std::cout << "Initializing Solution..." << std::endl;

    std::function<void(const double*, double *)> ic;

    // first check if the initial condiiton is identified by string
    sol::optional<std::string> ic_string = lua_state["initial_condition"];
    if(ic_string){
        if(eq_icase(ic_string.value(), "zero")) {
            ic = [](const double *, double *out){ out[0] = 0.0; };
        }
    } else {
        // else we take ic as a lua function with
        // 2 inputs (ndim) and one output (neq)
        // and wrap in a lambda
        ic = [&lua_state](const double *xarr, double *out){
            out[0] = lua_state["initial_condition"](xarr[0], xarr[1]);
        };
    }


    auto analytic_sol = [](const double *xarr, double *out) -> void{
        double x = xarr[0];
        double y = xarr[1];

        out[0] = 0.1 * std::sinh(M_PI * x) / std::sinh(M_PI) * std::sin(M_PI * y) + 1.0;
    };

    auto dirichlet_func = [](const double *xarr, double *out) ->void{
        double x = xarr[0];
        double y = xarr[1];

        out[0] = 1 + 0.1 * std::sin(M_PI * y);
    };
    
    // Manufactured solution BC 
    heat_equation.dirichlet_callbacks.resize(2);
    heat_equation.dirichlet_callbacks[1] = dirichlet_func;


    DISC::Projection<T, IDX, ndim, neq> projection{ic};
    // TODO: extract into LinearFormSolver
    std::vector<T> u_local_data(fespace.dg_map.max_el_size_reqirement(neq));
    std::vector<T> res_local_data(fespace.dg_map.max_el_size_reqirement(neq));
    std::for_each(fespace.elements.begin(), fespace.elements.end(), 
        [&](const ELEMENT::FiniteElement<T, IDX, ndim> &el){
            // form the element local views
            // TODO: maybe instead of scatter from local view 
            // we can directly create the view on the subset of u 
            // for CG this might require a different compact Layout 
            FE::dofspan u_local{u_local_data.data(), u.create_element_layout(el.elidx)};
            u_local = 0;

            FE::dofspan res_local{res_local_data.data(), u.create_element_layout(el.elidx)};
            res_local = 0;

            // project
            projection.domainIntegral(el, fespace.meshptr->nodes, res_local);

            // solve 
            SOLVERS::ElementLinearSolver<T, IDX, ndim, neq> solver{el, fespace.meshptr->nodes};
            solver.solve(u_local, res_local);

            // scatter to global array 
            // (note we use 0 as multiplier for current values in global array)
            FE::scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);
        }
    );

    bool use_explicit = false;

    sol::optional<sol::table> solver_params_opt = lua_state["solver"];
    if(solver_params_opt){
        sol::table solver_params = solver_params_opt.value();
        // ====================================
        // = Solve with Explicit Timestepping =
        // ====================================
        using namespace ICEICLE::SOLVERS;
        using namespace ICEICLE::UTIL;
        using namespace ICEICLE::TMP;

        auto setup_and_solve = [&]<class ExplicitSolverType>(ExplicitSolverType &solver){
            solver.ivis = 1;
            sol::optional<IDX> ivis_input = solver_params["ivis"];
            if(ivis_input) solver.ivis = ivis_input.value();
            ICEICLE::IO::PVDWriter<T, IDX, ndim> pvd_writer;
            pvd_writer.register_fespace(fespace);
            pvd_writer.register_fields(u, "u");
            solver.vis_callback = [&](decltype(solver) &solver){
                T sum = 0.0;
                for(int i = 0; i < solver.res_data.size(); ++i){
                    sum += SQUARED(solver.res_data[i]);
                }
                std::cout << std::setprecision(8);
                std::cout << "itime: " << std::setw(6) << solver.itime 
                    << " | t: " << std::setw(14) << solver.time
                    << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
                    << std::endl;

                pvd_writer.write_vtu(solver.itime, solver.time);

            };
            solver.solve(fespace, heat_equation, u);
        };

        std::string solver_type = solver_params["type"];
        if(
            eq_icase(solver_type, "explicit_euler") || 
            eq_icase(solver_type, "rk3-ssp")        ||
            eq_icase(solver_type, "rk3-tvd")        
        ){
            // determine timestep criterion
            std::optional<TimestepVariant<T, IDX>> timestep;

            if(sol::optional<T>{solver_params["dt"]}){
                if(timestep.has_value()) AnomalyLog::log_anomaly(Anomaly{
                        "Cannot set fixed timestep criterion: other timestep criterion already set",
                        general_anomaly_tag{}});
                T fixed_dt = solver_params["dt"];
                timestep = FixedTimestep<T, IDX>{fixed_dt};
            }
            if(sol::optional<T>{solver_params["cfl"]}){
                if(timestep.has_value()) AnomalyLog::log_anomaly(Anomaly{
                        "Cannot set cfl timestep criterion: other timestep criterion already set",
                        general_anomaly_tag{}});
                T cfl = solver_params["cfl"];
                timestep = CFLTimestep<T, IDX>{cfl};
            } 
            if(!timestep.has_value()){
                AnomalyLog::log_anomaly(Anomaly{"No timestep criterion set", general_anomaly_tag{}});
            }

            // determine the termination criterion 
            std::optional<TerminationVariant<T, IDX>> stop_condition;
            if(sol::optional<T>{solver_params["tfinal"]}){
                if(stop_condition.has_value()) AnomalyLog::log_anomaly(Anomaly{
                        "Cannot set tfinal termination criterion: other termination criterion already set",
                        general_anomaly_tag{}});
                T tfinal = solver_params["tfinal"];
                stop_condition = TfinalTermination<T, IDX>{tfinal};
            }
            if(sol::optional<IDX>{solver_params["ntime"]}){
                if(stop_condition.has_value()) AnomalyLog::log_anomaly(Anomaly{
                        "Cannot set ntime termination criterion: other termination criterion already set",
                        general_anomaly_tag{}});
                IDX ntime = solver_params["ntime"];
                stop_condition = TimestepTermination<T, IDX>{ntime};
            }
            if(!stop_condition.has_value()){
                AnomalyLog::log_anomaly(Anomaly{"No termination criterion set", general_anomaly_tag{}});
            }

            if(timestep.has_value() && stop_condition.has_value()){

                std::tuple{timestep.value(), stop_condition.value()} >> select_fcn{
                    [&](const auto &ts, const auto &sc){

                        T t_final;
                        if(eq_icase(solver_type, "explicit_euler")){
                            ExplicitEuler solver{fespace, heat_equation, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        } else if(eq_icase(solver_type, "rk3-ssp")){
                            RK3SSP solver{fespace, heat_equation, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        } else if(eq_icase(solver_type, "rk3-tvd")){
                            RK3TVD solver{fespace, heat_equation, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        }

                        // ========================
                        // = Compute the L2 Error =
                        // =   (if applicable)    =
                        // ========================
                        sol::optional<sol::function> exact_sol = lua_state["exact_sol"];
                        if(exact_sol){
                            std::function<void(T*, T*)> exactfunc = 
                                [&lua_state, t_final](T *x, T *out) -> void {
                                    sol::function fexact = lua_state["exact_sol"];
                                    out[0] = fexact(x[0], x[1], t_final);
                                };
                            T l2_error = DISC::l2_error(exactfunc, fespace, u);
                            std::cout << "L2 error: " << std::setprecision(9) << l2_error << std::endl;
                        }
                    }
                };
            }
        } else if(
            eq_icase(solver_type, "newton") ||
            eq_icase(solver_type, "newton-ls")
        ) {
#ifdef ICEICLE_USE_PETSC
            // ==============================
            // = Solve with Newton's Method =
            // ==============================
            using namespace ICEICLE::SOLVERS;

            // default is machine zero convergence 
            // with maximum of 5 nonlinear iterations
            ConvergenceCriteria<T, IDX> conv_criteria{
                .tau_abs = (sol::optional<T>{solver_params["tau_abs"]}) ? solver_params["tau_abs"].get<T>() : std::numeric_limits<T>::epsilon(),
                .tau_rel = (sol::optional<T>{solver_params["tau_rel"]}) ? solver_params["tau_rel"].get<T>() : 0.0,
                .kmax = (sol::optional<IDX>{solver_params["kmax"]}) ? solver_params["kmax"].get<IDX>() : 5 
            };

            // select the linesearch type
            LinesearchVariant<T, IDX> linesearch;
            sol::optional<sol::table> ls_arg_opt = solver_params["linesearch"];
            if(ls_arg_opt){
                sol::table ls_arg = ls_arg_opt.value();
                if(eq_icase(ls_arg["type"].get<std::string>(), "wolfe") 
                        || eq_icase(ls_arg["type"].get<std::string>(), "cubic"))
                {
                    IDX kmax = (sol::optional<IDX>{ls_arg["kmax"]}) ? ls_arg["kmax"].get<IDX>() : 5; 
                    T alpha_initial = (sol::optional<T>{ls_arg["alpha_initial"]}) ? ls_arg["alpha_initial"].get<T>() : 1; 
                    T alpha_max = (sol::optional<T>{ls_arg["alpha_max"]}) ? ls_arg["alpha_max"].get<T>() : 10.0; 
                    T c1 = (sol::optional<T>{ls_arg["c1"]}) ? ls_arg["c1"].get<T>() : 1e-4; 
                    T c2 = (sol::optional<T>{ls_arg["c2"]}) ? ls_arg["c2"].get<T>() : 0.9; 
                    linesearch = wolfe_linesearch{kmax, alpha_initial, alpha_max, c1, c2};
                } else {
                    linesearch = no_linesearch<T, IDX>{};
                }
            } else {
                linesearch = no_linesearch<T, IDX>{};
            };

            // dispatch with the set lineset strategy
            linesearch >> select_fcn{
                [&](const auto& ls){
                    PetscNewton solver{fespace, heat_equation, conv_criteria, ls};
                    solver.idiag = (sol::optional<IDX>{solver_params["idiag"]}) ? solver_params["idiag"].get<IDX>() : -1;
                    solver.ivis = (sol::optional<IDX>{solver_params["ivis"]}) ? solver_params["ivis"].get<IDX>() : 1;
                    ICEICLE::IO::PVDWriter<T, IDX, ndim> pvd_writer;
                    pvd_writer.register_fespace(fespace);
                    pvd_writer.register_fields(u, "u");
                    pvd_writer.write_vtu(0, 0.0); // print the initial solution
                    solver.vis_callback = [&](decltype(solver) &solver, IDX k, Vec res_data, Vec du_data){
                        T res_norm;
                        PetscCallAbort(solver.comm, VecNorm(res_data, NORM_2, &res_norm));
                        std::cout << std::setprecision(8);
                        std::cout << "itime: " << std::setw(6) << k
                            << " | residual l2: " << std::setw(14) << res_norm
                            << std::endl;
                        // offset by initial solution iteration
                        pvd_writer.write_vtu(k + 1, (T) k + 1);
                    };
                    solver.solve(u);

                    // ========================
                    // = Compute the L2 Error =
                    // =   (if applicable)    =
                    // ========================
                    sol::optional<sol::function> exact_sol = lua_state["exact_sol"];
                    if(exact_sol){
                        std::function<void(T*, T*)> exactfunc = 
                            [&lua_state](T *x, T *out) -> void {
                                sol::function fexact = lua_state["exact_sol"];
                                out[0] = fexact(x[0], x[1]);
                            };
                        T l2_error = DISC::l2_error(exactfunc, fespace, u);
                        std::cout << "L2 error: " << std::setprecision(9) << l2_error << std::endl;
                    }

                }
            };

//            // ===============================
//            // = move the nodes around a bit =
//            // ===============================
//            FE::nodeset_dof_map<IDX> nodeset{FE::select_nodeset(fespace, heat_equation, u, 0.003, ICEICLE::TMP::to_size<ndim>())};
//            FE::node_selection_layout<IDX, ndim> new_coord_layout{nodeset};
//            std::vector<T> new_coord_storage(new_coord_layout.size());
//            FE::dofspan dx_coord{new_coord_storage.data(), new_coord_layout};
//
//            
//            std::vector<T> node_radii = FE::node_freedom_radii(fespace);
//            dx_coord = 0;
//            for(IDX idof = 0; idof < dx_coord.ndof(); ++idof){
//                IDX inode = nodeset.selected_nodes[idof];
//
//                // push to the right a little 
//                dx_coord[idof, 0] = 0.5 * node_radii[inode];
//            }
//
//            // apply new nodes 
//            FE::scatter_node_selection_span(1.0, dx_coord, 1.0, fespace.meshptr->nodes);
//            // fixup interior nodes 
//            FE::regularize_interior_nodes(fespace);

#else 
            AnomalyLog::log_anomaly(Anomaly{"Newton solver requires PETSC!"), general_anomaly_tag{}});
#endif
        }
    } else {

#ifdef ICEICLE_USE_PETSC
    //cleanup
    PetscFinalize();
#elifdef ICEICLE_USE_MPI
    // cleanup
    MPI_Finalize();
#endif
    }
    AnomalyLog::handle_anomalies();
    return 0;
}


