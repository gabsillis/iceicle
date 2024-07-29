/// @brief lua interface to dispatch solvers 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/string_utils.hpp"
#include "iceicle/writer.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/explicit_utils.hpp>
#include <iceicle/anomaly_log.hpp>
#include <iceicle/tmp_utils.hpp>
#include <iceicle/explicit_euler.hpp>
#include <iceicle/ssp_rk3.hpp>
#include <iceicle/tvd_rk3.hpp>
#include <iceicle/dat_writer.hpp>
#include <iceicle/pvd_writer.hpp>
#include <iceicle/writer.hpp>
#include <sol/sol.hpp>

#ifdef ICEICLE_USE_PETSC
#include <iceicle/petsc_newton.hpp>
#include <iceicle/petsc_gn_linesearch.hpp>
#endif

namespace iceicle::solvers {

    /// @brief Create a writer for output files 
    /// given the user configuration
    /// @param config_tbl the user configuration lua table 
    /// @param fespace the finite element space 
    /// @param disc the discretization
    /// @param u the solution to write
    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_get_writer(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        DiscType disc,
        fespan<T, LayoutPolicy> u 
    ) -> io::Writer 
    {
        using namespace iceicle::util;
        io::Writer writer;
        sol::optional<sol::table> output_tbl_opt = config_tbl["output"];
        if(output_tbl_opt){
            sol::table output_tbl = output_tbl_opt.value();
            sol::optional<std::string> writer_name = output_tbl["writer"];

            // .dat file writer
            // NOTE: short circuiting &&
            if(writer_name && eq_icase(writer_name.value(), "dat")){
                if constexpr (ndim == 1){
                    io::DatWriter<T, IDX, ndim> dat_writer{fespace};
                    dat_writer.register_fields(u, "u");
                    writer = io::Writer{dat_writer};
                } else {
                    AnomalyLog::log_anomaly(Anomaly{"dat writer not defined for greater than 1D", general_anomaly_tag{}});
                }
            }

            // .vtu writer 
            if(writer_name && eq_icase(writer_name.value(), "vtu")){
                io::PVDWriter<T, IDX, ndim> pvd_writer{};
                pvd_writer.register_fespace(fespace);
                pvd_writer.register_fields(u, "u");
                writer = pvd_writer;
            }
        }
        return writer;
    }

    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_solve(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        DiscType disc, 
        fespan<T, LayoutPolicy> u
    ) -> void {
        using namespace iceicle::util;
        using namespace iceicle::tmp;

        sol::table solver_params = config_tbl["solver"];
        std::string solver_type = solver_params["type"];
        // check for explicit solvers 
        if(eq_icase_any(solver_type, "explicit_euler", "rk3-ssp", "rk3-tvd")){

            // ========================================
            // = determine the timestepping criterion =
            // ========================================
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

            // =======================================
            // = determine the termination criterion =
            // =======================================
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

            // =====================
            // = Dispatch Function =
            // =      for all      =
            // = Explicit Solvers  =
            // =====================
            auto setup_and_solve = [&]<class ExplicitSolverType>(ExplicitSolverType& solver){
    
                // ==============================
                // = During solve visualization =
                // ==============================
                solver.ivis = 1;
                sol::optional<IDX> ivis_input = solver_params["ivis"];
                if(ivis_input) solver.ivis = ivis_input.value();

                sol::optional<sol::table> output_tbl_opt = config_tbl["output"];
                io::Writer writer{lua_get_writer(config_tbl, fespace, disc, u)};

                solver.vis_callback = [&](ExplicitSolverType& solver) mutable {
                    T sum = 0.0;
                    for(int i = 0; i < solver.res_data.size(); ++i){
                        sum += SQUARED(solver.res_data[i]);
                    }
                    std::cout << std::setprecision(8);
                    std::cout << "itime: " << std::setw(6) << solver.itime 
                        << " | t: " << std::setw(14) << solver.time
                        << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
                        << std::endl;

                    if(writer) writer.write(solver.itime, solver.time);
                };
                // =====================
                // = Perform the solve =
                // =====================
                solver.solve(fespace, disc, u);
            };

            // by solver type and the timestep and termination variants 
            // dispatch to the proper solver execution
            if(timestep.has_value() && stop_condition.has_value()){

                std::tuple{timestep.value(), stop_condition.value()} >> select_fcn{
                    [&](const auto &ts, const auto &sc){

                        T t_final;
                        if(eq_icase(solver_type, "explicit_euler")){
                            ExplicitEuler solver{fespace, disc, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        } else if(eq_icase(solver_type, "rk3-ssp")){
                            RK3SSP solver{fespace, disc, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        } else if(eq_icase(solver_type, "rk3-tvd")){
                            RK3TVD solver{fespace, disc, ts, sc};
                            setup_and_solve(solver);
                            t_final = solver.time;
                        }
                    }
                };
            }
        } else if(eq_icase_any(solver_type, "newton", "newton-ls", "gauss-newton")) {
            // Newton Solvers
#ifdef ICEICLE_USE_PETSC
            
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

            // Output Setup
            io::Writer writer{lua_get_writer(config_tbl, fespace, disc, u)};

            linesearch >> select_fcn{
                [&](const auto& ls){

                    sol::optional<sol::table> mdg_params_opt = solver_params["mdg"];
                    if(mdg_params_opt){
                        sol::table mdg_params = mdg_params_opt.value();
                        IDX ncycles = (sol::optional<IDX>{mdg_params["ncycles"]}) ? mdg_params["ncycles"].get<IDX>() : 1;

                        sol::optional<sol::function> ic_selection_function{mdg_params["ic_selection_threshold"]};
                        sol::optional<T> ic_selection_value{mdg_params["ic_selection_threshold"]};


                        nodeset_dof_map<IDX> nodeset{}; // select no nodes initially
                        IDX total_nl_vis = 0; // cumulative index for times visualization function gets called
                        for(IDX icycle = 0; icycle < ncycles; ++icycle){
                            
                            // select the nodes
                            T ic_selection_threshold = 0.1;
                            if(ic_selection_value){
                                ic_selection_threshold = ic_selection_value.value();
                            }
                            // selection function takes the cycle number to give dynamic threshold
                            if(ic_selection_function){ 
                                ic_selection_threshold = ic_selection_function.value()(icycle);
                            }

                            nodeset = select_nodeset(fespace, disc, u, ic_selection_threshold, tmp::to_size<ndim>{});
                            std::cout << "=================" << std::endl;
                            std::cout << " MDG CYCLE : " << icycle << std::endl;
                            std::cout << "=================" << std::endl << std::endl;
                            

                            auto setup_and_solve = [&]<class SolverT>(SolverT& solver){
                                solver.idiag = (sol::optional<IDX>{solver_params["idiag"]}) ? solver_params["idiag"].get<IDX>() : -1;
                                solver.ivis = (sol::optional<IDX>{solver_params["ivis"]}) ? solver_params["ivis"].get<IDX>() : 1;
                                solver.verbosity = (sol::optional<IDX>{solver_params["verbosity"]}) ? solver_params["verbosity"].get<IDX>() : 0;
                                solver.vis_callback = [&](decltype(solver) &solver, IDX k, Vec res_data, Vec du_data){
                                    T res_norm;
                                    PetscCallAbort(solver.comm, VecNorm(res_data, NORM_2, &res_norm));
                                    std::cout << std::setprecision(8);
                                    std::cout << "itime: " << std::setw(6) << k
                                        << " | residual l2: " << std::setw(14) << res_norm
                                        << std::endl;
                                    // offset by initial solution iteration
                                    writer.write(total_nl_vis + k + 1, (T) total_nl_vis +  k + 1);

                                    sol::optional<sol::table> output_tbl_opt = config_tbl["output"];
                                    if(output_tbl_opt){
                                        sol::table output_tbl = output_tbl_opt.value();
                                        sol::optional<std::string> writer_name = output_tbl["writer"];
                                        if(writer_name && eq_icase(writer_name.value(), "vtu")){

                                            // setup output for mdg data
                                            io::PVDWriter<T, IDX, ndim> mdg_writer;
                                            mdg_writer.register_fespace(fespace);
                                            petsc::VecSpan res_view{res_data};
                                            petsc::VecSpan dx_view{du_data};

                                            // get the start indices for the petsc matrix on this processor
                                            PetscInt proc_range_beg, proc_range_end, mdg_range_beg;
                                            PetscCallAbort(MPI_COMM_WORLD, VecGetOwnershipRange(res_data, &proc_range_beg, &proc_range_end));
                                            mdg_range_beg = proc_range_beg + u.size();

                                            node_selection_layout<IDX, ndim> mdg_layout{nodeset};
                                            dofspan mdg_res{res_view.data() + mdg_range_beg, mdg_layout};
                                            dofspan mdg_dx{dx_view.data() + mdg_range_beg, mdg_layout};
                                            mdg_writer.register_fields(mdg_res, "mdg residual");
                                            mdg_writer.register_fields(mdg_dx, "-dx");
                                            mdg_writer.collection_name = "mdg_data";
                                            mdg_writer.write_vtu(total_nl_vis + k + 1, (T) total_nl_vis +  k + 1);
                                        }
                                    }
                                };
                                total_nl_vis += solver.solve(u);
                            };

                            // set up solver and solve
                            if(eq_icase_any(solver_type, "newton", "newton-ls")){
                                PetscNewton solver{fespace, disc, conv_criteria, ls, nodeset};
                                setup_and_solve(solver);
                            } else if(eq_icase_any(solver_type, "gauss-newton")) {

                                sol::optional<bool> form_subproblem_opt = solver_params["form_subproblem_mat"];
                                bool form_subproblem = false;
                                if(form_subproblem_opt) form_subproblem = form_subproblem_opt.value();
                                GaussNewtonPetsc solver{fespace, disc, conv_criteria, ls, nodeset, form_subproblem};

                                sol::optional<T> regularization = solver_params["regularization"];
                                if(regularization){
                                    T val = regularization.value();
                                    solver.regularization_callback = [val](decltype(solver)& solver, IDX k){
                                        return val;
                                    };
                                }

                                sol::optional<sol::function> regularization_as_func = solver_params["regularization"];
                                if(regularization_as_func){
                                    sol::function reg_f = regularization_as_func.value();
                                    solver.regularization_callback = [reg_f](decltype(solver)& solver, IDX k) -> T {
                                        T res_norm;
                                        PetscCallAbort(solver.comm, VecNorm(solver.res_data, NORM_2, &res_norm));
                                        T reg = reg_f(k, res_norm);
                                        return reg;
                                    };
                                }
                                setup_and_solve(solver);
                            }
                        }

                    } else {
                        auto setup_and_solve = [&]<class SolverT>(SolverT& solver){
                            solver.idiag = (sol::optional<IDX>{solver_params["idiag"]}) ? solver_params["idiag"].get<IDX>() : -1;
                            solver.ivis = (sol::optional<IDX>{solver_params["ivis"]}) ? solver_params["ivis"].get<IDX>() : 1;
                            solver.verbosity = (sol::optional<IDX>{solver_params["verbosity"]}) ? solver_params["verbosity"].get<IDX>() : 0;
                            solver.vis_callback = [&](decltype(solver) &solver, IDX k, Vec res_data, Vec du_data){
                                T res_norm;
                                PetscCallAbort(solver.comm, VecNorm(res_data, NORM_2, &res_norm));
                                std::cout << std::setprecision(8);
                                std::cout << "itime: " << std::setw(6) << k
                                    << " | residual l2: " << std::setw(14) << res_norm
                                    << std::endl << std::endl;
                                // offset by initial solution iteration
                                writer.write(k + 1, (T) k + 1);
                            };
                            solver.solve(u);
                        };

                        // set up solver and solve
                        if(eq_icase_any(solver_type, "newton", "netwon-ls")){
                            PetscNewton solver{fespace, disc, conv_criteria, ls};
                            setup_and_solve(solver);
                        } else if(eq_icase_any(solver_type, "gauss-newton")) {

                            sol::optional<bool> form_subproblem_opt = solver_params["form_subproblem_mat"];
                            bool form_subproblem = false;
                            if(form_subproblem_opt) form_subproblem = form_subproblem_opt.value();
                            GaussNewtonPetsc solver{fespace, disc, conv_criteria, ls, form_subproblem};

                            sol::optional<T> regularization = solver_params["regularization"];
                            if(regularization){
                                T val = regularization.value();
                                solver.regularization_callback = [val](decltype(solver)& solver, IDX k){
                                    return val;
                                };
                            }

                            sol::optional<sol::function> regularization_as_func = solver_params["regularization"];
                            if(regularization_as_func){
                                sol::function reg_f = regularization_as_func.value();
                                solver.regularization_callback = [reg_f](decltype(solver)& solver, IDX k) -> T {
                                    T res_norm;
                                    PetscCallAbort(solver.comm, VecNorm(solver.res_data, NORM_2, &res_norm));
                                    T reg = reg_f(k, res_norm);
                                    return reg;
                                };
                            }

                            setup_and_solve(solver);
                        }
                    }
                }
            };
#else 
            AnomalyLog::log_anomaly(Anomaly{"No non-petsc newton solvers currently implemented.", general_anomaly_tag{}});
#endif

        }
    }
}
