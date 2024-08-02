/// @brief lua interface to dispatch solvers 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/string_utils.hpp"
#include "iceicle/writer.hpp"
#include <array>
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/explicit_utils.hpp>
#include <iceicle/anomaly_log.hpp>
#include <iceicle/tmp_utils.hpp>
#include <iceicle/explicit_euler.hpp>
#include <iceicle/ssp_rk3.hpp>
#include <iceicle/tvd_rk3.hpp>
#include <iceicle/dat_writer.hpp>
#include <iceicle/pvd_writer.hpp>
#include <iceicle/fe_function/restart.hpp>
#include <iceicle/writer.hpp>
#include <sol/sol.hpp>

#ifdef ICEICLE_USE_PETSC
#include <iceicle/corrigan_lm.hpp>
#include <iceicle/petsc_newton.hpp>
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


    template<class T, class IDX, int ndim, class disc_type, class LayoutPolicy>
    auto lua_select_mdg_geometry(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        disc_type disc, 
        IDX icycle,
        fespan<T, LayoutPolicy> u
    ) -> geo_dof_map<T, IDX, ndim>
    {
        using index_type = IDX;
        using trace_type = FESpace<T, IDX, ndim>::TraceType;

        // Select relevant traces
        sol::optional<sol::table> mdg_params_opt = config_tbl["mdg"];
        if(mdg_params_opt){
            sol::table mdg_params = mdg_params_opt.value();

            sol::optional<sol::function> ic_selection_function{mdg_params["ic_selection_threshold"]};
            sol::optional<T> ic_selection_value{mdg_params["ic_selection_threshold"]};
            // select the nodes
            T ic_selection_threshold = 0.1;
            if(ic_selection_value){
                ic_selection_threshold = ic_selection_value.value();
            }
            // selection function takes the cycle number to give dynamic threshold
            if(ic_selection_function){ 
                ic_selection_threshold = ic_selection_function.value()(icycle);
            }

            // we will be filling the selected traces, nodes, 
            // and selected nodes -> gnode index map respectively
            std::vector<index_type> selected_traces{};

            std::vector<T> res_storage{};
            // preallocate storage for compact views of u and res 
            const std::size_t max_local_size =
                fespace.dg_map.max_el_size_reqirement(disc_type::dnv_comp);
            const std::size_t ncomp = disc_type::dnv_comp;
            std::vector<T> uL_storage(max_local_size);
            std::vector<T> uR_storage(max_local_size);

            // loop over the traces and select traces and nodes based on IC residual
            for(const trace_type &trace : fespace.get_interior_traces()){
                // compact data views 
                dofspan uL{uL_storage.data(), u.create_element_layout(trace.elL.elidx)};
                dofspan uR{uR_storage.data(), u.create_element_layout(trace.elR.elidx)};

                // trace data view
                trace_layout_right<IDX, disc_type::nv_comp> ic_res_layout{trace};
                res_storage.resize(ic_res_layout.size());
                dofspan ic_res{res_storage, ic_res_layout};

                // extract the compact values from the global u view 
                extract_elspan(trace.elL.elidx, u, uL);
                extract_elspan(trace.elR.elidx, u, uR);

                // zero out and then get interface conservation
                ic_res = 0.0;
                disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, ic_res);

                std::cout << "Interface nr: " << trace.facidx; 
                std::cout << " | nodes:";
                for(index_type inode : trace.face->nodes_span()){
                    std::cout << " " << inode;
                }
                std::cout << " | ic residual: " << ic_res.vector_norm() << std::endl; 

                // if interface conservation residual is high enough,
                // add the trace and nodes of the trace
                if(ic_res.vector_norm() >= ic_selection_threshold){
                    selected_traces.push_back(trace.facidx);
                }
            }

            geo_dof_map geo_map{selected_traces, fespace};

            // ========================
            // = Geometry Constraints =
            // ========================
            sol::optional<sol::table> uniform_mesh_tbl_opt = config_tbl["uniform_mesh"];
            if(uniform_mesh_tbl_opt){
                sol::table mesh_table = uniform_mesh_tbl_opt.value();

                std::array<IDX, ndim> nelem;
                sol::table nelem_table = mesh_table["nelem"];
                for(int idim = 0; idim < ndim; ++idim ){
                    nelem[idim] = nelem_table[idim + 1]; // NOTE: Lua is 1-indexed
                }

                // bounding box
                std::array<T, ndim> xmin;
                std::array<T, ndim> xmax;
                sol::table bounding_box_table = mesh_table["bounding_box"];
                for(int idim = 0; idim < ndim; ++idim){
                    xmin[idim] = bounding_box_table["min"][idim + 1];
                    xmax[idim] = bounding_box_table["max"][idim + 1];
                }
                mesh_parameterizations::hyper_rectangle(nelem, xmin, xmax, geo_map);

                // === Dirichlet BC => nodes cannot move ===
                for(auto trace : fespace.get_boundary_traces()){
                    if(trace.face->bctype == BOUNDARY_CONDITIONS::DIRICHLET){
                        for(index_type inode : trace.face->nodes_span()){
                            auto node_data = fespace.meshptr->nodes[inode];
                            std::array<T, ndim> fixed_coordinates;
                            for(int idim = 0; idim < ndim; ++idim)
                                { fixed_coordinates[idim] = node_data[idim]; }
                            parametric_transformations::Fixed<T, ndim> parameterization{fixed_coordinates};
                            geo_map.register_parametric_node(inode, parameterization);
                        }
                    }
                }
                geo_map.finalize();
            }

            return geo_map;
        } else {
            // select no traces 
            geo_dof_map geo_map{std::array<IDX, 0>{}, fespace};
            return geo_map;
        }

    }


    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_solve(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        geo_dof_map<T, IDX, ndim>& geo_map,
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
        } else if(eq_icase_any(solver_type, "newton", "lm", "gauss-newton")) {
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
                } else if(eq_icase(ls_arg["type"].get<std::string>(), "corrigan") ){
                    IDX kmax = ls_arg.get_or("kmax", 5);
                    T alpha_initial = ls_arg.get_or("alpha_initial", 1.0);
                    T alpha_max = ls_arg.get_or("alpha_max", 1.0);
                    T alpha_min = ls_arg.get_or("alpha_min", 0.0);
                    linesearch = corrigan_linesearch{kmax, alpha_initial, alpha_max, alpha_min};
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

                    auto setup_and_solve = [&]<class SolverT>(SolverT& solver){

                        // set common options between solvers
                        sol::optional<T> idiag = solver_params["idiag"];
                        if(idiag) solver.idiag = idiag.value();
                        sol::optional<T> ivis = solver_params["ivis"];
                        if(ivis) solver.ivis = ivis.value();
                        sol::optional<T> verbosity = solver_params["verbosity"];
                        if(verbosity) solver.verbosity = verbosity.value();

                        // visualization callback
                        solver.vis_callback = [&](IDX k, Vec res_data, Vec du_data){
                                 T res_norm;
                                PetscCallAbort(PETSC_COMM_WORLD, VecNorm(res_data, NORM_2, &res_norm));
                                std::cout << std::setprecision(8);
                                std::cout << "itime: " << std::setw(6) << k
                                    << " | residual l2: " << std::setw(14) << res_norm
                                    << std::endl << std::endl;

                                // offset by initial solution iteration
                                writer.write(k + 1, (T) k + 1);

                                write_restart(fespace, u, k + 1);
                        };
                        solver.solve(u);
                    };

                    if(eq_icase_any(solver_type, "lm", "gauss-newton")){
                        bool form_subproblem = solver_params.get_or("form_subproblem_mat", true); 
                        CorriganLM solver{fespace, disc, conv_criteria, ls, geo_map, form_subproblem};

                        // set options for the solver 
                        sol::optional<T> lambda_u = solver_params["lambda_u"];
                        if(lambda_u) solver.lambda_u = lambda_u.value();
                        sol::optional<T> lambda_lag = solver_params["lambda_lag"];
                        if(lambda_lag) solver.lambda_lag = lambda_lag.value();
                        sol::optional<T> lambda_1 = solver_params["lambda_1"];
                        if(lambda_1) solver.lambda_1 = lambda_1.value();
                        sol::optional<T> lambda_b = solver_params["lambda_b"];
                        if(lambda_b) solver.lambda_b = lambda_b.value();
                        sol::optional<T> alpha = solver_params["alpha"];
                        if(alpha) solver.alpha = alpha.value();
                        sol::optional<T> beta = solver_params["beta"];
                        if(beta) solver.beta = beta.value();
                        sol::optional<T> J_min = solver_params["J_min"];
                        if(J_min) solver.J_min = J_min.value();

                        setup_and_solve(solver);
                    } else if(eq_icase_any(solver_type, "newton")) {
                        PetscNewton solver{fespace, disc, conv_criteria, ls};
                        setup_and_solve(solver);
                    }
                }
            };
#else 
            AnomalyLog::log_anomaly(Anomaly{"No non-petsc newton solvers currently implemented.", general_anomaly_tag{}});
#endif

        }
    }
}
