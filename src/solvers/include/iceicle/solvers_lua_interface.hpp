/// @brief lua interface to dispatch solvers 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/disc/l2_error.hpp"
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
#include <utility>

#ifdef ICEICLE_USE_PETSC
#include <iceicle/corrigan_lm.hpp>
#include <iceicle/petsc_newton.hpp>
#include <iceicle/matrix_free_newton_krylov.hpp>
#endif

namespace iceicle::solvers {

    namespace impl {
        template<class Disc_Type>
        struct source_term_callback_supported {
            using value_type = Disc_Type::value_type;
            using callback_opt_type = std::optional< std::function<void(const value_type*, value_type*)> >;
            static constexpr bool value = requires(Disc_Type disc) {
                {disc.user_source} -> std::convertible_to<callback_opt_type>;
            };
        };
    }

    /// @brief Whether or not the discretization type supports the source term callback interface 
    /// Given a real value type T = Disc_Type::value_type 
    /// The source term callback interface is a data member named user_source 
    /// of type std::optional<std::function<void(T*,T*)>>
    ///
    /// This function takes a position in the physical domain (ndim) 
    /// and outputs (neq) source terms 
    ///
    /// @tparam DiscType the discretization type
    template<class Disc_Type>
    concept source_term_callback_supported = impl::source_term_callback_supported<Disc_Type>::value;

    template<class Disc_Type>
    auto add_source_term_callback(Disc_Type &disc, sol::table config_tbl)
    -> void 
    {
        if constexpr(source_term_callback_supported<Disc_Type>){
            using value_type = Disc_Type::value_type;
            static constexpr int ndim = Disc_Type::dimensionality;
            static constexpr int neq = Disc_Type::nv_comp;
            sol::table cons_law_tbl = config_tbl["conservation_law"];
            sol::optional<sol::function> source_opt = cons_law_tbl["source"];
            if(source_opt){
                sol::function source_f = source_opt.value();
                std::function<void(const value_type*, value_type*)> func 
                    = [source_f](const value_type *x_in, value_type *fout)
                    {
                        std::integer_sequence seq = std::make_integer_sequence<int, ndim>{};
                        auto helper = [source_f]<int...Eq_Indices>(const value_type* x_in, value_type* f_out,
                                std::integer_sequence<int, Eq_Indices...> seq){
                            tmp::sized_tuple_t<value_type, neq> result = source_f(x_in[Eq_Indices]...);
                            auto arr = tmp::get_array_from_tuple(result);
                            std::ranges::copy(arr, f_out);
                        };
                        helper(x_in, fout, seq);
                    };
                disc.user_source = std::optional{func};
            }
        }
    }

    /// @brief Create a writer for output files 
    /// given the user configuration
    ///
    ///
    /// @param config_tbl the user configuration lua table 
    /// @param fespace the finite element space 
    /// @param disc the discretization
    /// @param u the solution to write
    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_get_writer(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        DiscType& disc,
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
                    dat_writer.register_fields(u, disc.field_names);
                    writer = io::Writer{dat_writer};
                } else {
                    AnomalyLog::log_anomaly(Anomaly{"dat writer not defined for greater than 1D", general_anomaly_tag{}});
                }
            }

            // .vtu writer 
            if(writer_name && eq_icase(writer_name.value(), "vtu")){
                io::PVDWriter<T, IDX, ndim> pvd_writer{};
                pvd_writer.register_fespace(fespace);
                pvd_writer.register_fields(u, disc.field_names);
                writer = pvd_writer;
            }
        }
        return writer;
    }

    /// @brief create a writer for residuals
    /// @param config_tbl the user configuration lua table 
    /// @param fespace the finite element space 
    /// @param disc the discretization
    /// @param u the solution to write
    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_get_residuals_writer(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        DiscType& disc,
        fespan<T, LayoutPolicy> u 
    ) -> io::Writer 
    {
        using namespace iceicle::util;
        io::Writer writer;
        sol::optional<sol::table> output_tbl_opt = config_tbl["output"];
        if(output_tbl_opt){
            sol::table output_tbl = output_tbl_opt.value();
            sol::optional<std::string> writer_name = output_tbl["writer"];

            // .vtu writer 
            if(writer_name && eq_icase(writer_name.value(), "vtu")){
                io::PVDWriter<T, IDX, ndim> pvd_writer{};
                pvd_writer.collection_name = "residuals";
                pvd_writer.register_fespace(fespace);
                pvd_writer.register_residuals(u, disc.residual_names, disc);
                writer = pvd_writer;
            }
        }
        return writer;
    }


    template<class T, class IDX, int ndim, class disc_type, class LayoutPolicy>
    auto lua_select_mdg_geometry(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        disc_type& disc, 
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
                disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, ic_res);

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
                            auto node_data = fespace.meshptr->coord[inode];
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
            sol::optional<sol::table> mixed_uniform_mesh_tbl_opt = config_tbl["mixed_uniform_mesh"];
            if(mixed_uniform_mesh_tbl_opt){
                sol::table mesh_table = mixed_uniform_mesh_tbl_opt.value();

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
                            auto node_data = fespace.meshptr->coord[inode];
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
            sol::optional<sol::table> burgers_mesh_tbl_opt = config_tbl["burgers_mesh"];
            if constexpr(ndim == 2) if(burgers_mesh_tbl_opt){
                sol::table mesh_table = burgers_mesh_tbl_opt.value();

                std::array<IDX, ndim> nelem{{3, 2}};

                // bounding box
                std::array<T, ndim> xmin{{0, 0}};
                std::array<T, ndim> xmax{{1.0, 0.5}};
                mesh_parameterizations::hyper_rectangle(nelem, xmin, xmax, geo_map);

                // === Dirichlet BC => nodes cannot move ===
                for(auto trace : fespace.get_boundary_traces()){
                    if(trace.face->bctype == BOUNDARY_CONDITIONS::DIRICHLET){
                        for(index_type inode : trace.face->nodes_span()){
                            auto node_data = fespace.meshptr->coord[inode];
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

            sol::optional<sol::table> user_mesh_tbl_opt = config_tbl["user_mesh"];
            if(user_mesh_tbl_opt){
                sol::table mesh_table = user_mesh_tbl_opt.value();

                sol::optional<sol::table> constraints_opt = mesh_table["geo_constraints"];
                if(constraints_opt){

                    // bounding box
                    sol::optional<sol::table> bounding_box_opt = constraints_opt.value()["bounding_box"];
                    if(bounding_box_opt){
                        std::array<T, ndim> xmin;
                        std::array<T, ndim> xmax;
                        sol::table bounding_box_table = bounding_box_opt.value();
                        for(int idim = 0; idim < ndim; ++idim){
                            xmin[idim] = bounding_box_table["min"][idim + 1];
                            xmax[idim] = bounding_box_table["max"][idim + 1];
                        }
                        mesh_parameterizations::bounding_box(*(fespace.meshptr), xmin, xmax, geo_map);

                    }

                    /// manually fixed nodes 
                    sol::optional<sol::table> fixed_nodelist_opt = constraints_opt.value()["fixed_nodes"];
                    if(fixed_nodelist_opt){
                        sol::table nodelist_tbl = fixed_nodelist_opt.value();
                        std::vector<IDX> fixed_nodes{};
                        for(int i = 1; i <= nodelist_tbl.size(); ++i){
                            IDX inode = nodelist_tbl[i];
                            fixed_nodes.push_back(inode);
                        }
                        mesh_parameterizations::fixed_nodelist(std::span{fixed_nodes}, *(fespace.meshptr), geo_map);
                    }

                }
                // === Dirichlet BC => nodes cannot move ===
                for(auto trace : fespace.get_boundary_traces()){
                    if(trace.face->bctype == BOUNDARY_CONDITIONS::DIRICHLET){
                        for(index_type inode : trace.face->nodes_span()){
                            auto node_data = fespace.meshptr->coord[inode];
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
        DiscType& disc, 
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
                io::Writer residuals_writer{lua_get_residuals_writer(config_tbl, fespace, disc, u)};

                solver.vis_callback = [&](ExplicitSolverType& solver) mutable {
                    T sum = 0.0;
                    for(int i = 0; i < solver.res_data.size(); ++i){
                        sum += SQUARED(solver.res_data[i]);
                    }

#ifdef ICEICLE_USE_MPI
                    T sum_reduce;
                    MPI_Allreduce(&sum, &sum_reduce, 1, mpi_get_type<T>(), MPI_SUM, MPI_COMM_WORLD);
                    int myrank;
                    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                    if(myrank == 0){
                        std::cout << std::setprecision(8);
                        std::cout << "itime: " << std::setw(6) << solver.itime 
                            << " | t: " << std::setw(14) << solver.time
                            << " | residual l2: " << std::setw(14) << std::sqrt(sum_reduce) 
                            << std::endl;

                    }
#else
                    std::cout << std::setprecision(8);
                    std::cout << "itime: " << std::setw(6) << solver.itime 
                        << " | t: " << std::setw(14) << solver.time
                        << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
                        << std::endl;
#endif
                    if(writer) writer.write(solver.itime, solver.time);
                    if(residuals_writer) residuals_writer.write(solver.itime, solver.time);
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
        } else if(eq_icase_any(solver_type, "newton", "lm", "gauss-newton", "mfnk", "matrix-free-newton")) {
            // Newton Solvers
#ifdef ICEICLE_USE_PETSC
            
            // default is machine zero convergence 
            // with maximum of 5 nonlinear iterations
            ConvergenceCriteria<T, IDX> conv_criteria{
                .tau_abs = solver_params.get_or("tau_abs", std::numeric_limits<T>::epsilon()),
                .tau_rel = solver_params.get_or("tau_rel", 0.0),
                .kmax = solver_params.get_or("kmax", 5)
            };

            // select the linesearch type
            LinesearchVariant<T, IDX> linesearch;
            sol::optional<sol::table> ls_arg_opt = solver_params["linesearch"];
            if(ls_arg_opt){
                sol::table ls_arg = ls_arg_opt.value();
                if(eq_icase(ls_arg["type"].get<std::string>(), "wolfe") 
                        || eq_icase(ls_arg["type"].get<std::string>(), "cubic"))
                {
                    IDX kmax = ls_arg.get_or("kmax", 5);
                    T alpha_initial = ls_arg.get_or("alpha_initial", 1.0);
                    T alpha_max = ls_arg.get_or("alpha_max", 10.0);
                    T c1 = ls_arg.get_or("c1", 1e-4);
                    T c2 = ls_arg.get_or("c2", 0.9);
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

            // create writers
            io::Writer writer{lua_get_writer(config_tbl, fespace, disc, u)};
            io::Writer residuals_writer{lua_get_residuals_writer(config_tbl, fespace, disc, u)};

            // create an equivalent H1 Fespace for MDG residuals
            FESpace space_h1{fespace.meshptr};
            fe_layout_right h1_layout{fespace.cg_map, tmp::to_size<DiscType::nv_comp>()};
            std::vector<T> h1_mdg_residual_data(h1_layout.size());
            fespan res_mdg_h1{h1_mdg_residual_data, h1_layout};

            // writer for mdg residuals
            io::Writer writer_mdg{lua_get_writer(config_tbl, space_h1, disc, res_mdg_h1)};
            writer_mdg.rename_collection("mdg_residuals");
            ic_residual_layout<T, IDX, ndim, DiscType::nv_comp> ic_layout{geo_map};

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
                                writer.write(k, (T) k);

                                write_restart(fespace, u, k);
                        };

                        // diagnostics callback views the residuals
                        solver.diag_callback = [&](IDX k, Vec res_data, Vec du_data){
                            residuals_writer.write(k, (T) k);

                            // get the MDG residuals 
                            petsc::VecSpan res_span{res_data};
                            dofspan ic_residual{res_span.data() + u.size(), ic_layout};
                            auto res_mdg_dof_view = dof_view(res_mdg_h1);
                            extract_icespan(ic_residual, res_mdg_dof_view);
                            writer_mdg.write(k, (T) k);
                        };

                        // write the final iteration
                        IDX kfinal = solver.solve(u);
                        writer.write(kfinal, (T) kfinal);
                        residuals_writer.write(kfinal, (T) kfinal);
                    };

                    if(eq_icase_any(solver_type, "lm", "gauss-newton")){
                        bool form_subproblem = solver_params.get_or("form_subproblem_mat", false); 
                        bool sparse_jacobian = solver_params.get_or("sparse_jacobian_calculation", true);
                        CorriganLM solver{fespace, disc, conv_criteria, ls, geo_map, form_subproblem, sparse_jacobian};

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
                    } else if(eq_icase_any(solver_type, "mfnk", "matrix-free-newton")) {
                        MFNK solver{fespace, disc, conv_criteria, ls, geo_map};
                        setup_and_solve(solver);
                    }
                }
            };
#else 
            AnomalyLog::log_anomaly(Anomaly{"No non-petsc newton solvers currently implemented.", general_anomaly_tag{}});
#endif

        }
    }


    template<class T, class IDX, int ndim, class DiscType, class LayoutPolicy>
    auto lua_error_analysis(
        sol::table config_tbl,
        FESpace<T, IDX, ndim>& fespace,
        DiscType& disc, 
        fespan<T, LayoutPolicy> u
    ) -> void {
        using namespace iceicle::util;

        sol::optional<sol::table> post_opt = config_tbl["post"];
        if(post_opt){
            sol::table post_config = post_opt.value();

            // get the exact solution and convert to std::vunction
            sol::optional<sol::function> exact_opt = post_config["exact_solution"];
            sol::optional<sol::table> tasks_opt = post_config["tasks"];

            if(exact_opt) {
                sol::function exact = exact_opt.value();

                if(tasks_opt){
                    sol::table task_list = tasks_opt.value();
                    for(auto o : task_list){
                        std::string task_name = o.second.as<std::string>();
                        if(eq_icase(task_name, "l2_error")){
                            // =================
                            // = L2 error Task =
                            // =================
                            
                            std::function<void(T*, T*)> exactfunc = [exact](T *x, T *u_exact){
                                if constexpr(DiscType::nv_comp == 1){
                                    // 1 equation case 
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        u_exact[0] = exact(x[Indices]...);
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                } else {
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        sol::table fout = exact(x[Indices]...);
                                        for(int i = 0; i < DiscType::nv_comp; ++i)
                                            u_exact[i] = fout[i + 1]; // lua 1-index
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                }
                            };

                            T error = l2_error(exactfunc, fespace, u);
#ifdef ICEICLE_USE_MPI
                            error = error * error; // un-sqrt it before we sum :3
                            T error_reduce;
                            MPI_Allreduce(&error, &error_reduce, 1, mpi_get_type<T>(), MPI_SUM, MPI_COMM_WORLD);

                            int myrank;
                            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                            if(myrank == 0)
                                std::cout << "L2 error: " << std::setprecision(12) << std::sqrt(error_reduce) << std::endl;
#else
                            std::cout << "L2 error: " << std::setprecision(12) << error << std::endl;
#endif
                        } 

                        if(eq_icase(task_name, "l1_error")){
                            // =================
                            // = L1 error Task =
                            // =================
                            
                            std::function<void(T*, T*)> exactfunc = [exact](T *x, T *u_exact){
                                if constexpr(DiscType::nv_comp == 1){
                                    // 1 equation case 
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        u_exact[0] = exact(x[Indices]...);
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                } else {
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        sol::table fout = exact(x[Indices]...);
                                        for(int i = 0; i < DiscType::nv_comp; ++i)
                                            u_exact[i] = fout[i + 1]; // lua 1-index
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                }
                            };

                            T error = l1_error(exactfunc, fespace, u);
#ifdef ICEICLE_USE_MPI
                            T error_reduce;
                            MPI_Allreduce(&error, &error_reduce, 1, mpi_get_type<T>(), MPI_SUM, MPI_COMM_WORLD);

                            int myrank;
                            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                            if(myrank == 0)
                                std::cout << "L1 error: " << std::setprecision(12) << error_reduce << std::endl;
#else
                            std::cout << "L1 error: " << std::setprecision(12) << error << std::endl;
#endif
                        } 

                        if(eq_icase(task_name, "linf_error")) {

                            // =========================
                            // = L infinity error Task =
                            // =========================
                            std::function<void(T*, T*)> exactfunc = [exact](T *x, T *u_exact){
                                if constexpr(DiscType::nv_comp == 1){
                                    // 1 equation case 
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        u_exact[0] = exact(x[Indices]...);
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                } else {
                                    std::integer_sequence x_idx_seq = std::make_integer_sequence<int, ndim>();
                                    auto helper = [exact]<int... Indices>(T *x, T *u_exact, std::integer_sequence<int, Indices...> seq){
                                        sol::table fout = exact(x[Indices]...);
                                        for(int i = 0; i < DiscType::nv_comp; ++i)
                                            u_exact[i] = fout[i + 1]; // lua 1-index
                                    };
                                    helper(x, u_exact, x_idx_seq);
                                }
                            };

                            std::vector<T> errors = linf_error(exactfunc, fespace, u);
#ifdef ICEICLE_USE_MPI
                            std::vector<T> errors_reduce(DiscType::nv_comp);
                            MPI_Allreduce(errors.data(),
                                    errors_reduce.data(), DiscType::nv_comp, mpi_get_type<T>(), MPI_MAX, MPI_COMM_WORLD);

                            int myrank;
                            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                            if(myrank == 0){
                                std::cout << "L_infty error: " << std::setprecision(12);
                                for(int iv = 0; iv < DiscType::nv_comp; ++iv){
                                    std::cout << errors_reduce[iv] << " ";
                                }
                                std::cout << std::endl;
                            }
#else
                            std::cout << "L_infty error: " << std::setprecision(12);
                            for(int iv = 0; iv < DiscType::nv_comp; ++iv){
                                std::cout << errors[iv] << " ";
                            }
                            std::cout << std::endl;
#endif
                        }

                        if(eq_icase(task_name, "ic_residual")){
                            // ====================
                            // = ic residual Task =
                            // ====================
                            
                            T error = ic_error(fespace, u, disc);
#ifdef ICEICLE_USE_MPI
                            T error_reduce;
                            MPI_Allreduce(&error, &error_reduce, 1, mpi_get_type<T>(), MPI_SUM, MPI_COMM_WORLD);

                            int myrank;
                            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
                            if(myrank == 0)
                                std::cout << "IC residual norm: " << std::setprecision(12) << error_reduce << std::endl;
#else
                            std::cout << "IC residual norm: " << std::setprecision(12) << error << std::endl;
#endif
                        } 
                    }
                }

            } else {
                util::AnomalyLog::check(!tasks_opt,
                    util::Anomaly{"Post processing tasks require `exact_solution` to be set", util::general_anomaly_tag{}});
            }
            
        }
        

    }
}
