/// @brief lua interface to dispatch solvers 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/writer.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/explicit_utils.hpp>
#include <iceicle/anomaly_log.hpp>
#include <iceicle/tmp_utils.hpp>
#include <iceicle/explicit_euler.hpp>
#include <iceicle/ssp_rk3.hpp>
#include <iceicle/tvd_rk3.hpp>
#include <iceicle/dat_writer.hpp>
#include <iceicle/writer.hpp>
#include <sol/sol.hpp>

namespace iceicle::solvers {

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
                io::Writer writer;
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
                    // TODO: 
                }

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
        }
        
    }
}
