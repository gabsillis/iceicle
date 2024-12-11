#pragma once 
#include <sol/sol.hpp>
#include <iceicle/disc/navier_stokes.hpp>
namespace iceicle {

    namespace navier_stokes{
        /// @brief parse free stream variables
        /// from a lua table
        ///
        /// The free stream table is a table that contains named values for
        /// quantities that can be used to determine the free stream state 
        ///
        /// Case 1: Specify primitive 
        /// "rho" - density
        /// "u" - u
        /// "T" - temperature
        /// "mu" - viscosity : Optional (determined from T)
        /// "l" - reference length : Optional
        ///
        /// Case 2: Reynolds number and mach number
        /// "rho" - density
        /// "T" - temperature 
        /// "Re" - Reynolds number 
        /// "mach" - mach number 
        /// "l" - reference length : Optional
        ///
        template<class T>
        [[nodiscard]]
        auto parse_free_stream(sol::table free_stream_table, T gamma, T Rgas)
        -> std::optional<FreeStream<T>>
        {
            auto all_has_value = [free_stream_table]
                (std::vector<const char *> required_params) -> bool {
                for(std::string_view key : required_params){
                    sol::optional<T> val = free_stream_table[key];
                    if(!val)
                        return false;
                }
                return true;
            };

            if(all_has_value(std::vector{"rho", "u", "T"})){
                T temp = free_stream_table["T"];
                T mu = free_stream_table.get_or("mu", 
                    Sutherlands<T>{}(temp));
                return std::optional{FreeStream<T>{
                    .rho_inf = free_stream_table["rho"],
                    .u_inf = free_stream_table["u"],
                    .temp_inf = temp,
                    .mu_inf = mu,
                    .l_ref = free_stream_table["l"].get_or((T) 1.0)
                }};
            }

            if(all_has_value(std::vector{"rho", "T", "Re", "mach"})){
               T temp = free_stream_table["T"];
               T csound = std::sqrt(gamma * Rgas * temp);
               T mach = free_stream_table["mach"];
               T u = mach / csound;
               T l_ref = free_stream_table["l"].get_or((T) 1.0);
               T rho = free_stream_table["rho"];
               T Re = free_stream_table["Re"];
               T mu = rho * u * l_ref / Re;

               return std::optional{navier_stokes::FreeStream<T>{
                    .rho_inf = rho,
                    .u_inf = u,
                    .temp_inf = temp,
                    .mu_inf = mu,
                    .l_ref = l_ref
               }};
            }

            return std::nullopt;
        }

        /// @brief parse the physical quantities from 
        /// the lua table describing the conservation law
        template<class T>
        auto parse_physical_quantities(sol::table cons_law_tbl)
        -> std::tuple<T, T, T>
        {
            T gamma = cons_law_tbl.get_or("gamma", 1.4);
            T Rgas = cons_law_tbl.get_or("Rgas", 287.052874);
            T Pr = cons_law_tbl.get_or("Pr", 0.72);
            return std::tuple{gamma, Rgas, Pr};
        }

        /// @brief parse the physics for NS from 
        /// the lua table describing the conservation law
        template< class T, int ndim >
        auto parse_physics(sol::table cons_law_tbl){
            using namespace util;
            // get physical quantities
            auto [gamma, Rgas, Pr] = parse_physical_quantities<T>(cons_law_tbl);

            // get the free stream quantities 
            sol::optional<sol::table> free_stream_table_opt 
                = cons_law_tbl["free_stream"];
            std::optional<FreeStream<T>> free_stream_opt = std::nullopt;
            if(free_stream_table_opt){
                free_stream_opt = parse_free_stream(free_stream_table_opt.value(), gamma, Rgas);
            }

            // set up the reference parameters
            ReferenceParameters<T> ref;
            sol::optional<std::string> ref_param_name_opt = cons_law_tbl["reference_quantities"];
            if(ref_param_name_opt){
                std::string ref_param_name = ref_param_name_opt.value();

                // Case 1: use free stream 
                if(eq_icase(ref_param_name, "free_stream")){
                    if(free_stream_opt){
                        FreeStream<T> fs = free_stream_opt.value();
                        ref = ReferenceParameters{
                            fs.rho_inf, fs.u_inf, fs.temp_inf, fs.mu_inf, fs.l_ref};
                    } else {
                        util::AnomalyLog::log_anomaly("Free stream reference parameterizaton specified "
                                "but free stream state was not successfully parsed.");
                    }
                }
            } else {
                ref = ReferenceParameters<T>{};
            }

            Physics<T, ndim> physics{ref, gamma, Rgas, Pr};
            if(free_stream_opt)
                physics.free_stream = free_stream_opt.value();
            return physics;
        }
    }
}
