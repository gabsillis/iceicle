/// @brief lua interface for boundary conditions
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/anomaly_log.hpp"
#include <functional>
#include <utility>
#include <vector>
#include <sol/sol.hpp>
#include <iceicle/tmp_utils.hpp>

namespace iceicle {

    namespace impl {
        template<class Disc_Type>
        struct dirichlet_callback_supported {
            using value_type = Disc_Type::value_type;
            using container_type = std::vector< std::function<void(const value_type*, value_type*)>>;
            static constexpr bool value = requires(Disc_Type disc) {
                {disc.dirichlet_callbacks} -> std::convertible_to<container_type>;
            };
        };


        template<class Disc_Type>
        struct neumann_callback_supported {
            using value_type = Disc_Type::value_type;
            using container_type = std::vector< std::function<void(const value_type*, value_type*)>>;
            static constexpr bool value = requires(Disc_Type disc) {
                {disc.dirichlet_callbacks} -> std::convertible_to<container_type>;
            };
        };
    }

    /// @brief Whether or not the discretization type supports the dirichlet callback interface 
    /// Given a real value type T = Disc_Type::value_type 
    /// The dirichlet callback interface is a data member named dirichlet_callbacks
    /// of type std::vector<std::function<void(T*, T*)>>
    ///
    /// where each of the functions that will be stored take a physical domain position of size ndim
    /// and convert it to an output value of size neq
    /// each output value represents the prescribed solution value
    /// The bcflag will correspond to the index in dirichlet_callbacks (0-indexed)
    ///
    /// @tparam DiscType the discretization type
    template<class Disc_Type>
    concept dirichlet_callback_supported = impl::dirichlet_callback_supported<Disc_Type>::value;

    /// @brief Whether or not the discretization type supports the neumann callback interface 
    /// Given a real value type T = Disc_Type::value_type 
    /// The neumann callback interface is a data member named neumann_callbacks
    /// of type std::vector<std::function<void(T*, T*)>>
    ///
    /// where each of the functions that will be stored take a physical domain position of size ndim
    /// and convert it to an output value of size neq
    /// each output value represents the prescribed normal flux
    /// The bcflag will correspond to the index in neumann_callbacks (0-indexed)
    ///
    /// @tparam DiscType the discretization type
    template<class Disc_Type>
    concept neumann_callback_supported = impl::neumann_callback_supported<Disc_Type>::value;

    /// @brief add dirichlet callback functions from the lua interface 
    /// @param disc the discretization 
    ///        must specify nv_comp and dimensionality
    /// @param the table (usually keyed ["boundary_conditions"]) that specifies the boundary conditions 
    ///        In this table there should be a ["dirichlet"] table 
    ///        entries in this table can be real values (which get converted to a callback function here)
    ///        or functions which get registered with a callback function
    template<class Disc_Type>
    auto add_dirichlet_callbacks(Disc_Type &disc, sol::table bc_table) -> void
    requires(dirichlet_callback_supported<Disc_Type>) {
        using value_type = Disc_Type::value_type;
        static constexpr int neq = Disc_Type::nv_comp;
        static constexpr int ndim = Disc_Type::dimensionality;
        sol::optional<sol::table> dirichlet_opt = bc_table["dirichlet"];
        if(dirichlet_opt){
            sol::table dirichlet_tbl = dirichlet_opt.value();
            for(int itbl = 1; itbl <= dirichlet_tbl.size(); ++itbl){

                // check for value first
                // all equations get set to this value
                sol::optional<value_type> dirichlet_val = dirichlet_tbl[itbl];
                if(dirichlet_val){
                    value_type val = dirichlet_val.value();
                    std::function<void(const value_type*, value_type*)> func 
                        = [val](const value_type *x, value_type *out){
                        for(int ieq = 0; ieq < neq; ++ieq){
                            out[ieq] = val;
                        }
                    };
                    disc.dirichlet_callbacks.push_back(func);
                    continue;
                }

                // check for function 
                // this should be a function that returns a table of size neq for neq > 1
                // or a single value for neq == 1
                sol::optional<sol::function> dirichlet_func = dirichlet_tbl[itbl];
                if(dirichlet_func){
                    sol::function bc_f = dirichlet_func.value();
                    std::function<void(const value_type*, value_type*)> func 
                        = [bc_f](const value_type* xin, value_type* xout)
                        {
                            std::integer_sequence seq = std::make_integer_sequence<int, ndim>{};
                            auto helper = [bc_f]<int... Dim_Indices>(const value_type* xin, value_type* xout,
                                    std::integer_sequence<int, Dim_Indices...> seq){
                                if constexpr (neq == 1){
                                    xout[0] = bc_f(xin[Dim_Indices]...);
                                } else {
                                    tmp::sized_tuple_t<value_type, neq> result 
                                        = bc_f(xin[Dim_Indices]...);
                                    NUMTOOL::TMP::constexpr_for_range<0, neq>(
                                        [result, xout]<int ieq>{
                                            xout[ieq] = std::get<ieq>(result);
                                            return 0;
                                        }
                                    );
                                }
                            };
                            helper(xin, xout, seq);
                        };
                    disc.dirichlet_callbacks.push_back(func);
                    continue;
                }

                // table of values or functions
                sol::optional<sol::table> dirichlet_condition_table = dirichlet_tbl[itbl];
                if(dirichlet_condition_table) {
                    sol::table bc_per_eqn = dirichlet_condition_table.value();
                    util::AnomalyLog::check(bc_per_eqn.size() == Disc_Type::nv_comp, 
                            util::Anomaly{"Number of Dirichlet entries in the table doesn't match the number of equations", util::general_anomaly_tag{}});

                    std::vector<value_type> values{};
                    for(int jtbl = 1; jtbl <= bc_per_eqn.size(); ++jtbl){
                        sol::optional<value_type> bcval = bc_per_eqn[jtbl];
                        if(bcval) values.push_back(bcval.value());
                    }

                    auto func = [values](const value_type *xin, value_type *uout) -> void {
                        std::copy(values.begin(), values.end(), uout);
                    };
                    disc.dirichlet_callbacks.push_back(func);

                    // TODO: check if its a table of values or functions and implement table of functions
                }

            }
        }
    }


    /// @brief add neumann callback functions from the lua interface 
    /// @param disc the discretization 
    ///        must specify nv_comp and dimensionality
    /// @param the table (usually keyed ["boundary_conditions"]) that specifies the boundary conditions 
    ///        In this table there should be a ["neumann"] table 
    ///        entries in this table can be real values (which get converted to a callback function here)
    ///        or functions which get registered with a callback function
    template<class Disc_Type>
    auto add_neumann_callbacks(Disc_Type &disc, sol::table bc_table) -> void
    requires(neumann_callback_supported<Disc_Type>) {
        using value_type = Disc_Type::value_type;
        static constexpr int neq = Disc_Type::nv_comp;
        static constexpr int ndim = Disc_Type::dimensionality;
        sol::optional<sol::table> neumann_opt = bc_table["neumann"];
        if(neumann_opt){
            sol::table neumann_tbl = neumann_opt.value();
            for(int itbl = 1; itbl <= neumann_tbl.size(); ++itbl){

                // check for value first
                // all equations get set to this value
                sol::optional<value_type> neumann_val = neumann_tbl[itbl];
                if(neumann_val){
                    value_type val = neumann_val.value();
                    std::function<void(const value_type*, value_type*)> func 
                        = [val](const value_type *x, value_type *out){
                        for(int ieq = 0; ieq < neq; ++ieq){
                            out[ieq] = val;
                        }
                    };
                    disc.neumann_callbacks.push_back(func);
                    continue;
                }

                // check for function 
                // this should be a function that returns a table of size neq for neq > 1
                // or a single value for neq == 1
                sol::optional<sol::function> neumann_func = neumann_tbl[itbl];
                if(neumann_func){
                    sol::function bc_f = neumann_func.value();
                    std::function<void(const value_type*, value_type*)> func 
                        = [bc_f](const value_type* xin, value_type* xout)
                        {
                            std::integer_sequence seq = std::make_integer_sequence<int, ndim>{};
                            auto helper = [bc_f]<int... Dim_Indices>(const value_type* xin, value_type* xout,
                                    std::integer_sequence<int, Dim_Indices...> seq){
                                if constexpr (neq == 1){
                                    xout[0] = bc_f(xin[Dim_Indices]...);
                                } else {
                                    tmp::sized_tuple_t<value_type, neq> result 
                                        = bc_f(xin[Dim_Indices]...);
                                    NUMTOOL::TMP::constexpr_for_range<0, neq>(
                                        [result, xout]<int ieq>{
                                            xout[ieq] = std::get<ieq>(result);
                                            return 0;
                                        }
                                    );
                                }
                            };
                            helper(xin, xout, seq);
                        };
                    disc.neumann_callbacks.push_back(func);
                    continue;
                }

                // TODO: check for table of values and table of functions
            }
        }
    }
}
