/// @brief functions to initialize finite element data 
/// NOTE: This is a Lua interface (only use with -DICEICLE_USE_LUA)
///
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/fe_function/layout_enums.hpp"
#include "iceicle/pvd_writer.hpp"
#include "iceicle/string_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/disc/projection.hpp>
#include <iceicle/linear_form_solver.hpp>
#include <sol/sol.hpp>
#include <utility>

namespace iceicle {

    /// @brief initialize 
    template<class T, class IDX, int ndim, int neq, class LayoutPolicy, class AccessorPolicy>
    auto projection_initialization(
        FESpace<T, IDX, ndim>& fespace,
        ProjectionFunction<T, ndim, neq> func,
        tmp::compile_int<neq> neq_tag,
        fespan<T, LayoutPolicy, AccessorPolicy> u
    ) -> void {
        Projection<T, IDX, ndim, neq> projection{func};
        solvers::LinearFormSolver projection_solver(fespace, projection);
        projection_solver.solve(u);
    }

    /// @brief initialize the fespan u 
    /// @param [in] config_table the configuration lua table to get the ic description from 
    ///             key: "initial_condition"
    /// @param [in] fespace the finite element space 
    /// @param [out] u the solution to set
    template<class T, class IDX, int ndim, class LayoutPolicy, class AccessorPolicy>
    auto initialize_solution_lua(
        sol::table config_table,
        FESpace<T, IDX, ndim>& fespace,
        fespan<T, LayoutPolicy, AccessorPolicy> u
    ) requires (!is_dynamic_size<decltype(u)::static_extent()>::value)
    {
        static constexpr std::size_t neq = decltype(u)::static_extent();

        // check if restart file is specified 
        sol::optional<std::string> restart_name = config_table["restart"];
        if(restart_name){
            read_restart(fespace, u, restart_name.value());
        } else {
            // the initial condition function (default to zero)
            std::function<void(const T*, T*)> ic_func = [](const T* xin, T* out){
                for(int ieq = 0; ieq < neq; ++ieq){
                    out[ieq] = 0.0;
                }
            };

            // check if IC specified by name
            sol::optional<std::string> ic_as_name = config_table["initial_condition"];
            if(ic_as_name){
                std::string ic_name = ic_as_name.value();
                if(util::eq_icase(ic_name, "zero")){
                    // already defaulted to this case
                }
            }

            // check if IC is a function
            if constexpr (neq == 1) {
                sol::optional<sol::function> ic_as_function = config_table["initial_condition"];
                if(ic_as_function){
                    sol::function ic_f = ic_as_function.value();
                    ic_func = [ic_f](const T* xin, T* xout){
                        std::integer_sequence seq = std::make_integer_sequence<int, ndim>{};
                        auto helper = [ic_f]<int... Indices>(const T* xin, T* xout, 
                                std::integer_sequence<int, Indices...> seq){
                            xout[0] = ic_f(xin[Indices]...);
                        };
                        helper(xin, xout, seq);
                    };
                }
            } else {
                sol::optional<sol::function> ic_as_function = config_table["initial_condition"];
                if(ic_as_function){
                    sol::function ic_f = ic_as_function.value();
                    ic_func = [ic_f](const T* xin, T* xout){
                        std::integer_sequence seq = std::make_integer_sequence<int, ndim>{};
                        auto helper = [ic_f]<int... Indices>(const T* xin, T* xout, 
                                std::integer_sequence<int, Indices...> seq){
//                            tmp::sized_tuple_t<T, neq> fout = ic_f(xin[Indices]...);
//                            std::array<T, neq> vals = tmp::get_array_from_tuple(fout);
//                            std::ranges::copy(vals, xout);
                            sol::table fout = ic_f(xin[Indices]...);
                            for(int i = 0; i < neq; ++i)
                                xout[i] = fout[i + 1];
                        };
                        helper(xin, xout, seq);
                    };
                }

            }

            projection_initialization(fespace, ic_func, tmp::compile_int<neq>{}, u);
        }
    }
}
