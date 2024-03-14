/**
 * @brief Template Metaprogramming (TMP) utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <concepts>
#include <type_traits>
#include <variant>
#include <tuple>
namespace ICEICLE::TMP{

    /** @brief a compile time constant integer type
     * for template argument deduction */
    template<int ival>
    using compile_int = std::integral_constant<int, ival>;


    // =====================
    // = Variant Utilities =
    // =====================

    /**
     * @brief a list of functors to be selected from
     */
    template <typename... fcns>
    struct select_fcn : fcns... {
        using fcns::operator()...;
    };

    // deduction guide
    template<class... fcns> select_fcn(fcns...) -> select_fcn<fcns...>;

    /** @brief call the corresponding function with the currently active variant using std::visit */
    template <typename... argTs, typename... Fcns>
    constexpr decltype(auto) operator>>(std::variant<argTs...> const& arglist, select_fcn<Fcns...> const& fcnlist){
        std::visit(fcnlist, arglist);
    }

    template<class Visitor, class... Variants>
    constexpr decltype(auto) operator>>(const std::tuple<Variants...> &vars, Visitor &&visitor){
        std::apply([visitor = std::forward<Visitor>(visitor)](auto&&... args){
            std::visit(visitor, args...);
        }, vars);

    }

    template<typename T>
    concept not_default = !std::same_as<T, std::monostate>;

}
