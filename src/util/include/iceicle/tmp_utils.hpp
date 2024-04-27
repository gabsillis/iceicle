/**
 * @brief Template Metaprogramming (TMP) utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <type_traits>
#include <variant>
#include <tuple>
#include <mdspan/mdspan.hpp>
namespace iceicle::tmp {

    /** @brief a compile time constant integer type
     * for template argument deduction */
    template<int ival>
    using compile_int = std::integral_constant<int, ival>;

    /// @brief template that takes any integral value and 
    /// converts to a size_t integral_constant
    /// this will still warn if you put a negative (in clang at least)
    template<auto value>
    using to_size = std::integral_constant<std::size_t, static_cast<std::size_t>(value)>;

    /// ================
    /// = Type Queries =
    /// ================

    /// @brief get the index type as defined by the class
    /// ignoring references as this should be a using statement
    template<class T>
    using index_type_of = std::remove_reference_t<T>::index_type;

    /// @brief get the value type as defined by the class
    /// ignoring references as this should be a using statement
    template<class T>
    using value_type_of = std::remove_reference_t<T>::value_type;

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

    /// @brief tag when constructing from range
    struct from_range_t{};


    // ===============
    // = sized tuple =
    // ===============

    namespace impl{
        template<std::size_t remaining, typename T, typename... Types>
        struct sized_tuple_helper{
            using type = sized_tuple_helper<remaining - 1, T, Types..., T>::type;
        };

        template<typename T, typename... Types>
        struct sized_tuple_helper<0, T, Types...> {
            using type = std::tuple<Types...>;
        };
    }

    /// @brief expands into a std::tuple of nvalues type T 
    /// i.e sized_tuple<double, 3>::type is std::tuple<double, double, double>
    template<typename T, std::size_t nvalues>
    struct sized_tuple {
        using type = impl::sized_tuple_helper<nvalues, T>::type;
    };

    /// @brief expands into a std::tuple of nvalues type T 
    /// sized_tuple_t<double, 3> is std::tuple<double, double, double>
    template<typename T, std::size_t nvalues>
    using sized_tuple_t = sized_tuple<T, nvalues>::type;

    static_assert(std::same_as<sized_tuple_t<double, 4>, std::tuple<double, double, double, double>>);
}
