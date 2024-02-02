/**
 * @brief Template Metaprogramming (TMP) utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <concepts>
#include <type_traits>
namespace ICEICLE::TMP{

    /** @brief a compile time constant integer type
     * for template argument deduction */
    template<int ival>
    using compile_int = std::integral_constant<int, ival>;

}
