/**
 * @brief Template Metaprogramming (TMP) utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
namespace ICEICLE::TMP{

    /** @brief a compile time constant integer type
     * for template argument deduction */
    template<int ival>
    struct compile_int{};
}
