/**
 * @file build_config.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Build configuration option integration
 * @version 0.1
 * @date 2022-11-07
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once
namespace iceicle::build_config {
    #ifdef T_QUAD_PRECISION
    using T = long double;
    #endif

    #ifdef T_SINGLE_PRECISION
    using T = float;
    #endif

    #ifdef T_DOUBLE_PRECISION
    using T = double;
    #endif

    #ifdef IDX_64_bit
    using IDX = long
    #endif

    #ifdef IDX_32_BIT
    using IDX = int;
    #endif

    /** 
     * The maximum polynomial order of basis functions that gets generated at compile time 
     * for fespace 
     */
#ifdef MAX_POLYNOMIAL_ORDER
    static constexpr int FESPACE_BUILD_PN = MAX_POLYNOMIAL_ORDER
#else
    static constexpr int FESPACE_BUILD_PN = 5;
#endif
    /** 
     * The maximum polynomial order of basis functions that gets generated at compile time 
     * for fespace 
     */
#ifdef MAX_GEO_ORDER
    static constexpr int FESPACE_BUILD_GEO_PN = MAX_GEO_ORDER
#else
    static constexpr int FESPACE_BUILD_GEO_PN = 1;
#endif
}
