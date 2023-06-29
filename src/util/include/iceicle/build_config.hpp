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

namespace BUILD_CONFIG {
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
}
