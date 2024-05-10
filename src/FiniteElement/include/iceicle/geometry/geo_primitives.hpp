/// @brief Geometry Primitives
/// @author Gianni Absillis (gabsill@ncsu.edu)
#pragma once
#include <array>

namespace iceicle {

    /// @brief a bounding box
    /// a ndim dimensional hypercube definition
    ///
    /// @tparam T the real value type 
    /// @tparam ndim the number of dimensions
    template<class T, int ndim>
    struct BoundingBox {
        /// @brief the minimal corner of the hypercube
        std::array<T, ndim> xmin;

        /// @brief the maxmimal corner of the hypercube
        std::array<T, ndim> xmax;
    };

}
