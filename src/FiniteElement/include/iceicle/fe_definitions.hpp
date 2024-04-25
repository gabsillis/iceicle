/**
 * @brief enumerations for general use in the finite element API
 */
#pragma once

#include <vector>
#include <Numtool/point.hpp>
namespace iceicle {

    /// @brief we use a std::vector of Points to represent the nodes
    /// @tparam T the real number type 
    /// @tparam ndim the number of dimensions
    template<class T, int ndim>
    using NodeArray = std::vector<MATH::GEOMETRY::Point<T, ndim>>;

    /** 
     * @brief what the reference domain is 
     * This is important for determining equivalent external API objects
     * such as the element type when creating output files 
     */
    enum class DOMAIN_TYPE : int {
        HYPERCUBE = 0, // maps to a [-1.0, 1.0]^d reference hypercube domain 
        SIMPLEX = 1, // maps to the reference simplex with nodes at the origin and 1.0 in each coordinate direction
        DYNAMIC = 2, // other domain types that will not output cleanly
        N_DOMAIN_TYPES = 3
    };
}
