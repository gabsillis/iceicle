#pragma once
#include <array>
/**
 * @brief Transformations from the [-1, 1] hypercube
 * to an arbitrary order hypercube element
 *
 * @tparam T the floating point type
 * @tparam IDX the index type for large lists
 * @tparam ndim the number of dimensions
 * @tparam Pn the polynomial order of the element
 */
template<typename T, typename IDX, int ndim, int Pn>
class HypercubeElementTransformation {
    private:
    /// Nodes in the reference domain
    /// Ordering is vertices, edge nodes, face nodes, ..., internal nodes
    std::array<T, (Pn + 1) * ndim> reference_nodes = []{
        // Generate the vertices

        
    }();
    
};
