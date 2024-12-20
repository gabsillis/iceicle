#pragma once 
#include <array>
#include <vector>

namespace iceicle {

    namespace shapes {

        // @brief a Hypercube 
        // Vertices at every corner of the n dimensional hypercube 
        // local vertex indices are ordered last dimension fastest
        // e.g in a square 
        //
        // 1---3   ^ y
        // |   |   |
        // 0---2   --> x
        template< int ndim >
        struct hypercube {

            // @brief get the number of facets of given dimensionality 
            // e.g facet_dim 0 is points which for a 3D hypercube would be 8
            // @param facet_dim the dimensionality of the facet
            [[nodiscard]] static inline constexpr 
            auto n_facets(int facet_dim) noexcept -> int 
            {
                static constexpr std::array<int, ndim + 1> facet_counts = []{
                    std::array<int, ndim + 1> ret;
                    ret[0] = 1;
                    for(int idim = 1; idim <= ndim; ++idim){
                        // extrude each dimension

                        ret[idim] = 1;
                        for(int jdim = idim - 1; jdim > 0; --jdim){
                            ret[jdim] = ret[jdim - 1] + 2 * ret[jdim];
                        }
                        ret[0] *= 2; // vertices
                    }
                    return ret;
                }();
                return facet_counts[facet_dim]; 
            }

            // @brief get the local vertex indices of the vertices of the given facet
            // @param facet_dim, the dimensionality of the face 
            // @param facet_number the unique index for the facet (consistent with face numbers)
            [[nodiscard]] static inline constexpr 
            auto get_vertices(int facet_dim, int facet_number)
            -> std::vector<int>
            {
                return std::vector<int>{};
                // TODO: rethink
            }

        };
    }
}
