/**
 * @file point_face.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A 1D face type
 *
 */
#pragma once
#include <iceicle/geometry/face.hpp>

namespace ELEMENT {
    
    template<typename T, typename IDX>
    class PointFace final : public Face<T, IDX, 1> {
        static constexpr int ndim = 1;
        using Point = GEOMETRY::Point<T, ndim>;

        // === Index data ===
        /// the node corresponding to this face
        IDX node;

        // === Precomputation ===
        /// the normal vector
        T normal;
        /// the face area
        T area = 1.0;
        /// the centroid in the physical domain
        Point centoid;
    };
}
