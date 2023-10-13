/** 
 * @file segment.hpp 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A line segment
 * 
 */
#pragma once
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SegmentTransformation.hpp>

namespace ELEMENT {

    template<typename T, typename IDX>
    class Segment final : public GeometricElement<T, IDX, 1> {
        private:
        static constexpr int ndim = 1;
        static constexpr int nnodes = 2;

        using Point = MATH::GEOMETRY::Point<T, ndim>;

        IDX node_idxs[nnodes];

        public:

        static inline TRANSFORMATIONS::SegmentTransformation<T, IDX> transform{};

        Segment(IDX node1, IDX node2) {
            node_idxs[0] = node1;
            node_idxs[1] = node2;
        }

        constexpr int n_nodes() const override { return nnodes; }

        const IDX *nodes() const override { return node_idxs; }

        void Jacobian(
            std::vector<Point> &node_coords,
            const Point &xi,
            T J[ndim][ndim]
        ) const override 
        { transform.Jacobian(node_coords, node_idxs, xi, J); }


        void Hessian(
            std::vector<Point> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const override {
            transform.Hessian(node_coords, node_idxs, xi, hess);
        }
    };
}
