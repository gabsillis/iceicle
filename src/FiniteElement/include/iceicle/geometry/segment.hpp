/** 
 * @file segment.hpp 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A line segment
 * 
 */
#pragma once
#include <iceicle/geometry/geo_element.hpp>

namespace ELEMENT {

    template<typename T, typename IDX>
    class Segment final : public GeometricElement<T, IDX, 1> {
        private:
        static constexpr int ndim = 1;
        static constexpr int nnodes = 2;

        using Point = MATH::GEOMETRY::Point<T, ndim>;

        IDX node_idxs[nnodes];

        public:

        Segment(IDX node1, IDX node2) {
            node_idxs[0] = node1;
            node_idxs[1] = node2;
        }

        constexpr int n_nodes() override { return nnodes; }

        IDX *nodes() override { return node_idxs; }

        void getCentroid(const std::vector<Point> &nodeCoords, T *centroid) override {
            auto &node0 = nodeCoords[node_idxs[0]];
            auto &node1 = nodeCoords[node_idxs[1]];
            centroid[0] = 0.5 * (node0[0] + node1[0]);
        }

        void updateGeometry(std::vector< MATH::GEOMETRY::Point<T, ndim> > &nodeCoords) override {}
    };
}
