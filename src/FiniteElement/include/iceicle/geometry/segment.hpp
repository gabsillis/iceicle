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

        using Point = GEOMETRY::Point<T, ndim>;

        IDX node_idxs[nnodes];
        T jacobian_; /// precomputed jacobian

        public:

        Segment(IDX node1, IDX node2) {
            node_idxs[0] = node1;
            node_idxs[1] = node2;
        }

        void updateGeometry(const std::vector<Point> &nodeCoords) override {
            auto &node0 = nodeCoords[node_idxs[0]];
            auto &node1 = nodeCoords[node_idxs[1]];
            jacobian_ = 0.5 * std::abs(node1[0] - node0[0]);
        }

        constexpr int n_nodes() override { return nnodes; }

        IDX *nodes() override { return node_idxs; }

        const T *referenceDomnNodes(const IDX i) override {
            static constexpr T refnodes[2] = {-1, 1};
            return &(refnodes[i]);
        }

        void getCentroid(const std::vector<Point> &nodeCoords, T *centroid) override {
            auto &node0 = nodeCoords[node_idxs[0]];
            auto &node1 = nodeCoords[node_idxs[1]];
            centroid[0] = 0.5 * (node0[0] + node1[0]);
        }

        void jacobian(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T *jac) 
        override {
            jac[0] = jacobian_;
        }

        T jacobianDet(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi) override {
            return jacobian_;
        }

        void jacobianAdj(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T *adj) 
        override {
            adj[0] = 1;
        }

        void hessian(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T hess[ndim][ndim][ndim]) 
        override {
            ***hess = 0;
        }

        void convertRefToAct(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &ref, GEOMETRY::Point<T, ndim> &act) 
        override {
            auto &node0 = nodeCoords[node_idxs[0]];
            auto &node1 = nodeCoords[node_idxs[1]];
            T dist = GEOMETRY::distance(node0, node1);
            act = node0[0] + 0.5 * dist * (ref[0] + 1.0) * dist;
        }
    };
}
