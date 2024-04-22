/**
 * @file hypercube_element.hpp
 * @brief GeometricElement implementation for hypercubes 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "Numtool/point.hpp"
#include "iceicle/fe_definitions.hpp"
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/HypercubeTransformations.hpp>

namespace iceicle {

    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeElement final : public GeometricElement<T, IDX, ndim> {

        private:
        // namespace aliases
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using PointView = MATH::GEOMETRY::PointView<T, ndim>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
        using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

        public:
        static inline transformations::HypercubeElementTransformation<T, IDX, ndim, Pn> transformation{};


        private:
        // ================
        // = Private Data =
        // =   Members    =
        // ================
        IDX _nodes[transformation.n_nodes()];

        public:
        // ====================
        // = GeometricElement =
        // =  Implementation  =
        // ====================
        constexpr int n_nodes() const override { return transformation.n_nodes(); }

        constexpr DOMAIN_TYPE domain_type() const noexcept override  { return DOMAIN_TYPE::HYPERCUBE; }

        constexpr int geometry_order() const noexcept override { return Pn; }

        const IDX *nodes() const override { return _nodes; }

        void transform(NodeArray<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys)
        const override {
            return transformation.transform(node_coords, _nodes, pt_ref, pt_phys);
        }

        JacobianType Jacobian(
            NodeArray< T, ndim > &node_coords,
            const Point &xi
        ) const override {
            return transformation.Jacobian(node_coords, _nodes, xi);
        }

        HessianType Hessian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const override {
            return transformation.Hessian(node_coords, _nodes, xi);
        }

        auto regularize_interior_nodes(
            NodeArray<T, ndim>& node_coords
        ) const -> void override {
            transformation.regularize_nodes(_nodes, node_coords);
        }

        /** @brief set the node index at idx to value */
        void setNode(int idx, int value){_nodes[idx] = value; }
    };
}
