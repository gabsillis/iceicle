/** 
 * @file segment.hpp 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A line segment
 * 
 */
#pragma once
#include "iceicle/fe_enums.hpp"
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

        static inline TRANSFORMATIONS::SegmentTransformation<T, IDX> transformation{};

        Segment(IDX node1, IDX node2) {
            node_idxs[0] = node1;
            node_idxs[1] = node2;
        }

        constexpr int n_nodes() const override { return nnodes; }

        constexpr FE::DOMAIN_TYPE domain_type() const noexcept override { return FE::DOMAIN_TYPE::HYPERCUBE; }

        constexpr int geometry_order() const noexcept override { return 1; }

        const IDX *nodes() const override { return node_idxs; }

        void transform(FE::NodalFEFunction<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys) const  override {
            return transformation.transform(node_coords, node_idxs, pt_ref, pt_phys);        
        }
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
            FE::NodalFEFunction<T, ndim> &node_coords,
            const Point &xi
        ) const override 
        { return transformation.Jacobian(node_coords, node_idxs, xi); }


        void Hessian(
            FE::NodalFEFunction<T, ndim> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const override {
            transformation.Hessian(node_coords, node_idxs, xi, hess);
        }
    };
}
