/** 
 * @file segment.hpp 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A line segment
 * 
 */
#pragma once
#include "iceicle/fe_definitions.hpp"
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SegmentTransformation.hpp>

namespace iceicle {

    template<typename T, typename IDX>
    class Segment final : public GeometricElement<T, IDX, 1> {
        private:
        static constexpr int ndim = 1;
        static constexpr int nnodes = 2;
        using index_type = IDX; 
        using value_type = T;

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

        IDX node_idxs[nnodes];

        public:

        static inline transformations::SegmentTransformation<T, IDX> transformation{};

        Segment(IDX node1, IDX node2) {
            node_idxs[0] = node1;
            node_idxs[1] = node2;
        }

        constexpr int n_nodes() const override { return nnodes; }

        constexpr DOMAIN_TYPE domain_type() const noexcept override { return DOMAIN_TYPE::HYPERCUBE; }

        constexpr int geometry_order() const noexcept override { return 1; }

        const IDX *nodes() const override { return node_idxs; }

        void transform(NodeArray<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys) const  override {
            return transformation.transform(node_coords, node_idxs, pt_ref, pt_phys);        
        }
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const override 
        { return transformation.Jacobian(node_coords, node_idxs, xi); }


        HessianType Hessian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const override {
            HessianType ret{{{{0}}}};;
            return ret;
        }

        auto regularize_interior_nodes(
            NodeArray<T, ndim>& coord /// [in/out] the node coordinates array 
        ) const -> void override {
            // do nothing: no interior nodes
        }

        
        /// @brief get the number of vertices in a face
        auto n_face_vert(
            int face_number /// [in] the face number
        ) const -> int {
            return 1;
        };

        /// @brief get the vertex indices on the face
        /// NOTE: These vertices must be in the same order as if get_element_vert() 
        /// was called on the transformation corresponding to the face
        auto get_face_vert(
            int face_number,      /// [in] the face number
            index_type* vert_fac  /// [out] the indices of the vertices of the given face
        ) const -> void override {
            switch (face_number) {
                case 0:
                    vert_fac[0] = node_idxs[0];
                    break;
                case 1:
                    vert_fac[1] = node_idxs[1];
                    break;
            }
        };


        auto get_face_nodes(
            int face_number,      /// [in] the face number
            index_type* nodes_fac  /// [out] the indices of the nodes of the given face
        ) const -> void override {
            switch (face_number) {
                case 0:
                    nodes_fac[0] = node_idxs[0];
                    break;
                case 1:
                    nodes_fac[1] = node_idxs[1];
                    break;
            }
        };

        /// @brief get the face number of the given vertices 
        /// @return the face number of the face with the given vertices
        auto get_face_nr(
            index_type* vert_fac /// [in] the indices of the vertices of the given face
        ) const -> int override {
            return node_idxs[vert_fac[0]];
        };
    };
}
