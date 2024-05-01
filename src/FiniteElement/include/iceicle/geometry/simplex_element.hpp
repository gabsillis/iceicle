/**
 * @file simplex_element.hpp
 * @brief GeometricElement implementation for simplices
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

namespace iceicle {
    
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexGeoElement final : public GeometricElement<T, IDX, ndim> {
        private:
        // namespace aliases
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        /// @brief the transformation that properly converts to the reference domain for this element (must be inline to init)
        static inline transformations::SimplexElementTransformation<T, IDX, ndim, Pn> transformation{};
        using value_type = T;
        using index_type = IDX;

        private:
        // ================
        // = Private Data =
        // =   Members    =
        // ================
        IDX _nodes[transformation.nnodes()];

        public:
        // ====================
        // = GeometricElement =
        // =  Implementation  =
        // ====================
        constexpr int n_nodes() const override { return transformation.nnodes(); }

        constexpr DOMAIN_TYPE domain_type() const noexcept override { return DOMAIN_TYPE::SIMPLEX; }

        constexpr int geometry_order() const noexcept override { return Pn; }

        const IDX *nodes() const override { return _nodes; }

        void transform(NodeArray<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys)
        const override {
            return transformation.transform(node_coords, _nodes, pt_ref, pt_phys);
        }

        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
            NodeArray< T, ndim > &node_coords,
            const Point &xi
        ) const override {
            return transformation.Jacobian(node_coords, _nodes, xi);
        }

        void Hessian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const override {
            return transformation.Hessian(node_coords, _nodes, xi, hess);
        }


        auto regularize_interior_nodes(
            NodeArray<T, ndim>& coord /// [in/out] the node coordinates array 
        ) const -> void {
            // TODO: 
        }

        /// @brief get the number of vertices in a face
        virtual 
        auto n_face_vert(
            int face_number /// [in] the face number
        ) const -> int override {
            return ndim;
        };

        /// @brief get the vertex indices on the face
        /// NOTE: These vertices must be in the same order as if get_element_vert() 
        /// was called on the transformation corresponding to the face
        virtual 
        auto get_face_vert(
            int face_number,      /// [in] the face number
            index_type* vert_fac  /// [out] the indices of the vertices of the given face
        ) const -> void override {
            transformation.getTraceVertices(face_number, _nodes, vert_fac);
        };

        /// @brief get the node indices on the face
        ///
        /// NOTE: Nodes are all the points defining geometry (vertices are endpoints)
        ///
        /// NOTE: These vertices must be in the same order as if get_nodes
        /// was called on the transformation corresponding to the face
        virtual 
        auto get_face_nodes(
            int face_number,      /// [in] the face number
            index_type* nodes_fac /// [out] the indices of the nodes of the given face
        ) const -> void override {
            // TODO: 
        };

        /// @brief get the face number of the given vertices 
        /// @return the face number of the face with the given vertices
        virtual 
        auto get_face_nr(
            index_type* vert_fac /// [in] the indices of the vertices of the given face
        ) const -> int override {
            // TODO: 
            return -1;
        };

        /** @brief set the node index at idx to value */
        void setNode(int idx, int value){_nodes[idx] = value; }
    };
}
