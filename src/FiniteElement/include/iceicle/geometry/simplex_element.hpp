/**
 * @file simplex_element.hpp
 * @brief GeometricElement implementation for simplices
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fe_definitions.hpp"
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

namespace iceicle {


    /// @brief a linear triangle element
    ///
    /// nodes:
    /// 2
    /// |\
    /// | \
    /// |  \
    /// |   \
    /// 0 -- 1
    ///
    template<class T, class IDX>
    class TriangleElement final : public GeometricElement<T, IDX, 2> {
    public:

        static constexpr int ndim = 2;
        static constexpr int nshp = ndim + 1;
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
        using value_type = T;
        using index_type = IDX;

        // ================
        // = Data Members =
        // ================
        std::array<IDX, nshp> node_idxs;

        // =============================
        // = Coordinate Transformation =
        // =============================

        void transform(
            NodeArray<T, ndim> &node_coords,
            const Point &pt_ref,
            Point &pt_phys
        ) const override {

            std::array<T, nshp> lambdas = {
                pt_ref[0], 
                pt_ref[1],
                1.0 - pt_ref[0] - pt_ref[1]
            };

            std::ranges::fill(pt_phys, 0.0);
            for(int ishp = 0; ishp < nshp; ++ishp) {
                pt_phys[0] += lambdas[ishp] * node_coords[node_idxs[ishp]][0];
                pt_phys[1] += lambdas[ishp] * node_coords[node_idxs[ishp]][1];
            }
        }

        auto Jacobian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const -> JacobianType override {
            T x0 = node_coords[node_idxs[0]][0], x1 = node_coords[node_idxs[1]][0], 
              x2 = node_coords[node_idxs[2]][0];
            T y0 = node_coords[node_idxs[0]][1], y1 = node_coords[node_idxs[1]][1], 
              y2 = node_coords[node_idxs[2]][1];

            JacobianType J{
                { {x0 - x2}, {x1 - x2} },
                { {y0 - y2}, {y1 - y2} },
            };
            return J;
        }

        auto Hessian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const -> HessianType override {
            HessianType H;
            H = 0;
            return H;
        }

        // ===============
        // = Node Access =
        // ===============

        constexpr auto 
        n_nodes() const -> int override
        { return nshp; }

        const IDX *nodes() const 
        { return node_idxs.data(); }

        // =====================
        // = Domain Definition =
        // =====================
        
        constexpr DOMAIN_TYPE domain_type() const noexcept override
        { return DOMAIN_TYPE::SIMPLEX; }

        constexpr int geometry_order() const noexcept override
        { return 1; }


        // =====================
        // = Face Connectivity =
        // =====================
        
        auto n_faces() const -> int override
        { return nshp; }

        auto face_domain_type(int face_number) const -> DOMAIN_TYPE override {
            // we use the hypercube domain type for the line segment face 
            // x \in [-1, 1]
            return DOMAIN_TYPE::HYPERCUBE;
        };

        auto n_face_vert(
            int face_number /// [in] the face number
        ) const -> int override {
            return 2;
        }

        auto get_face_vert(
            int face_number,      
            index_type* vert_fac  
        ) const -> void override {

            // face nodes are the indices that are not the face number 
            // i.e face 0 has nodes 1 and 2
            switch(face_number){
                case 0:
                    vert_fac[0] = node_idxs[1];
                    vert_fac[1] = node_idxs[2];
                    break;
                case 1:
                    vert_fac[0] = node_idxs[2];
                    vert_fac[1] = node_idxs[0];
                    break;
                case 2:
                    vert_fac[0] = node_idxs[0];
                    vert_fac[1] = node_idxs[1];
            }
        }

        auto n_face_nodes(int face_number) const -> int {
            return 2;
        }

        auto get_face_nodes(
            int face_number,      /// [in] the face number
            index_type* nodes_fac /// [out] the indices of the nodes of the given face
        ) const -> void override {
            switch(face_number){
                case 0:
                    nodes_fac[0] = node_idxs[1];
                    nodes_fac[1] = node_idxs[2];
                    break;
                case 1:
                    nodes_fac[0] = node_idxs[2];
                    nodes_fac[1] = node_idxs[0];
                    break;
                case 2:
                    nodes_fac[0] = node_idxs[0];
                    nodes_fac[1] = node_idxs[1];
            }
        }

        auto get_face_nr(
            index_type* vert_fac /// [in] the indices of the vertices of the given face
        ) const -> int override {

            switch(vert_fac[0]){
                case 0:
                    if(vert_fac[1] == 1)
                        return 2;
                    else // second vertex == 2
                        return 1;
                    break;
                case 1:
                    if(vert_fac[1] == 0)
                        return 2;
                    else // second vertex == 2
                        return 0;
                    break;
                case 2:
                    if(vert_fac[1] == 0)
                        return 1;
                    else // second vertex == 1
                        return 0;
                    break;
            }
            return -1;
        }

        // =============
        // = Geometric =
        // =============

        auto regularize_interior_nodes(
            NodeArray<T, ndim>& coord 
        ) const -> void override { /* no internal nodes */ }

        // ===========
        // = Utility =
        // ===========

        /// @brief clone this geometric element (create a copy)
        auto clone() const -> std::unique_ptr<GeometricElement<T, IDX, ndim>> override {
            return std::make_unique<TriangleElement<T, IDX>>(*this);
        };
    };

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

        /// @brief clone this element
        auto clone() const -> std::unique_ptr<GeometricElement<T, IDX, ndim>> override {
            return std::make_unique<SimplexGeoElement<T, IDX, ndim, Pn>>(*this);
        }
    };
}
