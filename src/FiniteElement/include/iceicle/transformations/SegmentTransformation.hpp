/**
 * @brief SegmentTransformation.hpp
 * Transformations to a bi-unit segment reference domain centered on xi = 0;
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include <Numtool/integer_utils.hpp>
#include <Numtool/point.hpp>
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/fe_definitions.hpp>

namespace iceicle::transformations {
    
    /**
     * @brief Transformation to the reference bi-unit simplex
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large integers
     */
    template<typename T, typename IDX>
    class SegmentTransformation{    
        private:

        static constexpr int ndim = 1;

        static constexpr T xi_poin[2] = { -1.0, 1.0 };

        // === Aliases ===
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        SegmentTransformation() {}

        /** @brief get the number of nodes that define a segment */
        constexpr int nnodes() const { return 2; }

        /**
         * @brief transform from the reference domain to the physcial domain
         * T(s): s -> x
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to the element
         * @param [in] xi the position in the refernce domain
         * @param [out] x the position in the physical domain
         */
        void transform(
            NodeArray<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi, Point &x
        ) const {
            T x1 = node_coords[node_indices[0]][0];
            T x2 = node_coords[node_indices[1]][0];
            T dist = x2 - x1;
            x[0] = x1 + dist * (xi[0] + 1.0) / 2.0;
        }

        /**
         * @brief get the Jacobian matrix of the transformation
         * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to this element in order
         * @param [in] xi the position in the reference domain at which to calculate the Jacobian
         * @return the jacobian matrix
         */
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
            NodeArray<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi
        ) const {
            T jacobian_ = 0.5 * (
                node_coords[node_indices[1]][0] 
                - node_coords[node_indices[0]][0]
            );
            return NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>{{jacobian_}};  
        }

        /**
         * @brief get the Hessian of the transformation
         * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
         *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to this element in order
         * @param [in] xi the position in the reference domain at which to calculate the hessian
         * @param [out] the Hessian in tensor form indexed [k][i][j] as described above
         */
        void Hessian(
            NodeArray<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const {
            hess[0][0][0] = 0.0;
        }

        /**
         * @brief get a pointer to the array of node coordinates in the reference domain
         * @return the Lagrange points in the reference domain
         */
        const Point *reference_nodes() const { return xi_poin; }
    };

    /**
     * @brief transformation from the reference trace space 
     * to the left and right element space
     *
     * @tparam T the floating point type
     * @tparam IDX the indexing type for large lists
     */
    template<typename T, typename IDX>
    class PointTraceTransformation {
        
        using ElPoint = MATH::GEOMETRY::Point<T, 1>;
        using TracePoint = MATH::GEOMETRY::Point<T, 0>;

        public: 
        /**
         * @brief transform from the trace space reference domain 
         *        to the left and right element reference domain
         *
         * WARNING: This assumes the vertices are the first ndim+1 nodes
         * This is the current ordering used in SimplexElementTransformation
         * @param [in] node_indicesL the global node indices for the left element
         * @param [in] node_indicesR the global node indices for the right element
         * @param [in] faceNrL the left trace number 
         * (the position of the barycentric coordinate that is 0 for all points on the face)
         * @param [in] faceNrR the right trace number (see faceNrL)
         * @param [in] s the location in the trace reference domain
         * @param [out] xiL the position in the left reference domain
         * @param [out] xiR the position in the right reference domain
         */
        void transform(
                IDX *node_indicesL,
                IDX *node_indicesR,
                int traceNrL, int traceNrR,
                const TracePoint &s,
                ElPoint &xiL, ElPoint &xiR
        ) const {
            xiL[0] = (traceNrL == 0) ? -1.0 : 1.0;
            xiR[0] = (traceNrR == 0) ? -1.0 : 1.0;
        }
    };
}
