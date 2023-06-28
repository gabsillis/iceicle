/**
 * @file geo_element.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Abstract definition for Geometric Elements
 * @date 2023-06-26
 */

#pragma once
#include <vector>
#include <iceicle/geometry/point.hpp>

namespace ELEMENT{
    
    /**
     * @brief A Geometric element
     * Contains information and methods for geometric description of elements
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large lists
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim> 
    class GeometricElement {
        private:

        // namespace aliases
        using Point = GEOMETRY::Point<T, ndim>;

        public:
        /**
         * @brief Perform Geoemetry precomputation
         * precomputes Jacobians, etc 
         *
         * @param nodeCoords the node coordinate array
         */
        virtual
        void updateGeometry(const std::vector<Point> &nodeCoords) = 0;

        /**
         * @brief Get the number of nodes
         * 
         * @return int the number of nodes for this element
         */
        virtual
        constexpr int n_nodes() = 0;

        /**
         * @brief Get the nodes array for this element
         * 
         * @return std::ptrdiff_t* the array of nodes
         */
        virtual
        IDX *nodes() = 0;

        /**
         * @brief Get the node coordinates in the reference domain
         * 
         * @param n the node number
         * @return T* an array of coordinates in the reference domain
         */
        virtual
        const T *referenceDomnNodes(const IDX n) = 0;

        /**
         * @brief Get the location of the centroid of the element
         * @param [in] nodeCoords the node coordinates
         * @param [out] centroid the centroid of the element [ndim]
         */
        virtual
        void getCentroid(const std::vector< Point > &nodeCoords, T *centroid) = 0;

        /**
         * @brief Get the Jacobian 
         * of the mapping from the physical domain to the reference domain
         * \frac{\partial x_i}{\partial \xi_j}
         * The generalized version of the jacobian call
         * 
         * @param [in] nodeCoords the node coordinates
         * @param [in] xi the location in the reference domain
         * @param [out] jac the jacobian
         */
        virtual
        void jacobian(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T *jac) = 0;

        /**
         * @brief Get the determinant of the jacobian
         * 
         * @return T the determinant of the jacobian
         */
        virtual
        T jacobianDet(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi){
            T jac[SQUARED(ndim)];
            jacobian(nodeCoords, xi, jac);
            return MATH::MATRIX_T::determinant<ndim, T>(jac);
        }

        /**
         * @brief Gets the adjugate of the jacobian matrix
         * Adj(\frac{\partial x_i}{\partial \xi_j})
         * 
         * @tparam the number of dimensions to use template optimized adjugate
         * @param xi the position in the reference domain
         * @param adj [out] the adjugate of the jacobian size = ndim * ndim of element
         */
        virtual
        void jacobianAdj(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T *adj) {
            T jac[SQUARED(ndim)];
            jacobian(nodeCoords, xi, jac);
            return MATH::MATRIX_T::adjugate<ndim, T>(jac, adj);
        }

        /**
         * @brief get the hessian of the coordinate transformation
         * \frac{\partial^2 x_i}{\partial \xi_j \partial\xi_k}
         *
         * @param [in] nodeCoords the node coordinates
         * @param [in] xi the point in the reference domain to get the hessian at
         * @param [out] hess the hessian of the coordinate transformation [i][j][k] (see brief)
         */
        virtual 
        void hessian(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &xi, T hess[ndim][ndim][ndim]) = 0;

        /**
         * @brief Convert reference coordinates to actual coordinates
         * provided that the reference coordinates are actually in the domain of the element
         * 
         * @param nodeCoords[in] The node coordinates
         * @param ref [in] the reference coordinates [size = ndim]
         * @param ref [out] the actual coordinates [size = ndim]
         * 
         */
        virtual
        void convertRefToAct(const std::vector< Point > &nodeCoords, const GEOMETRY::Point<T, ndim> &ref, GEOMETRY::Point<T, ndim> &act) = 0;
    };
}
