/**
 * @file geo_element.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Abstract definition for Geometric Elements
 * @date 2023-06-26
 */

#pragma once
#include <vector>
#include <Numtool/point.hpp>
#include <Numtool/matrixT.hpp>

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
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        /**
         * @brief Get the number of nodes
         * 
         * @return int the number of nodes for this element
         */
        virtual
        constexpr int n_nodes() const = 0;

        /**
         * @brief Get the nodes array for this element
         * 
         * @return std::ptrdiff_t* the array of nodes
         */
        virtual
        const IDX *nodes() const = 0;

        /**
         * @brief get the Jacobian matrix of the transformation
         * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] xi the position in the reference domain at which to calculate the Jacobian
         * @param [out] the jacobian matrix
         */
        virtual
        void Jacobian(
            std::vector<Point> &node_coords,
            const Point &xi,
            T J[ndim][ndim]
        ) const = 0;

        /**
         * @brief get the Hessian of the transformation
         * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
         *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] xi the position in the reference domain at which to calculate the hessian
         * @param [out] the Hessian in tensor form indexed [k][i][j] as described above
         */
        virtual
        void Hessian(
            std::vector<Point> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const = 0;
    };
}
