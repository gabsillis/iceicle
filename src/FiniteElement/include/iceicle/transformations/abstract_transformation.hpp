#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/fe_function/nodal_fe_function.hpp"
namespace ELEMENT::TRANSFORMATIONS {

    /**
     * @brief Abstract definition of the transformation from 
     * a reference domain to the physical domain 
     */
    template<typename T, typename IDX, int ndim>
    class AbstractElementTransformation {

        public:

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;

        /**
         * @brief transform from the reference domain to the physcial domain
         * T(s): s -> x
         * @param [in] coord the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to the
         * element
         * @param [in] xi the position in the refernce domain
         * @param [out] x the position in the physical domain
         */
        virtual void transform(
            const FE::NodalFEFunction<T, ndim> &coord,
            const IDX *node_indices,
            const Point &xi,
            Point x
        ) const = 0;

        /**
         * @brief get the Jacobian matrix of the transformation
         * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to this element in order
         * @param [in] xi the position in the reference domain at which to calculate the Jacobian
         * @return the Jacobian matrix
         */
        virtual JacobianType Jacobian(
            const FE::NodalFEFunction<T, ndim> &coord,
            const IDX *node_indices,
            const Point &xi
        ) const = 0;

        /**
            * @brief get the Hessian of the transformation
            * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
            *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
            * @param [in] node_coords the coordinates of all the nodes
            * @param [in] node_indices the indices in node_coords that pretain to this element in order
            * @param [in] xi the position in the reference domain at which to calculate the hessian
            * @param [out] the Hessian in tensor form indexed [k][i][j] as described above
            */
        virtual void Hessian(
            const FE::NodalFEFunction<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const = 0;
    };
}
