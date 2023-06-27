/**
 * @file QuadratureRule.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Quadrature Rule Abstract Definition
 * @version 0.1
 * @date 2022-01-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once
#include <iceicle/geometry/point.hpp>

namespace QUADRATURE {

    /**
     * @brief An abstract definition of a quadrature rule
     * Quadrature rules will be defined on a reference domain
     * Quadrature rules provide quadrature points \xi_{i,g}
     * - these points are used for evaluation of the function being integrated
     * - these points are provided in the reference domain
     * 
     * Quadrature rules provide quadrature weights for integration w_g
     * 
     * an integration of the function f(\xi_i) is \sum\limits_g f(\xi_{i, g}) w_g
     * 
     * Quadrature rules also provide quadrature points and weights for the reference trace space
     * 
     * @tparam T the floating point type
     * @tparam IDX the index type
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim>
    class QuadratureRule {
        public:
        /**
         * @brief Gets the number of quadrature points
         * 
         * @return int the number of quadrature points
         */
        virtual
        int npoints() const = 0;

        /**
         * @brief Gets the quadrature points in the reference domain
         * 
         * @return const GEOMETRY::Point<T, ndim>* an array of the points
         */
        virtual
        const GEOMETRY::Point<T, ndim> *quadraturePoints() const = 0;

        /**
         * @brief The quadrature waights
         * 
         * @return const T* an array wi[npoint]
         */
        virtual
        const T *quadratureWeights() const = 0;
    };

    /**
     * @brief Quadrature rule for the trace space
     * 
     * @tparam T the floating point type
     * @tparam TS_ndim 
     */
    template<typename T, typename IDX, int TS_ndim>
    class TraceQuadratureRule {
        public:
        /**
         * @brief The number of quadrature points for this trace space
         * 
         * @return int the number of quadrature points
         */
        virtual int npoints() const = 0;

        /**
         * @brief The quadrature points for this trace space
         * Points are in the local trace space for the face,
         * this corresponds the local trace space for the Left hand element
         * @return const GEOMETRY::Point<T, ndim - 1>* array of points, size = npoints()
         */
        virtual const GEOMETRY::Point<T, TS_ndim> *quadraturePoints() const = 0;

        /**
         * @brief The quadrature weights for this trace space
         * 
         * @return const T* array of weights, size = npoints()
         */
        virtual const T *quadratureWeights() const = 0;
    };
}
