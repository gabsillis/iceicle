/**
 * @file basis.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Basis function abstract definition
 * @version 0.1
 * @date 2023-06-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */

/**
 * @brief namespace for basis functions
 */
#pragma once
#include <stdexcept>

 namespace BASIS {

    template<typename T, int ndim>
    class Basis {
        public:

        /**
         * @brief Get the number of basis functions
         * 
         * @return int the number of basis functions
         */
        virtual
        int nbasis() const = 0;

        /**
         * @brief evaluate basis functions
         * 
         * @param [in] xi the point in the reference domain [size = ndim]
         * @param [out] Bi the values of the basis functions at the point [size = nbasis]
         */
        virtual
        void evalBasis(const T *xi, T *Bi) const = 0;

        /**
         * @brief evaluate the first derivatives of the basis functions
         * 
         * @param [in] xi  the point in the reference domain [size = ndim]
         * @param [out] b the values of the first derivatives of the basis functions at that point
         *                This is in the form of a 2d pointer array that must be preallocated
         *                row major order
         *                \frac{dB_i}{d\xi_j} where i is ibasis
         *                takes a pointer to the first element of this data structure
         *                [size = [nbasis : i][ndim : j]] 
         */
        virtual
        void evalGradBasis(const T *xi, T **dBidxj) const = 0;

        /**
         * @brief evaluate the hessian of the basis functions in the reference domain
         *
         * @param [in] a the point [size = ndim]
         * @param [in] ibasis the basis function index
         * @param [out] the hessian [size = ndim * ndim]
         */
        virtual
        void evalHessBasis(const T*a, int ibasis, T Hessian[ndim][ndim]) const {
            throw std::logic_error("Not Overriden");
        };

        /**
         * @brief Tell if a basis is orthonormal (L2 inner product B_i \otimes B_j is diagonal) or not
         * 
         * @return true if the basis is orthonormal
         * @return false if the basis is not orthonormal
         */
        virtual
        bool isOrthonormal() const = 0;

        /**
         * @brief Tell wether this basis is a Nodal basis or not
         * where the value of each basis function corresponds to a node
         *
         * TODO: consider extract into subclass abstract class
         *
         * @return true if this is a nodal basis
         */
        virtual
        bool isNodal() const { return false; }

        /**
         * @brief Get the Polynomial Order for this basis function
         * 
         * @return int the polynomial order
         */
        virtual
        inline int getPolynomialOrder() const = 0;
    };
 }
