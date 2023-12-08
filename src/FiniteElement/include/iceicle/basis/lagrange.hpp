/**
 * @file lagrange.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Lagrange Basis functions
 * @version 0.1
 * @date 2023-06-26
 * 
 * @copyright Copyright (c) 2023
 * 
 */
#pragma once
#include "Numtool/polydefs/LagrangePoly.hpp"
#include "iceicle/transformations/HypercubeElementTransformation.hpp"
#include <algorithm>
#include <iceicle/basis/basis.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

namespace BASIS {
    
    /**
     * @brief Lagrange Basis functions on simplex elements
     * @tparam T the floating point type
     * @tparam IDX the index type
     * @tparam ndim the number of dimensions
     * @tparam Pn the polynomial order
     */
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexLagrangeBasis final : public Basis<T, ndim>{
        private:

        static inline ELEMENT::TRANSFORMATIONS::SimplexElementTransformation<T, IDX, ndim, Pn> transform;

        public:

        // ==============
        // = Basis Impl =
        // ==============

        int nbasis() const override { return transform.nnodes(); }

        void evalBasis(const T *xi, T *Bi) const override {
            for(int inode = 0; inode < transform.nnodes(); ++inode){
                Bi[inode] = transform.shp(xi, inode);
            }
        }

        void evalGradBasis(const T *xi, T **dBidxj) const override 
        {
            for(int inode = 0; inode < transform.nnodes(); ++inode){
                for(int jderiv = 0; jderiv < ndim; ++jderiv){
                    dBidxj[inode][jderiv] = transform.dshp(xi, inode, jderiv);
                }
            }
        }

        void evalHessBasis(const T *xi, int ibasis, T Hessian[ndim][ndim]) const override 
        {
            for(int ideriv = 0; ideriv < ndim; ++ideriv){
                for(int jderiv = 0; jderiv < ndim; ++jderiv){
                    Hessian[ideriv][jderiv] = transform.dshp2(xi, ibasis, ideriv, jderiv);
                }
            }
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };


    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeLagrangeBasis final: public Basis<T, ndim> {
        static inline ELEMENT::TRANSFORMATIONS::HypercubeElementTransformation<T, IDX, ndim, Pn> transform{};

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        public:

        // ==============
        // = Basis Impl =
        // ==============

        int nbasis() const override { return transform.n_nodes(); }

        void evalBasis(const T*xi, T *Bi) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            transform.fill_shp(xipt, Bi);
        }

        void evalGradBasis(const T *xi, T **dBidxj) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            transform.fill_deriv(xipt, dBidxj);
        }
        
        void evalHessBasis(const T *xi, int ibasis, T Hessian[ndim][ndim]) const override {
            transform.dshp2(xi, ibasis, Hessian);
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };
}
