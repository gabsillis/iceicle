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
#include "iceicle/basis/lagrange_1d.hpp"
#include "iceicle/basis/tensor_product.hpp"
#include "iceicle/fe_enums.hpp"
#include <algorithm>
#include <iceicle/basis/basis.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <mdspan/mdspan.hpp>
namespace BASIS {
    
    /**
     * @brief Lagrange Basis functions on simplex elements
     * TODO: go to tensor product structure like Hyprecube
     *
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

        constexpr FE::DOMAIN_TYPE domain_type() const noexcept override { return FE::DOMAIN_TYPE::SIMPLEX; }

        void evalBasis(const T *xi, T *Bi) const override {
            for(int inode = 0; inode < transform.nnodes(); ++inode){
                Bi[inode] = transform.shp(xi, inode);
            }
        }

        void evalGradBasis(const T *xi, T *dBidxj) const override 
        {
            auto dB = std::experimental::mdspan(dBidxj, transform.nnodes(), ndim);
            for(int inode = 0; inode < transform.nnodes(); ++inode){
                for(int jderiv = 0; jderiv < ndim; ++jderiv){
                   dB[inode, jderiv] = transform.dshp(xi, inode, jderiv);
                }
            }
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };


    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeLagrangeBasis final: public Basis<T, ndim> {

        static inline BASIS::UniformLagrangeInterpolation<T, Pn> lagrange_1d;
        using Basis1DType = decltype(lagrange_1d);
        static inline BASIS::QTypeProduct<T, ndim, Basis1DType::nbasis> tensor_prod;
        using TensorProdType = decltype(tensor_prod);

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        public:

        // ==============
        // = Basis Impl =
        // ==============

        int nbasis() const override { return TensorProdType::nvalues; }

        constexpr FE::DOMAIN_TYPE domain_type() const noexcept override { return FE::DOMAIN_TYPE::HYPERCUBE; }

        void evalBasis(const T*xi, T *Bi) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            tensor_prod.fill_shp(lagrange_1d, xipt, Bi);
        }

        void evalGradBasis(const T *xi, T *dBidxj) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, TensorProdType::nvalues, ndim> dBi; 
            tensor_prod.fill_deriv(lagrange_1d, xipt, dBi);
            // TODO: get rid of memmove called here
            std::copy_n(dBi.ptr(), ndim * TensorProdType::nvalues, dBidxj);
        }
        
        void evalHessBasis(const T *xi, T *HessianData) const override {
            (void) tensor_prod.fill_hess(lagrange_1d, xi, HessianData);
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };
}
