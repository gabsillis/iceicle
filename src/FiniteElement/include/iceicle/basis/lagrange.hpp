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
#include "iceicle/fe_definitions.hpp"
#include "iceicle/transformations/polytope_transformations.hpp"
#include <algorithm>
#include <iceicle/basis/basis.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <mdspan/mdspan.hpp>
namespace iceicle {
    
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

        static constexpr polytope::tcode<ndim> tri_code{0};
        static inline transformations::SimplexElementTransformation<T, IDX, ndim, Pn> transform;
        UniformLagrangeInterpolation<T, Pn> basis1d{};
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:

        // ==============
        // = Basis Impl =
        // ==============

        int nbasis() const override { return transform.nnodes(); }

        constexpr DOMAIN_TYPE domain_type() const noexcept override { return DOMAIN_TYPE::SIMPLEX; }

        void evalBasis(const T *xi, T *Bi) const override {

            T x = xi[0];
            T y = xi[1];
            switch(Pn){
                case 0:
                    Bi[0] = 1.0;
                    break;
                case 1:
                {
                    // 1d functions
                    auto l0 = [](T s) { return 1.0 - s; };
                    auto l1 = [](T s) { return s; };

                    Bi[0] = l0(x) * l0(y);
                    Bi[1] = l1(x) * l0(y);
                    Bi[2] = l0(x) * l1(y);
                    break;
                }
                case 2:
                {
                    auto l0 = [](T s) { return 2 * (s - 0.5) * (s - 1); };
                    auto l1 = [](T s) { return -4 * (s) * (s - 1); };
                    auto l2 = [](T s) { return 2 * (s - 0.5) * (s); };
                    Bi[0] = l0(x) * l0(y);
                    Bi[1] = l1(x) * l0(y);
                    Bi[2] = l2(x) * l0(y);
                    Bi[3] = l0(x) * l1(y);
                    Bi[4] = l1(x) * l1(y);
                    Bi[5] = l0(x) * l2(y);
                    break;
                }
            }
//            for(int inode = 0; inode < transform.nnodes(); ++inode){
//                Bi[inode] = transform.shp(xi, inode);
//            }
        }

        void evalGradBasis(const T *xi, T *grad_data) const override 
        {
            T x = xi[0];
            T y = xi[1];
            switch(Pn){
                case 0:
                    grad_data[0] = 0.0;
                    grad_data[1] = 0.0;
                    break;
                case 1:
                {
                    std::extents grad_extents{3, 2};
                    std::mdspan grad{grad_data, grad_extents};

                    // 1d functions
                    auto l0 = [](T s) { return 1.0 - s; };
                    auto dl0 = [](T s) { return -1.0; };
                    auto l1 = [](T s) { return s; };
                    auto dl1 = [](T s) { return 1.0; };

                    grad[0, 0] = dl0(x) * l0(y);
                    grad[0, 1] = l0(x) * dl0(y);
                    grad[1, 0] = dl1(x) * l0(y);
                    grad[1, 1] = l1(x) * dl0(y);
                    grad[2, 0] = dl0(x) * l1(y);
                    grad[2, 1] = l0(x) * dl1(y);
                    break;
                }
                case 2:
                {
                    std::extents grad_extents{6, 2};
                    std::mdspan grad{grad_data, grad_extents};

                    // 1d functions
                    auto l0 = [](T s) { return 2 * (s - 0.5) * (s - 1); };
                    auto dl0 = [](T s) { return 4 * (s - 0.75); };
                    auto l1 = [](T s) { return -4 * (s) * (s - 1); };
                    auto dl1 = [](T s) { return 4 - 8 * s; };
                    auto l2 = [](T s) { return 2 * (s - 0.5) * (s); };
                    auto dl2 = [](T s) { return 4 * (s - 0.25); };
                    grad[0, 0] = dl0(x) * l0(y);
                    grad[0, 1] = l0(x) * dl0(y);
                    grad[1, 0] = dl1(x) * l0(y);
                    grad[1, 1] = l1(x) * dl0(y);
                    grad[2, 0] = dl2(x) * l0(y);
                    grad[2, 1] = l2(x) * dl0(y);
                    grad[3, 0] = dl0(x) * l1(y);
                    grad[3, 1] = l0(x) * dl1(y);
                    grad[4, 0] = dl1(x) * l1(y);
                    grad[4, 1] = l1(x) * dl1(y);
                    grad[5, 0] = dl0(x) * l2(y);
                    grad[5, 1] = l0(x) * dl2(y);
                    break;
                }
            }
//            auto dB = std::experimental::mdspan(dBidxj, transform.nnodes(), ndim);
//            for(int inode = 0; inode < transform.nnodes(); ++inode){
//                for(int jderiv = 0; jderiv < ndim; ++jderiv){
//                   dB[inode, jderiv] = transform.dshp(xi, inode, jderiv);
//                }
//            }
        }

        void evalHessBasis(const T *xi, T *hess_data) const override 
        {
            T x = xi[0];
            T y = xi[1];
            switch(Pn) {
                case 0:
                    hess_data[0] = 0.0;
                    hess_data[1] = 0.0;
                    hess_data[2] = 0.0;
                    hess_data[3] = 0.0;
                    break;
                case 1:
                {
                    std::extents hess_extents{3, 2, 2};
                    std::mdspan hess{hess_data, hess_extents};

                    // 1d functions
                    auto l0 = [](T s) { return 1.0 - s; };
                    auto dl0 = [](T s) { return -1.0; };
                    auto d2l0 = [](T s) { return 0.0; };
                    auto l1 = [](T s) { return s; };
                    auto dl1 = [](T s) { return 1.0; };
                    auto d2l1 = [](T s) { return 0.0; };

                    hess[0, 0, 0] = 0.0;
                    hess[0, 0, 1] = dl0(x) * dl0(y);
                    hess[0, 1, 0] = dl0(x) * dl0(y);
                    hess[0, 1, 1] = 0.0;

                    hess[1, 0, 0] = 0.0;
                    hess[1, 0, 1] = dl1(x) * dl0(y);
                    hess[1, 1, 0] = dl1(x) * dl0(y);
                    hess[1, 1, 1] = 0.0;

                    hess[2, 0, 0] = 0.0;
                    hess[2, 0, 1] = dl0(x) * dl1(y);
                    hess[2, 1, 0] = dl0(x) * dl1(y);
                    hess[2, 1, 1] = 0.0;
                    break;
                }
                case 2:
                {
                    std::extents hess_extents{6, 2, 2};
                    std::mdspan hess{hess_data, hess_extents};

                    // 1d functions
                    auto l0 = [](T s) { return 2 * (s - 0.5) * (s - 1); };
                    auto dl0 = [](T s) { return 4 * (s - 0.75); };
                    auto ddl0 = [](T s) { return 4 * s; };
                    auto l1 = [](T s) { return -4 * (s) * (s - 1); };
                    auto dl1 = [](T s) { return 4 - 8 * s; };
                    auto ddl1 = [](T s) { return - 8 * s; };
                    auto l2 = [](T s) { return 2 * (s - 0.5) * (s); };
                    auto dl2 = [](T s) { return 4 * (s - 0.25); };
                    auto ddl2 = [](T s) { return 4 * s ; };


                    hess[0, 0, 0] = ddl0(x) *   l0(y);
                    hess[0, 0, 1] =  dl0(x) *  dl0(y);
                    hess[0, 1, 0] =  dl0(x) *  dl0(y);
                    hess[0, 1, 1] =   l0(x) * ddl0(y);

                    hess[1, 0, 0] = ddl1(x) *   l0(y);
                    hess[1, 0, 1] =  dl1(x) *  dl0(y);
                    hess[1, 1, 0] =  dl1(x) *  dl0(y);
                    hess[1, 1, 1] =   l1(x) * ddl0(y);

                    hess[2, 0, 0] = ddl2(x) *   l0(y);
                    hess[2, 0, 1] =  dl2(x) *  dl0(y);
                    hess[2, 1, 0] =  dl2(x) *  dl0(y);
                    hess[2, 1, 1] =   l2(x) * ddl0(y);

                    hess[3, 0, 0] = ddl0(x) *   l1(y);
                    hess[3, 0, 1] =  dl0(x) *  dl1(y);
                    hess[3, 1, 0] =  dl0(x) *  dl1(y);
                    hess[3, 1, 1] =   l0(x) * ddl1(y);

                    hess[4, 0, 0] = ddl1(x) *   l1(y);
                    hess[4, 0, 1] =  dl1(x) *  dl1(y);
                    hess[4, 1, 0] =  dl1(x) *  dl1(y);
                    hess[4, 1, 1] =   l1(x) * ddl1(y);

                    hess[5, 0, 0] = ddl0(x) *   l2(y);
                    hess[5, 0, 1] =  dl0(x) *  dl2(y);
                    hess[5, 1, 0] =  dl0(x) *  dl2(y);
                    hess[5, 1, 1] =   l0(x) * ddl2(y);
                }
            }
//            Point xipt{};
//            std::copy_n(xi, ndim, xipt.data());
//            std::extents hessian_extents{nbasis(), ndim, ndim};
//            std::mdspan hess{hess_data, hessian_extents};
//            polytope::fill_hess<T, tri_code>(basis1d, xipt, hess);
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };


    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeLagrangeBasis final: public Basis<T, ndim> {

        static inline UniformLagrangeInterpolation<T, Pn> lagrange_1d;
        using Basis1DType = decltype(lagrange_1d);
        static inline QTypeProduct<T, ndim, Basis1DType::nbasis> tensor_prod;
        using TensorProdType = decltype(tensor_prod);

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        public:

        // ==============
        // = Basis Impl =
        // ==============

        int nbasis() const override { return TensorProdType::nvalues; }

        constexpr DOMAIN_TYPE domain_type() const noexcept override { return DOMAIN_TYPE::HYPERCUBE; }

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
