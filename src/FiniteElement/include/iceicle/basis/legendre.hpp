/**
 * @brief legendre basis functions
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once 
#include <Numtool/fixed_size_tensor.hpp>
#include "iceicle/basis/tensor_product.hpp"
#include <cmath>
#include "iceicle/basis/basis.hpp"

namespace BASIS {

    /**
     * @brief legendre basis functions in 1d 
     */
    template<typename T, int Pn>
    struct legendre_1d {

        // ============
        // = Typedefs =
        // ============

        /// alias Tensor
        template<typename T1, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T1, sizes...>;

        /// @brief the type of the real values
        using value_type = T;

        /// @brief the number of basis functions 
        static constexpr int nbasis = Pn + 1;

        /**
         * @brief evaluate every polynomial at the given point 
         * overloaded for consistent interface
         * @param xi the point in the reference domain [-1, 1]
         *           to evaluate at 
         * @param [out] Nj array of the evaluations
         */
        inline void eval_all(T xi, Tensor<T, nbasis>& Nj) const {
            using namespace std;
            switch(Pn){
                case 10:
                    Nj[10] = (
                        46189 * pow(xi, 10)
                        - 109395 * pow(xi, 8)
                        + 90090 * pow(xi, 6)
                        - 30030 * pow(xi, 4)
                        + 3465 * pow(xi, 2)
                        - 63
                    ) / 256.0;
                    // continue down: don't break
                case 9:
                    Nj[9] = (
                        12155 * pow(xi, 9)
                        - 25740 * pow(xi, 7)
                        + 18018 * pow(xi, 5)
                        - 4620 * pow(xi, 3)
                        + 315 * xi
                    ) / 128.0;
                case 8:
                    Nj[8] = (
                        6435 * pow(xi, 8)
                        - 12012 * pow(xi, 6)
                        + 6930 * pow(xi, 4)
                        - 1260 * pow(xi, 2)
                        + 35
                    ) / 128.0;
                case 7:
                    Nj[7] = (
                        429 * pow(xi, 7)
                        - 693 * pow(xi, 5)
                        + 315 * pow(xi, 3)
                        - 35 * xi
                    );
                case 6:
                    Nj[6] = (
                        231 * pow(xi, 6)
                        - 315 * pow(xi, 4)
                        + 105 * pow(xi, 2)
                        - 5
                    ) / 16.0;
                case 5:
                    Nj[5] = (
                        63 * pow(xi, 5)
                        - 70 * pow(xi, 3)
                        + 15 * xi
                    ) / 8.0;
                case 4:
//                    Nj[4] = (
//                        35 * pow(xi, 4)
//                        - 30 * pow(xi, 2)
//                        + 3
//                    ) / 8.0;

//                  // floating point match to Dr. Luo's impl
                    Nj[4] = (35.0 * xi * xi * xi * xi - 30 * xi * xi + 3) * 0.125;
                case 3:
                    Nj[3] = (
                        5 * pow(xi, 3)
                        - 3 * xi
                    ) / 2.0;
                case 2:
                    Nj[2] = (
                        3 * pow(xi, 2)
                        - 1
                    ) / 2.0;
                case 1:
                    Nj[1] = xi;
                case 0:
                    Nj[0] = 1.0;
                    break;
                default:
                    // TODO: handle continuation
                    break;
            }
        }

        /**
         * @brief evaluate every polynomial at the given point 
         * @param xi the point in the reference domain [-1, 1]
         *           to evaluate at 
         * @return an array of the evaluations
         */
        inline constexpr Tensor<T, nbasis> eval_all(T xi) const {
            Tensor<T, nbasis> bi{};
            eval_all(xi, bi);
            return bi;
        }

        /**
         * @brief evaluate the polynomials and first derivatives at the given point
         * @param xi the point in the reference domain [-1, 1]
         *           to evaluate at 
         * @param [out] Nj the basis function evaluations 
         * @param [out] dNj the derivative evaluations 
         */
        void deriv_all(
            T xi,
            Tensor<T, nbasis> &Nj,
            Tensor<T, nbasis> &dNj
        ) const {
            using namespace std;
            eval_all(xi, Nj);

            switch(Pn){
                case 10:
                    dNj[10] = 55.0 / 128.0 * xi * (
                        4199 * pow(xi, 8)
                        - 7956 * pow(xi, 6)
                        + 4914 * pow(xi, 4)
                        - 1092 * pow(xi, 2) 
                        + 63
                    );
                    // continue down: don't break
                case 9:
                    dNj[9] = 45.0 / 128.0 * (
                        2431 * pow(xi, 8)
                        - 4004 * pow(xi, 6)
                        + 2002 * pow(xi, 4)
                        - 308 * pow(xi, 2)
                        + 7
                    );
                case 8:
                    dNj[8] = 9.0 / 16.0 * xi * (
                        715 * pow(xi, 6)
                        - 1001 * pow(xi, 4)
                        + 385 * pow(xi, 2)
                        - 35
                    );
                case 7:
                    dNj[7] = 7.0 / 16.0 * (
                        429 * pow(xi, 6)
                        - 495 * pow(xi, 4)
                        + 135 * pow(xi, 2)
                        - 5
                    );
                case 6:
                    dNj[6] = 21.0 / 8.0 * xi * (
                        33 * pow(xi, 4)
                        - 30 * pow(xi, 2)
                        + 5
                    );
                case 5:
                    dNj[5] = 15.0 / 8.0 * (
                        21 * pow(xi, 4)
                        - 14 * pow(xi, 2)
                        + 1
                    );
                case 4:
                    dNj[4] = 5.0 / 2.0 * xi * (7 * pow(xi, 2) - 3);
                case 3:
                    dNj[3] = 1.5 * (5 * pow(xi, 2) - 1);
                case 2:
                    dNj[2] = 3 * xi;
                case 1:
                    dNj[1] = 1.0;
                case 0:
                    dNj[0] = 0.0;
                    break;
                default:
                    // TODO: handle continuation
                    break;
            }
        }

        /**
         * @brief get the basis value, 1st and 2nd derivative 
         * of all the basis functions evaluated at the given point 
         * @param [in] xi the location in the reference domain [-1, 1]
         * @param [out] Nj the basis function values 
         * @param [out] dNj the derivative values 
         * @param [out] d2Nj the second derivative values 
         */
        void d2_all(
            T xi,
            Tensor<T, nbasis> &Nj,
            Tensor<T, nbasis> &dNj,
            Tensor<T, nbasis> &d2Nj
        ) const {
            using namespace std;
            deriv_all(xi, Nj, dNj);
           
            switch(Pn) {
                case 10:
                    d2Nj[10] = 495.0 / 128.0 * (
                        4199 * pow(xi, 8)
                        - 6188 * pow(xi, 6)
                        + 2730 * pow(xi, 4)
                        - 364 * pow(xi, 2)
                        + 7
                    );
                case 9:
                    d2Nj[9] = 495.0 / 16.0 * xi *(
                        221 * pow(xi, 6)
                        - 273 * pow(xi, 4)
                        + 91 * pow(xi, 2)
                        - 7
                    );
                case 8:
                    d2Nj[8] = 315.0 / 16.0 * (
                        143 * pow(xi, 6)
                        - 143 * pow(xi, 4)
                        + 33 * pow(xi, 2)
                        -1
                    );
                case 7:
                    d2Nj[7] = 63.0 / 8.0 * xi * (
                        143 * pow(xi, 4)
                        - 110 * pow(xi, 2)
                        + 15
                    );
                case 6:
                    d2Nj[6] = 105.0 / 8.0 * (
                        33 * pow(xi, 4)
                        - 18 * pow(xi, 2)
                        + 1
                    );
                case 5:
                    d2Nj[5] = 105 / 2.0 * xi * (
                        3 * pow(xi, 2) - 1
                    );
                case 4:
                    d2Nj[4] = 15.0 / 2.0 * (7 * pow(xi, 2) - 1);
                case 3:
                    d2Nj[3] = 15 * xi;
                case 2:
                    d2Nj[2] = 3.0;
                case 1:
                    d2Nj[1] = 0.0;
                case 0:
                    dNj[0] = 0.0;
                default:
                    // TODO: handle continuation
                    break;
            }
        }
    };


    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeLegendreBasis final: public Basis<T, ndim> {

        static inline legendre_1d<T, Pn> basis_1d;
        using Basis1DType = decltype(basis_1d);

        // TODO: is this the right tensor product?
        static inline BASIS::QTypeProduct<T, ndim, Basis1DType::nbasis> tensor_prod;
        using TensorProdType = decltype(tensor_prod);

        using Point = MATH::GEOMETRY::Point<T, ndim>;
        public:

        // ==============
        // = Basis Impl =
        // ==============

        constexpr int nbasis() const override { return TensorProdType::nvalues; }

        constexpr FE::DOMAIN_TYPE domain_type() const noexcept override { return FE::DOMAIN_TYPE::HYPERCUBE; }

        void evalBasis(const T*xi, T *Bi) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            tensor_prod.fill_shp(basis_1d, xipt, Bi);
        }

        void evalGradBasis(const T *xi, T *dBidxj) const override {
            Point xipt{};
            std::copy_n(xi, ndim, xipt.data());
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, TensorProdType::nvalues, ndim> dBi; 
            tensor_prod.fill_deriv(basis_1d, xipt, dBi);
            // TODO: get rid of memmove called here
            std::copy_n(dBi.ptr(), ndim * TensorProdType::nvalues, dBidxj);
        }
        
        void evalHessBasis(const T *xi, T *HessianData) const override {
            (void) tensor_prod.fill_hess(basis_1d, xi, HessianData);
        }

        bool isOrthonormal() const override { return false; }

        bool isNodal() const override { return true; }

        inline int getPolynomialOrder() const override { return Pn; }
    };
}
