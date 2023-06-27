/**
 * @file HypercubeGaussLegendre.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Gauss-Legendre Quadrature defined on the reference hypercube
 * @version 0.1
 * @date 2022-01-20
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once
#include <iceicle/quadrature/QuadratureRule.hpp>
#include <iceicle/geometry/point.hpp>
namespace QUADRATURE {

    /**
     * @brief Gauss-Legendre Quadrature defined on the reference hypercube
     * 
     * @tparam T the floating point type
     * @tparam IDX the index type
     * @tparam ndim the number of dimensions
     * @tparam npoin the number of quadrature point 
     */
    template<typename T, typename IDX, int ndim, int npoin> 
    class HypercubeGaussLegendre final 
    : public QuadratureRule<T, IDX, ndim>,
      public TraceQuadratureRule<T, IDX, ndim> {
        static_assert(ndim >= 0);
        private:
        static constexpr int MAX_DEFINED_NPOIN = 6;
        static constexpr int MIN_DEFINED_NPOIN = 1;
       
        template<int x, int y>
        static constexpr int power_T_safe(){
            if constexpr(y < 0) return 1;
            else if constexpr(y == 0) return 1;
            else if constexpr(y == 1) return x;
            else return power_T_safe<x, y-1>() * x;
        }

        /* constexpr version for internal use */
        static constexpr int npoin_ndim = MATH::power_T<npoin, ndim>::value; 
        static constexpr int data_offset = MATH::consecutiveSum<MIN_DEFINED_NPOIN, npoin - 1>(); // offset for start for this order
        


        static constexpr T abscissae[MATH::consecutiveSum<MIN_DEFINED_NPOIN, MAX_DEFINED_NPOIN>()] = {
            0,
            -std::sqrt(3.0) / 3.0,
             std::sqrt(3.0) / 3.0,
            0,
            -std::sqrt(15) / 5.0,
             std::sqrt(15) / 5.0,
            -std::sqrt(525 - 70 * std::sqrt(30)) / 35.0,
             std::sqrt(525 - 70 * std::sqrt(30)) / 35.0,
            -std::sqrt(525 + 70 * std::sqrt(30)) / 35.0,
             std::sqrt(525 + 70 * std::sqrt(30)) / 35.0,
            0,
            -std::sqrt(245 - 14 * std::sqrt(70)) / 21.0,
             std::sqrt(245 - 14 * std::sqrt(70)) / 21.0,
            -std::sqrt(245 + 14 * std::sqrt(70)) / 21.0,
             std::sqrt(245 + 14 * std::sqrt(70)) / 21.0,
             0.6612093864662645,
             -0.6612093864662645,
             -0.2386191860831969,
             0.2386191860831969,
             -0.9324695142031521,
             0.9324695142031521
        };

        static constexpr T weights[MATH::consecutiveSum<MIN_DEFINED_NPOIN, MAX_DEFINED_NPOIN>()] = {
            2,
            1,
            1,
            8.0 / 9.0,
            5.0 / 9.0,
            5.0 / 9.0,
            1 / 36.0 * (18 + std::sqrt(30)),
            1 / 36.0 * (18 + std::sqrt(30)),
            1 / 36.0 * (18 - std::sqrt(30)),
            1 / 36.0 * (18 - std::sqrt(30)),
            128.0 / 225.0,
            (322 + 13 * std::sqrt(70)) / 900.0,
            (322 + 13 * std::sqrt(70)) / 900.0,
            (322 - 13 * std::sqrt(70)) / 900.0,
            (322 - 13 * std::sqrt(70)) / 900.0,
            0.3607615730481386,
            0.3607615730481386,
            0.4679139345726910,
            0.4679139345726910,
            0.1713244923791704,
            0.1713244923791704
        };

        GEOMETRY::Point<T, ndim> points_nd[npoin_ndim];
        T weights_nd[npoin_ndim];

        template<int idim, int ndim_arg>
        void dataFill(GEOMETRY::Point<T, ndim_arg> *points_nd, T *weights_nd){
            static_assert(idim >= 0);
            static_assert(ndim >= 0);
            if constexpr (idim == 1 && npoin > 0){
                for(int p = 0; p < npoin; p++){
                    points_nd[p][idim - 1] = abscissae[p + data_offset];
                    weights_nd[p] = weights[p + data_offset];
                }
            } else {
                constexpr int npoin_idim = power_T_safe<npoin, idim>();
                constexpr int npoin_idimm1 = power_T_safe<npoin, idim - 1>();
                for(int p = 0; p < npoin; p++){
                    // drill down
                    dataFill<idim - 1>(points_nd + p * npoin_idimm1, weights_nd + p * npoin_idimm1);
                    // fill all points
                    for(int pfill = p * npoin_idimm1; pfill < (p + 1) * npoin_idimm1; pfill++){
                        points_nd[pfill][idim - 1] = abscissae[p + data_offset];
                        weights_nd[pfill] *= weights[p + data_offset];
                    }
                }
            }
        }

        public:
        HypercubeGaussLegendre(){
            dataFill<ndim, ndim>(points_nd, weights_nd);
        }

        inline int npoints() const override { return npoin_ndim; }

        inline const GEOMETRY::Point<T, ndim> *quadraturePoints() const override { 
            return points_nd;
         }

        inline const T *quadratureWeights() const override {
            if constexpr (ndim == 1) {
                return weights + data_offset;
            } else {
                return weights_nd;
            }
        }

    };


    template<typename T, typename IDX, int order> 
    class HypercubeGaussLegendre<T, IDX, 0, order> final 
    : public QuadratureRule<T, IDX, 0>,
      public TraceQuadratureRule<T, IDX, 0> {
        GEOMETRY::Point<T, 0> quadpoint;
        static constexpr T quadweight = {1.0};
        int npoints() const override{ return 1; }

        const GEOMETRY::Point<T, 0> *quadraturePoints() const override {return &quadpoint;}

        const T *quadratureWeights() const override{return &quadweight;}
    };
 
}
