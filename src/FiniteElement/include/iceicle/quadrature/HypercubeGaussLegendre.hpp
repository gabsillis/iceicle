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
#include <iceicle/quadrature/quadrules_1d.hpp>
#include <iceicle/geometry/point.hpp>
#include <Numtool/integer_utils.hpp>
namespace QUADRATURE {

    template<typename T, typename IDX, int ndim, int npoin1d>
    class HypercubeGaussLegendre final : public QuadratureRule<T, IDX, ndim>, TraceQuadratureRule<T, IDX, ndim> {

        // the total number of quadrature points
        static constexpr int num_poin = MATH::power_T<npoin1d, ndim>();

        // ==================
        // = Working Arrays =
        // ==================
        QuadraturePoint<T, ndim> qpoints[num_poin];

        public:
        HypercubeGaussLegendre(){
            GaussLegendreQuadrature<T, IDX, npoin1d> quadrule_1d{};

            for(int idim = 0; idim < ndim; ++idim){
                // number of times to repeat the loop over 1d point set
                const int nrepeat = std::pow(npoin1d, idim);
                // the size that one loop through the quadrature point set gives
                const int blocksize = std::pow(npoin1d, ndim - idim);
                for(int irep = 0; irep < nrepeat; ++irep){
                    for(int ipoin = 0; ipoin < npoin1d; ++ipoin){
                        const int nfill = std::pow(npoin1d, ndim - idim - 1);

                        // offset for this point 
                        const int start_offset = ipoin * nfill;

                        for(int ifill = 0; ifill < nfill; ++ifill) {
                            const int offset = irep * blocksize + start_offset;
                            qpoints[offset + ifill].abscisse[idim] = quadrule_1d[ipoin].abscisse[0];
                            if(idim == 0){
                                qpoints[offset + ifill].weight = quadrule_1d[ipoin].weight;
                            } else {
                                qpoints[offset + ifill].weight *= quadrule_1d[ipoin].weight;
                            }
                        }
                    }
                }
            }
        }

        int npoints() const override { return num_poin; }

        const QuadraturePoint<T, ndim> &getPoint(int ipoint) const override { return qpoints[ipoint]; }
    };
 
}
