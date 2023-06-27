/**
 * @file SimplexQuadrature.hpp
 * @brief Quadrature on simplical elements
 * Using Grundmann Moller formuila
 */

#include <iceicle/quadrature/QuadratureRule.hpp>

namespace QUADRATURE {


    /**
     * @brief Grundmann Moller Simplex Quadrature Rules
     * based on implementation in mfem mfem/fem/intrules.cpp
     * 
     * see also https://epubs-siam-org/doi/epdf/10.1137/0715019
     * @tparam T The floating point type
     * @tparam IDX The Index Type
     * @tparam ndim the number of spatial dimensions
     * @tparam order, the polynomial order of the integration rule
     *      can integrate polynomials of order 2 * order + 1
     */
    template<typename T, typename IDX, int ndim, int order>
        class GrundmannMollerSimplexQuadrature final: public QuadratureRule<T, IDX, ndim>, TraceQuadratureRule<T, IDX, ndim>{
            
            /// d is the polynomial degree of the integration formula
            static constexpr int d = 2 * order + 1;

            /// The number of integration points
            static constexpr int num_poin = MATH::binomial< ndim + order + 1, ndim + 1 >();

            // ==================
            // = Working Arrays =
            // ==================
            T weights[num_poin];
            GEOMETRY::Point<T, ndim> abscisse[num_poin];


            public:
            GrundmannMollerSimplexQuadrature(){

                // fill the factorials
                T factorials[d + ndim + 1];
                factorials[0] = 1.0;
                for(int i = 1; i < d + ndim + 1; ++i){
                    factorials[i] = factorials[i - 1] * i;
                }

                // Loop over 1D points
                int iarr = 0;
                for(unsigned int ipoin = 0; ipoin <= order; ++ipoin){

                    // Compute Weight (eq 4.1 Grundmann Moller)
                    T weight = std::pow(2.0, -2 * order) * MATH::integer_pow(
                        static_cast<double>(d + ndim - 2 * ipoin),
                        static_cast<unsigned int>(d)
                    ) / factorials[ipoin] / factorials[d + ndim - ipoin];

                    if(ipoin % 2){ 
                        weight = -weight;
                    }

                    // compute points (also eq 4.1 Grundmann Moller)
                    int k = order - ipoin;
                    int beta[ndim] = {0};
                    int sums[ndim] = {0};
                    for( ; ;++iarr ){

                        // set the points and weights
                        weights[iarr] = weight;
                        for(int idim = 0; idim < ndim; ++idim){
                            abscisse[iarr][idim] = static_cast<double>(2 * beta[idim] + 1) / (d + ndim - 2 * ipoin);
                        }

                        // mfem method of incrementing beta (mfem/fem/intrules.cpp)
                        int j = 0;
                        while(sums[j] == k){
                            beta[j++] = 0;
                            if(j == ndim){
                                ++iarr;
                                goto done_beta;
                            }
                        }
                        beta[j]++;
                        sums[j]++;
                        for(j--; j >= 0; j--){
                            sums[j] = sums[j + 1];
                        }
                    }
                done_beta:
                    ;
                }
            }

            int npoints() const override { return num_poin; }

            const GEOMETRY::Point<T, ndim> *quadraturePoints() const override { return abscisse; }

            const T *quadratureWeights() const override { return weights; }
        };

}
