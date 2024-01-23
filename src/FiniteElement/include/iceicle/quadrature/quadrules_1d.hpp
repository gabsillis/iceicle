/**
 * @file quadrules_1d.hpp 
 * @brief 1D quadrature rules which can be extended 
 * to higher dimensions by tensor product techniques
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <Numtool/MathUtils.hpp>
#include <Numtool/polydefs/LegendrePoly.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <Numtool/constexpr_math.hpp>

#include <iceicle/quadrature/QuadratureRule.hpp>
namespace QUADRATURE {


    /**
     * @brief 
     */
    template<typename T, typename IDX, int npoin>
    class GaussLegendreQuadrature final : public QuadratureRule<double, int, 1> {
        
        /// d is the maximum polynomial degree of the integration formula
        static constexpr int d = 2 * npoin + 1;

        /// this is a one dimensional quadrature rule 
        static constexpr int ndim = 1;

        /// The Quadrature Points
        QuadraturePoint<T, ndim> qpoints[npoin];

        public:

        GaussLegendreQuadrature(){

            // rules are symmetric, only do half
            static constexpr int m = (npoin) / 2;

            T weightcenter;
            if constexpr (npoin % 2 != 0) weightcenter = 2.0;
            for(int i = 1; i <= m; i++){
                // populate initial guess 
                T xi = std::cos(M_PI * (i - 0.25) / (npoin + 0.5));
               
                T dlegendre = 0.0;
                // Newton-Raphson loop
                static constexpr int max_newton_it = 50;
                for(int inewton = 0; inewton < max_newton_it; ++inewton){
                    
                    T legendre_eval = MATH::POLYNOMIAL::legendre1d<T, npoin>(xi);
                    dlegendre = MATH::POLYNOMIAL::dlegendre1d<T, npoin>(xi);

                    T dxi = legendre_eval / dlegendre;
                    if(std::abs(dxi) < 1e-16){
                        xi -= dxi;
                        break;
                    }

                    xi -= dxi;
                }

                T weight = 2.0 / ((1 - SQUARED(xi)) * SQUARED(dlegendre));
                if constexpr (npoin % 2 != 0) weightcenter -=  2 * weight;
                int iarr = i - 1;
                qpoints[2 * iarr].abscisse = {xi};
                qpoints[2 * iarr].weight = weight;

                qpoints[2 * iarr + 1].abscisse = {-xi};
                qpoints[2 * iarr + 1].weight = weight;
                
            }

            if constexpr(npoin % 2 != 0){
                qpoints[npoin - 1].abscisse = {0.0};
                qpoints[npoin - 1].weight = weightcenter;
            }
        }

        int npoints() const override { return npoin; }

        const QuadraturePoint<T, ndim> &getPoint(int ipoint) const override { return qpoints[ipoint]; }
    };

    // Fast and Accurate Computation of Gauss-Legendre and Gauss-Jacobi Quadrature Nodes and Weights 
    // N. Hale, A. Townsend 
    // SIAM Journal on Scientific Computing vol 35 issue 2 Jan 2013
//    template<typename T, typename IDX, int npoin>
//    class GaussLegendreQuadrature final : public QuadratureRule<T, IDX, 1>, TraceQuadratureRule<T, IDX, 1> {
//
//        /* Approximations of bessel function roots near boundaries */
//        static constexpr double besselroots[32] = {
//            2.404825557695773,  5.520078110286311,
//            8.653727912911013, 11.791534439014280,
//            14.930917708487790, 18.071063967910920,
//            21.211636629879269, 24.352471530749284,
//            27.493479132040250, 30.634606468431990,
//            33.775820213573560, 36.917098353664045,
//            40.058425764628240, 43.199791713176737,
//            46.341188371661815, 49.482609897397822,
//            52.624051841114984, 55.765510755019974,
//            58.906983926080954, 62.048469190227159,
//            65.189964800206866, 68.331469329856787,
//            71.472981603593752, 74.614500643701817,
//            77.756025630388066, 80.897555871137627,
//            84.039090776938195, 87.180629843641128,
//            90.322172637210500, 93.463718781944763,
//            96.605267950996279, 99.746819858680624
//        };
//
//        /// @brief compute the constant in front of the sum in the evaluation of the 
//        ///        asymptotic formula for the Legendre polynomials
//        inline T getC(){
//            T dS, S, fn, fn5, nk, n5k, flt_npoin;
//            flt_npoin = static_cast<T>(npoin);
//
//            /* C = sqrt(pi/4)*gamma(n+1)/gamma(n+3/2) */
//            /* Use Stirling's series */
//            dS = -.125/flt_npoin;
//            S = dS; 
//            int k = 1;
//            while ( std::abs(dS/S) > std::numeric_limits<T>::epsilon()/100.0 ) {
//                k += 1;
//                dS *= -.5 * (k-1) / (k+1) / flt_npoin;
//                S += dS;
//            }
//            double stirling[10] = {
//                1., 1./12., 1./288., -139./51840., -571./2488320., 
//                163879./209018880., 5246819./75246796800., 
//                -534703531./902961561600., -4483131259./86684309913600., 
//                432261921612371./514904800886784000.
//            };
//
//            fn = 1; fn5 = 1; nk = flt_npoin; n5k = flt_npoin+.5;
//            for ( k = 1 ; k < 10 ; k++ ) {
//                fn += stirling[k]/nk;
//                fn5 += stirling[k]/n5k;
//                nk *= flt_npoin;
//                n5k *= flt_npoin + .5;
//            }
//            return std::exp(S)*std::sqrt(4.0/(flt_npoin+.5)/M_PI) * fn / fn5;  
//        }
//
//        /// @brief get the guess for the root
//        inline T theta_k_guess(int k){
//
//            // asymptotic formula eq (3.5) [Tricomi, 1950]
//            T phi = M_PI * (4.0*k - 1)/ (4.0*npoin+2.0);
//            T sinphi = std::sin(phi);
//            T x = (
//                1.0 - (npoin - 1.0) / (8.0*std::pow(npoin, 3)) 
//                -(39.0 - 28.0/SQUARED(sinphi)) / (384.0*std::pow(npoin, 4))
//            ) * cos(phi);
//            T theta = std::acos(x);
//
//            // asymptotic is less accurate near the ends 
//            // use a different guess 
//            if(x > 0.5){
//                T jk;
//
//                // bessel function roots 
//                if(k < 30){
//                    // these roots are hard coded
//                    jk = besselroots[k - 1];
//                } else {
//                    // compute more (Branders JCP 1981)
//                    T p = (k - 0.25)*M_PI;
//                    T pp = SQUARED(p);
//                    T num = 0.0682894897349453 + pp*(0.131420807470708 + pp*(0.0245988241803681 + pp*0.000813005721543268));
//                    T den = p*(1.0 + pp*(1.16837242570470 + pp*(0.200991122197811 + pp*(0.00650404577261471))));
//                    jk = p + num/den;
//                }
//
//                // evaluate asymptotic approximation of theta 
//                if(k <= 5){
//                    // Extreme boundary (Gatteschi, 1967)
//                    T pn = std::pow(npoin, 3) + 1.0/3.0;
//                    theta = jk/std::sqrt(pn) * (1.0 - (SQUARED(jk) - 1.0)/SQUARED(pn)/360.0);
//                } else {
//                    // Boundary (Olver, 1974)
//                    T p = jk/(npoin + 0.5);
//                    theta = p + (p * std::cos(p) / std::sin(p)-1.0) / (8.0*p*SQUARED(npoin + 0.5));
//                }
//            }
//            return theta;
//        }
//
//        /** @brief d/dtheta P_n(cos(theta)) where x_k = cos(theta_k) */
//        inline T dthetaPn(T costheta){
//            using namespace MATH::POLYNOMIAL;
//            return ( // note: signs have been flipped because div by -sin(theta)
//                npoin*costheta*legendre1d<T, npoin>(costheta)
//                - npoin * legendre1d<T, npoin - 1>(costheta)
//            ) / sin(std::acos(costheta));
//        }
//
//        struct FuncAndDerivativeEval{
//            T function_value;
//            T derivative_value;
//        };
//
//        inline FuncAndDerivativeEval evalPolyAndDeriv(int k, T C, T theta){
//            // TODO: add boundary formula with bessel function evaluations 
//
//            // interior evaluation formula (Steiltjes, 1980)
//            T sin_theta = std::sin(theta);
//            T cos_theta = std::cos(theta);
//            T cot_theta = cos_theta / sin_theta;
//            T denom = std::sqrt(2.0 * sin_theta);
//
//            T alpha = (npoin + 0.5) * theta - 0.25 * M_PI;
//            T cosA = cos(alpha);
//            T sinA = sin(alpha);
//
//            T f = C * cosA / denom;
//            T fprime = C * (0.5 * (cosA*cot_theta+sinA) + npoin * sinA) / denom;
//
//            static constexpr int niter_max = 50; // original: 30;
//            for( int m = 1; m <= niter_max; ++m) {
//                C *= (1.0 - 0.5/m)*(m - 0.5)/(npoin + m + 0.5);
//                denom *= 2.0 * sin_theta;
//        
//                alpha += theta - 0.5*M_PI;
//                cosA = std::cos(alpha);
//                sinA = std::sin(alpha);
//
//                T df = C * cosA/denom;
//                T dfp = C * ( (m+0.5)*(cosA*cot_theta+sinA) + npoin*sinA ) / denom;
//
//                f += df;
//                fprime += dfp;
//
//                if(std::abs(df) + std::abs(dfp) < std::numeric_limits<T>::epsilon() / 100.0) break;
//            }
//
//            FuncAndDerivativeEval ret = {.function_value = f, .derivative_value = fprime};
//            return ret;
//        }
//        
//
//        /// d is the maximum polynomial degree of the integration formula
//        static constexpr int d = 2 * npoin + 1;
//
//        /// this is a one dimensional quadrature rule 
//        static constexpr int ndim = 1;
//
//        /// The Quadrature Points
//        QuadraturePoint<T, ndim> qpoints[npoin];
//
//        public:
//
//        GaussLegendreQuadrature(){
//            // get the constant in front of polynomial evaluation
//            T C = getC();
//
//            T weightFinal = 1.0; // for odd number
//            static constexpr int isEven = npoin % 2;
//            for(int k = (npoin + isEven) / 2; k > 0; k--){
//                T theta = theta_k_guess(k);
//
//
//                // Newton iterations 
//                static constexpr int niter_max = 1000;
//                for(int inewton = 0; inewton < niter_max; ++inewton){
//                    auto result = evalPolyAndDeriv(k, C, theta);
//                    // newton step
//                    T dtheta = result.function_value / result.derivative_value;
//                    theta += dtheta;
//
//                    // stop condition
//                    if(std::abs(dtheta) > std::sqrt(std::numeric_limits<T>:: epsilon())/100.0 ) break;
//                }
//
//                // get the final evaluations 
//                auto result = evalPolyAndDeriv(k, C, theta);
//
//                // convert back to x space 
//                T x = std::cos(theta);
//                T weight = 2.0 / (SQUARED(result.derivative_value));
//
//                // assign quadrature
//                qpoints[npoin - k].abscisse = x;
//                qpoints[npoin - k].weight = weight;
//                qpoints[k - 1].abscisse = -x;
//                qpoints[k - 1].weight = weight;
//
//                if constexpr(!isEven){
//                    weightFinal -= 2 * weight;
//                }
//            }
//
//            if constexpr(!isEven){
//                qpoints[(npoin+isEven)/2 - 1].abscisse = 0.0;
//                qpoints[(npoin+isEven)/2 - 1].weight = weightFinal;
//            }
//
//        }
//
//        int npoints() const override { return npoin; }
//
//        const QuadraturePoint<T, ndim> &getPoint(int ipoint) const override { return qpoints[ipoint]; }
//    };
}
