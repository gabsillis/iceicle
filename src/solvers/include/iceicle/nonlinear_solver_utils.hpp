/**
 * @brief utilities for nonlinear solvers
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include <functional>
#include <variant>
#include <cmath>

namespace ICEICLE::SOLVERS {

    /**
     * @brief Criteria for convergence of a nonlinear solve 
     * based on the residual norm passed to the callback 
     * this determines whether or not the solver should terminate
     * due to convergence 
     *
     * @tparam T the data type
     * @tparam IDX the index type
     */
    template<class T, class IDX>
    struct ConvergenceCriteria {

        /// @brief the absolute tolerance for the residual
        T tau_abs = 1e-8;

        /// @brief the relative tolerance for the residual
        T tau_rel = 1e-8;

        /// @brief the maxmimum number of nonlinear iterations
        IDX kmax = 100;

        /// @brief the callback to determine if the solver is converged 
        /// takes the residual norm as the argument
        /// returns true if the solution is converged 
        /// false otherwise
        ///
        /// The default implementation takes the 
        /// absolute tolerance + initial residual times relative tolerance 
        /// and is connverged if the residual norm is less than that
        std::function<bool(T)> done_callback = [&](T res_norm) -> bool {
            T tau = tau_abs + tau_rel * r0;
            return res_norm <= tau;
        };

        /// @brief the initial residual 
        /// Generally set by the solver
        T r0 = 0.0;
        
        /**
         * @brief call the convergence callback function 
         * @param res_norm the residual norm 
         * @return true if the convergence criteria is met
         */
        bool operator()(T res_norm){
            return done_callback();
        }
    };

    // ========================
    // = Linesearch Utilities =
    // ========================
    
    enum class LINESEARCH_TYPE {
        NONE,         /// @brief no linesearch (take the full step)
        WOLFE_CUBIC   /// @cubic linesearch based on Wolfe Conditions
    };

    template<typename T>
    using function1d = std::function<T(T)>;


    template<typename T>
    inline T finite_difference(function1d<T> fcn, T EPSILON, T x){
        T phi1 = fcn(x);
        T phi2 = fcn(x + EPSILON);
        return (phi2 - phi1) / EPSILON;
    }

    template<typename T>
    T zoom(function1d<T> fcn, T alpha_lo, T alpha_hi, int kmax, T c1, T c2){
        static constexpr T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());

        for(int k = 0; k < kmax; k++){
            // cubic interpolation
            T phi_alo = fcn(alpha_lo);
            T dphi_alo = finite_difference(fcn, EPSILON, alpha_lo);
            T phi_ahi = fcn(alpha_hi);
            T dphi_ahi = finite_difference(fcn, EPSILON, alpha_hi);

            T d1 = dphi_alo + dphi_ahi - 3 * (phi_alo - phi_ahi) / (alpha_lo - alpha_hi);
            T d2 = std::sqrt(std::max(SQUARED(d1) - dphi_alo * dphi_ahi, 1e-8)); // safeguard squareroot
            d2 = std::copysign(d2, (alpha_hi - alpha_lo));

            T aj = alpha_hi - (alpha_hi - alpha_lo) * (
                (dphi_ahi + d2 - d1) / 
                (dphi_ahi - dphi_alo + 2 * d2)
            );

            T phi_aj = fcn(aj);
            T dphi_0 = finite_difference(fcn, EPSILON, 0.0);
            if(
                phi_aj > (fcn(0) + c1 * aj * dphi_0)
                || phi_aj >= phi_alo
            ){
                alpha_hi = aj;
            } else {
                T dphi_aj = finite_difference(fcn, EPSILON, aj);
                if( std::abs(dphi_aj <= -c2 * dphi_0) ){
                    return aj;
                }

                if(dphi_aj * (alpha_hi - alpha_lo) >= 0){
                    alpha_hi = alpha_lo;
                }
                alpha_lo = aj;
            }
        }

        // if zoom doesn't finish, average the remaining range
        return 0.5 * (alpha_lo + alpha_hi);
    }

    /**
     * @brief Perform a cubic linesearch based on the Wolfe Conditions
     * see Nocedal, Wright
     * 
     * @tparam T the floating point type
     * @param fcn the 1d function in terms of alpha
     * @param alpha_max the maximum alpha
     * @param alpha1 the initial alpha (reccomend 1 for Newton or scaled for CG)
     * @param kmax the maximum iterations for the linesearch
     * @param c1 the first linesearch constant (reccommend 1e-4)
     * @param c2 the second linesearch constant (reccomend 0.9)
     * @return T the step length alpha that provides sufficient decrease based on the Wolfe Conditions
     */
    template<typename T>
    T wolfe_ls(function1d<T> fcn, T alpha_max, T alpha1, int kmax, T c1, T c2){
        static T EPSILON = std::sqrt(std::numeric_limits<T>::epsilon());
        T a_im1 = 0;
        T a_i = alpha1;
        T phi_0 = fcn(0);
        T dphi_0 = finite_difference(fcn, EPSILON, 0.0);
        for(int k = 0; k < kmax; k++){
            T phi_i = fcn(a_i);
            T phi_im1 = fcn(a_im1);
            if(
                phi_i > phi_0 + c1 * a_i * dphi_0
                || (phi_i >= phi_im1 && k > 0)
            ){
                return zoom(fcn, a_im1, a_i, kmax, c1, c2);
            }
            T dphi_i = finite_difference(fcn, EPSILON, a_i);

            if(std::abs(dphi_i) <= -c2 * dphi_0) {
                return a_i;
            }

            if(dphi_i >= 0){
                return zoom(fcn, a_i, a_im1, kmax, c1, c2);
            }

            T temp = 0.5 * (a_i + alpha_max); // compute the new a_i as average of current a_i and alpha_max
            a_im1 = a_i;
            a_i = temp;
        }
        return a_i;
    }

    /**
     * @brief no linesearch strategy - take the full step 
     */
    template<typename T, typename IDX>
    struct no_linesearch {
        auto operator()(function1d<T> fcn) -> T { return 1.0; }
    };

    /**
     * @brief linesearch strategy that performs a cubic linesearch based on 
     * the Wolfe Conditions 
     * see Nocedal and Wright 
     */
    template<typename T, typename IDX>
    struct wolfe_linesearch {

        using value_type = T;
        using index_type = IDX;

        /// @brief maximum number of iterations for linesearch
        index_type max_it = 20;

        /// @brief the initial linesearch multiplier 
        /// (reccomend 1.0 for Newton or scaled for CG)
        value_type alpha_initial = 1;

        ///@brief the maximum linesurch multiplier
        value_type alpha_max = 10;

        /// @brief the first linesearch constant (reccomend 1e-4)
        value_type c1 = 1e-4;

        /// @brief the second linesearch constant (reccomend 0.9)
        value_type c2 = 0.9;

        auto operator()(function1d<T> fcn) -> T {
            return wolfe_ls(fcn, alpha_max, alpha_initial, max_it, c1, c2);
        }

    };

    template<class T, class IDX>
    using LinesearchVariant = std::variant<no_linesearch<T, IDX>, wolfe_linesearch<T, IDX>>;

} 



