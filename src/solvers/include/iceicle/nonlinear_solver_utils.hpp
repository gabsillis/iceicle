/**
 * @brief utilities for nonlinear solvers
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include <functional>

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

} 



