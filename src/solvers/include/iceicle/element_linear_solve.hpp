/**
 * @file element_linear_solve.hpp
 * @brief solve a Linear form on an individual element
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "Numtool/matrix/permutation_matrix.hpp"
#include "iceicle/anomaly_log.hpp"
#include <iceicle/element/finite_element.hpp>
#include <iceicle/fe_function/fe_function.hpp>
#include <iceicle/fe_function/fespan.hpp>
#include <Numtool/matrix/dense_matrix.hpp>
#include <Numtool/matrix/decomposition/decomp_lu.hpp>
namespace iceicle::solvers {
    
    /**
     * Solver for a linear form on an individual element
     * 
     * Once set up for a specific element
     */
    template<typename T, typename IDX, int ndim, int neq>
    class ElementLinearSolver {
        private:

        MATH::MATRIX::DenseMatrix<T> mass; /// the mass matrix (decomposed) (not including jacobian)
        MATH::MATRIX::PermutationMatrix<unsigned int> pi; // permutation matrix for decomposition

        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:

        /**
         * @brief constructor
         * @param el the element to make the solver for
         * @param node_coords the global node coordinates array
         */
        ElementLinearSolver(const FiniteElement<T, IDX, ndim> &el)
        : mass{calculate_mass_matrix(el)}, pi{} {

            // decompse the mass matrix
            try {
                pi = MATH::MATRIX::SOLVERS::decompose_lu(mass);
            } catch(MATH::MATRIX::SOLVERS::SingularMatrixException e){
                // set to Identity on failure and log anomaly
                pi = MATH::MATRIX::PermutationMatrix<unsigned int>{(unsigned int) el.nbasis()};
                mass = 0;
                for(int i = 0; i < el.nbasis(); ++i){
                    mass[i][i] = 1.0;
                }
                util::AnomalyLog::log_anomaly(util::Anomaly{"Singular Mass Matrix encountered on element " + std::to_string(el.elidx), util::general_anomaly_tag{}});
            }
        }

        /**
         * @brief solve Mu = b where M is the mass matrix of the element and b is the 
         *        linear form
         * @param [out] u the solution of Mu = b
         * @param [in] b the residual
         */
        void solve(ElementData<T, neq> &u, ElementData<T, neq> &b){
            MATH::MATRIX::SOLVERS::sub_lu(mass, pi, b.getData(), u.getData());
        }

        
        /**
         * @brief solve Mu = b where M is the mass matrix of the element and b is the 
         *        linear form
         * NOTE: This only works for elspans with the default accessor
         * 
         * @param [out] u the solution of Mu = res
         * @param [in] res the residual
         */
        void solve(
            elspan auto u, 
            const elspan auto res
        ){
            for(int ieq = 0; ieq < decltype(u)::static_extent(); ++ieq){
                std::vector<T> ueq(u.ndof());
                std::vector<T> reseq(u.ndof());

                for(int idof = 0; idof < u.ndof(); ++idof){
                    reseq[idof] = res[idof, ieq];
                }
                MATH::MATRIX::SOLVERS::sub_lu(mass, pi, reseq.data(), ueq.data());
                for(int idof = 0; idof < u.ndof(); ++idof){
                    u[idof, ieq] = ueq[idof];
                }
            }
        }
    };
}
