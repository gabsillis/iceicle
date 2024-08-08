/**
 * @file element_linear_solve.hpp
 * @brief solve a Linear form on an individual element
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
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
        ElementLinearSolver(const FiniteElement<T, IDX, ndim> &el, NodeArray<T, ndim> &node_coords) 
        : mass(el.nbasis(), el.nbasis()), pi{} {
            // calculate and decompose the mass matrix
            mass = 0.0; // fill with zeros
            
            for(int ig = 0; ig < el.nQP(); ++ig){
                const QuadraturePoint<T, ndim> quadpt = el.getQP(ig);

                // calculate the jacobian determinant
                auto J = el.geo_el->Jacobian(node_coords, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // integrate Bi * Bj
                for(int ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    for(int jbasis = 0; jbasis < el.nbasis(); ++jbasis){
                        mass[ibasis][jbasis] += el.basisQP(ig, ibasis) * el.basisQP(ig, jbasis) * quadpt.weight * detJ;
                    }
                }
            }
            pi = MATH::MATRIX::SOLVERS::decompose_lu(mass);
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
