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
namespace SOLVERS {
    
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
        ElementLinearSolver(const ELEMENT::FiniteElement<T, IDX, ndim> &el, FE::NodalFEFunction<T, ndim> &node_coords) 
        : mass(el.nbasis(), el.nbasis()), pi{} {
            // calculate and decompose the mass matrix
            mass = 0.0; // fill with zeros
            
            for(int ig = 0; ig < el.nQP(); ++ig){
                const QUADRATURE::QuadraturePoint<T, ndim> quadpt = el.getQP(ig);

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
        void solve(FE::ElementData<T, neq> &u, FE::ElementData<T, neq> &b){
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
        template<
            class LayoutPolicy_u, 
            class LayoutPolicy_res
        >
        void solve(
            FE::elspan<T, LayoutPolicy_u> &u, 
            const FE::elspan<T, LayoutPolicy_res> &res
        ){
            MATH::MATRIX::SOLVERS::sub_lu(mass, pi, res.data(), u.data());
        }
    };
}
