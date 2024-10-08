/**
 * @file projection.hpp
 * @brief weak form projection onto the finite element space
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include <functional>
#include <iceicle/element/finite_element.hpp>
#include <iceicle/fe_function/fe_function.hpp>
#include <Numtool/matrixT.hpp>
namespace iceicle {
    
    /**
     * @brief function to project to
     * takes in a point in the physical domain as the first argument
     * and returns neq values in the second argument
     * @tparam T the floating point type
     * @tparam ndim the number of dimensions
     * @tparam neq tne number of equations that it gets evaluated to
     */
    template<typename T, int ndim, int neq>
    using ProjectionFunction = std::function<void(const T[ndim], T[neq])>;


    template<typename T, typename IDX, int ndim, int neq>
    class Projection {
        using Point = MATH::GEOMETRY::Point<T, ndim>;
    
        ProjectionFunction<T, ndim , neq> func;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = neq;

        Projection(ProjectionFunction<T, ndim, neq> func) 
        : func{func} {}

        /**
         * @brief Integral over the element domains formed by 
         *        the weak form of u = f(x)
         *        (f(x), v)
         * @param el the element
         * @param res the residual function (WARNING: MUST BE ZEROED OUT)
         */
        void domain_integral(
            const FiniteElement<T, IDX, ndim> &el,
            ElementData<T, neq> &res
        ) {
            for(int ig = 0; ig < el.nQP(); ++ig){ // loop over the quadrature points
                
                // convert the quadrature point to the physical domain
                const QuadraturePoint<T, ndim> quadpt = el.getQP(ig);
                Point phys_pt = el.transform(quadpt.abscisse);

                // calculate the jacobian determinant
                auto J = el.jacobian(quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                T feval[neq];
                func(phys_pt, feval);

                // loop over the equations and test functions and construct the residual
                for(int eq = 0; eq < neq; eq++){
                    for(int b = 0; b < el.nbasis(); b++){
                        res.getValue(eq, b) += feval[eq] * quadpt.weight * el.basis_qp(ig, b) * detJ;
                    }          
                }
            }
        }

        /**
         * @brief Integral over the element domains formed by 
         *        the weak form of u = f(x)
         *        /int f(x) v dx
         * @param el the element
         * @param res the residual function (WARNING: MUST BE ZEROED OUT)
         */
        void domain_integral(
            const FiniteElement<T, IDX, ndim> &el,
            elspan auto res
        ) {
            T detJ; // TODO: put back in loop after debuggin for clarity
            for(int ig = 0; ig < el.nQP(); ++ig){ // loop over the quadrature points
                
                // convert the quadrature point to the physical domain
                const QuadraturePoint<T, ndim> quadpt = el.getQP(ig);
                Point phys_pt = el.transform(quadpt.abscisse);

                // calculate the jacobian determinant
                auto J = el.jacobian(quadpt.abscisse);
                detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                T feval[neq];
                func(phys_pt, feval);

                // loop over the equations and test functions and construct the residual
                for(std::size_t b = 0; b < el.nbasis(); b++){
                    for(std::size_t eq = 0; eq < neq; eq++){
                        res[b, eq] += feval[eq] * quadpt.weight * el.basis_qp(ig, b) * detJ;
                    }          
                }
            }
            
        }
    };
}
