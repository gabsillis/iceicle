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
namespace DISC {
    
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
        Projection(ProjectionFunction<T, ndim, neq> func) 
        : func{func} {}

        /**
         * @brief Integral over the element domains formed by 
         *        the weak form of u = f(x)
         *        /int f(x) v dx
         * @param el the element
         * @param node_coords the node coordinates array
         * @param res the residual function (WARNING: MUST BE ZEROED OUT)
         */
        void domainIntegral(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            FE::NodalFEFunction<T, ndim> &node_coords,
            FE::ElementData<T, neq> &res
        ) {
            for(int ig = 0; ig < el.nQP(); ++ig){ // loop over the quadrature points
                
                // convert the quadrature point to the physical domain
                const QUADRATURE::QuadraturePoint<T, ndim> quadpt = el.getQP(ig);
                Point phys_pt{};
                el.transform(node_coords, quadpt.abscisse, phys_pt);

                // calculate the jacobian determinant
                auto J = el.geo_el->Jacobian(node_coords, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                T feval[neq];
                func(phys_pt, feval);

                // loop over the equations and test functions and construct the residual
                for(int eq = 0; eq < neq; eq++){
                    for(int b = 0; b < el.nbasis(); b++){
                        res.getValue(eq, b) += feval[eq] * quadpt.weight * el.basisQP(ig, b) * detJ;
                    }          
                }
            }
        }

        template<class LayoutPolicy, class AccessorPolicy>
        void domainIntegral(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            FE::NodalFEFunction<T, ndim> &node_coords,
            FE::elspan<T, LayoutPolicy, AccessorPolicy> &res
        ) {
            T detJ; // TODO: put back in loop after debuggin for clarity
            for(int ig = 0; ig < el.nQP(); ++ig){ // loop over the quadrature points
                
                // convert the quadrature point to the physical domain
                const QUADRATURE::QuadraturePoint<T, ndim> quadpt = el.getQP(ig);
                Point phys_pt{};
                el.transform(node_coords, quadpt.abscisse, phys_pt);

                // calculate the jacobian determinant
                auto J = el.geo_el->Jacobian(node_coords, quadpt.abscisse);
                detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                T feval[neq];
                func(phys_pt, feval);

                // loop over the equations and test functions and construct the residual
                for(std::size_t b = 0; b < el.nbasis(); b++){
                    for(std::size_t eq = 0; eq < neq; eq++){
                        res[b, eq] += feval[eq] * quadpt.weight * el.basisQP(ig, b) * detJ;
                    }          
                }
            }
            
        }
    };
}
