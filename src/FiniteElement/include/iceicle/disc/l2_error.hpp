/**
 * @brief calculate the l2 error norm of a solution
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/layout_enums.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <cmath>
#include <cstddef>
#include <iceicle/fespace/fespace.hpp>
#include <functional>

namespace DISC {

    /**
     * @tparam T the floating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam uLayoutPolicy the layout policy for the finite element soution 
     * @tparam uAccessorPolicy the accessor policy for the finite element solution 
     *
     * @param exact_sol the exact solution to compare to
     *   f(x, out)
     *   where:
     *   x - the physical domain coordinate [size = ndim]
     *   out - the value at x [size = ncomp]
     *
     * @param fespace the finite element space
     * @param coord the node coordinate array 
     * @param fedata the finite element solution coefficients
     */
    template<
        class T,
        class IDX,
        int ndim,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    T l2_error(
        std::function<void(T*, T*)> exact_sol,
        FE::FESpace<T, IDX, ndim> &fespace,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> &fedata
    ) {
        using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;
       
        auto coord = fespace.meshptr->nodes;
        std::vector<T> l2_eq(fedata.get_ncomp(), 0.0);
        // reserve data
        std::vector<T> feval(fedata.get_ncomp());
        std::vector<T> u(fedata.get_ncomp());
        std::vector<T> bi_data(fespace.dg_offsets.max_el_size_reqirement(1));

        // loop over quadrature points
        for(const Element &el : fespace.elements) {
            for(int iqp = 0; iqp < el.nQP(); ++iqp) {
                // convert the quadrature point to the physical domain
                const QUADRATURE::QuadraturePoint<T, ndim> quadpt = el.getQP(iqp);
                Point phys_pt{};
                el.transform(coord, quadpt.abscisse, phys_pt);

                // calculate the jacobian determinant
                auto J = el.geo_el->Jacobian(coord, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                exact_sol(phys_pt.data(), feval.data());

                // evaluate the basis functions
                el.evalBasisQP(iqp, bi_data.data());

                // construct the solution
                std::fill(u.begin(), u.end(), 0.0);
                for(std::size_t ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    for(std::size_t iv = 0; iv < fedata.get_ncomp(); ++iv){
                        u[iv] += bi_data[ibasis] * fedata[FE::fe_index{(std::size_t) el.elidx, ibasis, iv}];
                    }
                }

                // add the contribution of the squared error
                for(std::size_t ieq = 0; ieq < fedata.get_ncomp(); ieq++){
                    l2_eq[ieq] += std::pow(u[ieq] - feval[ieq], 2) * quadpt.weight * detJ;
                }
            }
        }

        T l2_sum = 0;
        for(int ieq = 0; ieq < fedata.get_ncomp(); ++ieq){ l2_sum += std::sqrt(l2_eq[ieq]); }
        return l2_sum;
    }
}
