/**
 * @brief weak form for the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/matrixT.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/element/finite_element.hpp>
#include <iceicle/element/TraceSpace.hpp>
#include <vector>
namespace DISC {

    /**
     * @brief a discretization of the heat equation on one variable
     *
     * \frac{du}{dt} = \mu \Delta u
     *
     * this is a spatial-only discretization (semi-discrete)
     * this is an aggregate type where mu can be set directly
     *
     * this uses a DDG discretization
     *
     * @tparam T the floating point type 
     * @tparam IDX the index type 
     * @tparam ndim the number of spatial dimensions
     */
    template<typename T, typename IDX, int ndim>
    class HeatEquation final {
        using feidx = FE::compact_index;

    public:

        /// @brief the diffusion coefficient
        T mu = 0.001;

        /**
         * @brief calculate the domain integral 
         * \int \mu\frac{du}{dx}\cdot \frac{dv}{dx} d\Omega
         *
         * @tparam ULayoutPolicy the layout policy for the view of u coefficients
         * @tparam UAccessorPolicy the accessor policy for the view of u coefficients 
         * @tparam ResLayoutPolicy the layout policy for the view of the residual
         *
         * @param [in] el the element to perform the integration over 
         * @param [in] coord the global node coordinates array
         * @param [in] u the current solution coefficient set 
         * @param [out] res the residuals for each basis function
         *              WARNING: must be zeroed out
         */
        template<class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void domainIntegral(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::elspan<T, ULayoutPolicy, UAccessorPolicy> &u,
            FE::elspan<T, ResLayoutPolicy> &res
        ) const {

            // loop over the quadrature points
            for(int iqp = 0; iqp < el.nQP(); ++iqp){
                const QUADRATURE::QuadraturePoint<T, ndim> &quadpt = el.getQP(iqp);

                // calculate the jacobian determinant 
                auto J = el.geo_el->Jacobian(coord, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // get the gradients in the physical domain
                std::vector<T> grad_data(el.nbasis() * ndim);
                auto gradBi = el.evalPhysGradBasisQP(iqp, coord, J, grad_data.data());

                // construct the gradient of u
                T gradu[ndim] = {0};
                u.contract_mdspan(gradBi, gradu);

                // loop over the test functions and construct the residual 
                for(std::size_t itest = 0; itest < el.nbasis(); ++itest){
                    for(std::size_t jdim = 0; jdim < ndim; ++jdim){
                        res[feidx{.idof = itest, .iv = 0}]
                            -= mu * gradu[jdim] * gradBi[itest, jdim] * detJ * quadpt.weight;
                    }
                }
            }
        }


        template<class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void traceIntegral(
            const ELEMENT::TraceSpace<T, IDX, ndim> &trace,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::elspan<T, ULayoutPolicy, UAccessorPolicy> &uL,
            FE::elspan<T, ULayoutPolicy, UAccessorPolicy> &uR,
            FE::elspan<T, ResLayoutPolicy> &resL,
            FE::elspan<T, ResLayoutPolicy> &resR
        ) const {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;

            // calculate the centroids of the left and right elements
            // in the physical domain
            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;
            auto centroidL = elL.geo_el->centroid(coord);
            auto centroidR = elR.geo_el->centroid(coord);

            // loop over the quadrature points 
            for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                // calculate the riemannian metric tensor root
                // TODO: could maybe reuse the left and righte jacobians because
                // these are needed for the gradients and hessians in the physical domn 
                // but we also need Jfac for unit_normal

                auto Jfac = trace.face.Jacobian(coord, quadpt.abscisse);
                T sqrtg = trace.face.rootRiemannMetric(Jfac, quadpt.abscisse);

                // get the function values
                std::vector<T> bi_dataL(elL.nbasis());
                std::vector<T> bi_dataR(elR.nbasis());
                trace.evalGradBasisQPL(iqp, bi_dataL.data());
                trace.evalGradBasisQPR(iqp, bi_dataR.data());

                T value_uL = 0.0, value_uR = 0.0;
                for(std::size_t ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                { value_uL += uL[feidx{.idof = ibasis, .iv = 0}] * bi_dataL[ibasis]; }
                for(std::size_t ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                { value_uR += uR[feidx{.idof = ibasis, .iv = 0}] * bi_dataR[ibasis]; }

                // get the gradients the physical domain
                std::vector<T> grad_dataL(elL.nbasis() * ndim);
                std::vector<T> grad_dataR(elR.nbasis() * ndim);
                auto gradBiL = trace.evalGradBasisQPL(iqp, grad_dataL.data());
                auto gradBiR = trace.evalGradBasisQPR(iqp, grad_dataR.data());

                std::vector<T> gradu_dataL(ndim);
                std::vector<T> gradu_dataR(ndim);
                auto graduL = uL.contract_mdspan(gradBiL, gradu_dataL.data());
                auto graduR = uR.contract_mdspan(gradBiR, gradu_dataR.data());

                // get the hessians in the physical domain 
                std::vector<T> hess_dataL(elL.nbasis() * ndim * ndim);
                std::vector<T> hess_dataR(elR.nbasis() * ndim * ndim);
                auto hessBiL = trace.evalHessBasisQPL(iqp, hess_dataL.data());
                auto hessBiR = trace.evalHessBasisQPR(iqp, hess_dataR.data());

                std::vector<T> hessu_dataL(ndim * ndim);
                std::vector<T> hessu_dataR(ndim * ndim);
                auto hessuL = uL.contract_mdspan(hessBiL, hessu_dataL.data());
                auto hessuR = uR.contract_mdspan(hessBiR, hessu_dataR.data());

                // calculate the normal vector 
                auto unit_normal = normalize(calc_ortho(Jfac));

                // calculate the DDG distance
                T h_ddg = 0;
                for(int idim = 0; idim < ndim; ++idim){
                    h_ddg += unit_normal[idim] * (
                        std::abs(quadpt.abscisse[idim] - centroidL[idim])
                        + std::abs(quadpt.abscisse[idim] - centroidR[idim])
                    );
                }

                // construct the DDG derivatives
                T grad_ddg[ndim];
                int max_basis_order = std::min(
                    elL.basis->getPolynomialOrder(),
                    elR.basis->getPolynomialOrder()
                );
                // Danis and Yan reccomended for NS
                T beta0 = std::pow(max_basis_order + 1, 2);
                T beta1 = 1 / (T) (2 * max_basis_order * (max_basis_order * 2));
                T jumpu = value_uR - value_uL;
                for(int idim = 0; idim < ndim; ++idim){
                    grad_ddg[idim] = beta0 * jumpu / h_ddg * unit_normal[idim]
                        + 0.5 * (graduL[idim] + graduR[idim]);
                    T hessTerm = 0;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        hessTerm += (hessuR[jdim][idim] - hessuL[jdim][idim])
                            * unit_normal[jdim];
                    }
                    grad_ddg[idim] += beta1 * h_ddg * hessTerm;
                }

                T flux = mu * MATH::MATRIX_T::dotprod<T, ndim>(grad_ddg, unit_normal);
                // contribution to the residual 
                for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                    resL[feidx{.idof = itest, .iv = 0}] -= flux * bi_dataL[itest];
                }
                for(std::size_t itest = 0; itest < elR.nbasis(); ++itest){
                    resL[feidx{.idof = itest, .iv = 0}] += flux * bi_dataR[itest];
                }
            }

        }
    };
}
