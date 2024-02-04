/**
 * @brief weak form for the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/matrixT.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/element/finite_element.hpp>
#include <iceicle/element/TraceSpace.hpp>
#include <iostream>
#include <vector>
namespace DISC {

    /**
     * @brief a discretization of the heat equation on one variable
     *
     * \frac{du}{dt} = \mu \Delta u
     *
     * this is a spatial-only discretization (semi-discrete)
     *
     * the diffusion coefficient mu is publically accessible
     * as well as boundary condition lists
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

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;

        /// @brief the dynamic number of vector components
        static const int dnv_comp = 1;

        /// @brief the diffusion coefficient
        T mu = 0.001;

        /// @brief the interior penalty coefficient for 
        /// Dirichlet boundary conditions 
        // see Arnold et al 2002 sec 2.1
        // also: https://mooseframework.inl.gov/source/bcs/PenaltyDirichletBC.html
        // must be > 1.0
        T penalty = 1.001;

        /// @brief the value for each bcflag (index into this list) 
        /// for dirichlet bc
        std::vector<T> dirichlet_values;

        /// @brief the prescribed normal gradient for each bcflag 
        /// (index into this list) for neumann bc
        std::vector<T> neumann_values;

        /**
         * @brief get the timestep from cfl 
         * often this will require data to be set from the domain and boundary integrals 
         * such as wavespeeds, which will arise naturally during residual computation
         * (WARNING: except for the very first iteration)
         * WARNING: does not consider polynomial order of basis functions
         *
         * @param cfl the cfl condition 
         * @param reference_length the size to use for the length of the cfl condition 
         * @return the timestep based on the cfl condition
         */
        T dt_from_cfl(T cfl, T reference_length){
            return SQUARED(reference_length) / mu * cfl;
        }

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
                trace.evalBasisQPL(iqp, bi_dataL.data());
                trace.evalBasisQPR(iqp, bi_dataR.data());

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
                auto normal = calc_ortho(Jfac);
                auto unit_normal = normalize(normal);

                // calculate the DDG distance
                T h_ddg = 0;
                for(int idim = 0; idim < ndim; ++idim){
                    h_ddg += unit_normal[idim] * (
                        std::abs(quadpt.abscisse[idim] - centroidL[idim])
                        + std::abs(quadpt.abscisse[idim] - centroidR[idim])
                    );
                }

                static constexpr int ieq = 0;
                // construct the DDG derivatives
                T grad_ddg[ndim];
                int max_basis_order = std::min(
                    elL.basis->getPolynomialOrder(),
                    elR.basis->getPolynomialOrder()
                );
                // Danis and Yan reccomended for NS
                T beta0 = std::pow(max_basis_order + 1, 2);
                T beta1 = 1 / std::max((T) (2 * max_basis_order * (max_basis_order + 1)), 1.0);
                T jumpu = value_uR - value_uL;
                for(int idim = 0; idim < ndim; ++idim){
                    grad_ddg[idim] = beta0 * jumpu / h_ddg * unit_normal[idim]
                        + 0.5 * (graduL[ieq, idim] + graduR[ieq, idim]);
                    T hessTerm = 0;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        hessTerm += (hessuR[ieq, jdim, idim] - hessuL[ieq, jdim, idim])
                            * unit_normal[jdim];
                    }
                    grad_ddg[idim] += beta1 * h_ddg * hessTerm;
                }

                // calculate the flux weighted by the quadrature and face metric
                T flux = mu * MATH::MATRIX_T::dotprod<T, ndim>(grad_ddg, unit_normal.data()) 
                    * quadpt.weight * sqrtg;

                // contribution to the residual 
                for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                    resL[feidx{.idof = itest, .iv = 0}] -= flux * bi_dataL[itest];
                }
                for(std::size_t itest = 0; itest < elR.nbasis(); ++itest){
                    resR[feidx{.idof = itest, .iv = 0}] += flux * bi_dataR[itest];
                }
            }

        }

        /**
         * @brief calculate the weak form for a boundary condition 
         *        NOTE: Left is the interior element
         *
         * @tparam ULayoutPolicy the layout policy for the element data 
         * @tparam UAccessorPolicy the accessor policy for element data 
         * @tparam ResLayoutPolicy the layout policy for the residual data 
         *         (the accessor is the default for the residual)
         *
         * @param [in] trace the trace to integrate over 
         * @param [in] coord the global node coordinates array
         * @param [in] uL the interior element basis coefficients 
         * @param [in] uR is the same as uL unless this is a periodic boundary
         *                then this is the coefficients for the periodic element 
         * @param [out] resL the residual for the interior element 
         */
        template<class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void boundaryIntegral(
            const ELEMENT::TraceSpace<T, IDX, ndim> &trace,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::elspan<T, ULayoutPolicy, UAccessorPolicy> &uL,
            FE::elspan<T, ULayoutPolicy, UAccessorPolicy> &uR,
            FE::elspan<T, ResLayoutPolicy> &resL
        ) const {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;
            const FiniteElement &elL = trace.elL;

            switch(trace.face.bctype){
                case ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET: 
                {
                    // see Arnold et al 2002 sec 2.1
                    // Interior Penalty method 

                    // loop over quadrature points
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                        const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face.Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face.rootRiemannMetric(Jfac, quadpt.abscisse);

                        // get the function values
                        std::vector<T> bi_dataL(elL.nbasis());
                        trace.evalBasisQPL(iqp, bi_dataL.data());

                        T value_uL = 0.0;
                        for(std::size_t ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                        { value_uL += uL[feidx{.idof = ibasis, .iv = 0}] * bi_dataL[ibasis]; }

                        T dirichlet_val = dirichlet_values[trace.face.bcflag];

                        for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                            resL[feidx{.idof = itest, .iv = 0}] -= 
                                penalty * (value_uL - dirichlet_val) 
                                * bi_dataL[itest] * quadpt.weight * sqrtg;
                        }
                    }
                }
                break;

                case ELEMENT::BOUNDARY_CONDITIONS::NEUMANN:
                {
                    // loop over quadrature points 
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){

                        const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face.Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face.rootRiemannMetric(Jfac, quadpt.abscisse);

                        // get the basis function values
                        std::vector<T> bi_dataL(elL.nbasis());
                        trace.evalBasisQPL(iqp, bi_dataL.data());

                        // flux contribution weighted by quadrature and face metric 
                        // Li and Tang 2017 sec 9.1.1
                        T flux = mu * neumann_values[trace.face.bcflag]
                            * quadpt.weight * sqrtg;

                        for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                            resL[feidx{.idof = itest, .iv = 0}] -= bi_dataL[itest] * flux;
                        }
                    }
                }
                break;

                default:
                    // assume essential
                    std::cerr << "Warning: assuming essential BC." << std::endl;;
                    break;
            }
        }
    };
}
