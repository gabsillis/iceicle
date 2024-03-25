/**
 * @brief weak form for the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/MathUtils.hpp"
#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/matrixT.hpp"
#include "iceicle/anomaly_log.hpp"
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
     * \frac{du}{dt} = \mu \Delta u + f 
     * or -\mu\Delta u = f
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

    private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;
        using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Trace = ELEMENT::TraceSpace<T, IDX, ndim>;

    public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;

        /// @brief the dynamic number of vector components
        static const int dnv_comp = 1;

        /// @brief switch to use the interior penalty method instead of ddg 
        bool interior_penalty = false;

        /// @brief the diffusion coefficient (positive)
        T mu = 0.001;

        /// @brief advection coefficient 
        Tensor<T, ndim> a = []{
            Tensor<T, ndim> ret{};
            for(int idim = 0; idim < ndim; ++idim) ret[idim] = 0;
            return ret;
        }();

        /// @brief nonlinear advection coefficient
        Tensor<T, ndim> b = []{
            Tensor<T, ndim> ret{};
            for(int idim = 0; idim < ndim; ++idim) ret[idim] = 0;
            return ret;
        }();

        /// @brief the interior penalty coefficient for 
        /// Dirichlet boundary conditions 
        // see Arnold et al 2002 sec 2.1
        // This is an additional penalty see Huang, Chen, Li, Yan (2016)
        // WARNING: probably shouldn't be used 
        //
        // also: https://mooseframework.inl.gov/source/bcs/PenaltyDirichletBC.html
        T penalty = 0.0;

        /// @brief IC multiplier to get DDGIC
        /// see Danis Yan 2023 Journal of Scientific Computing
        /// DDGIC (sigma = 1)
        /// Default: Standard DDG (sigma = 0)
        T sigma_ic = 0.0;

        /// @brief the value for each bcflag (index into this list) 
        /// for dirichlet bc
        std::vector<T> dirichlet_values;

        /// @brief dirichlet value for each negative bcflag
        /// as a function callback 
        /// This function will take the physical domain point (size = ndim)
        /// and output neq values in the second argument
        std::vector< std::function<void(T *, T *)> > dirichlet_callbacks;

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
        void domainIntegral(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::elspan auto u,
            FE::elspan auto res
        ) const {

            std::vector<T> gradx_data(el.nbasis() * ndim);
            std::vector<T> bi(el.nbasis());

            // loop over the quadrature points
            for(int iqp = 0; iqp < el.nQP(); ++iqp){
                const QUADRATURE::QuadraturePoint<T, ndim> &quadpt = el.getQP(iqp);

                // calculate the jacobian determinant 
                auto J = el.geo_el->Jacobian(coord, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // get the gradients in the physical domain
                auto gradxBi = el.evalPhysGradBasisQP(iqp, coord, J, gradx_data.data());
                el.evalBasisQP(iqp, bi.data());

                // construct the value of u at the quadrature point
                T u_qp = 0;
                for(IDX ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    u_qp += u[ibasis, 0] * bi[ibasis];
                }

                Tensor<T, ndim> flux_adv{};
                for(int idim = 0; idim < ndim; ++idim){
                    flux_adv[idim] = a[idim] * u_qp + 0.5 * b[idim] * SQUARED(u_qp);
                }

                // construct the gradient of u
                T gradu[ndim] = {0};
                u.contract_mdspan(gradxBi, gradu);

                // loop over the test functions and construct the residual 
                for(std::size_t itest = 0; itest < el.nbasis(); ++itest){
                    for(std::size_t jdim = 0; jdim < ndim; ++jdim){
                        res[itest, 0]
                            += (flux_adv[jdim] - mu * gradu[jdim])
                                * gradxBi[itest, jdim] * detJ * quadpt.weight;
                    }
                }
            }
        }


        template<class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void traceIntegral(
            const ELEMENT::TraceSpace<T, IDX, ndim> &trace,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::dofspan<T, ULayoutPolicy, UAccessorPolicy> uL,
            FE::dofspan<T, ULayoutPolicy, UAccessorPolicy> uR,
            FE::dofspan<T, ResLayoutPolicy> resL,
            FE::dofspan<T, ResLayoutPolicy> resR
        ) const requires ( 
            FE::elspan<decltype(uL)> && 
            FE::elspan<decltype(uR)> && 
            FE::elspan<decltype(resL)> && 
            FE::elspan<decltype(resL)>
        ) {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;

            // calculate the centroids of the left and right elements
            // in the physical domain
            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;
            auto centroidL = elL.geo_el->centroid(coord);
            auto centroidR = elR.geo_el->centroid(coord);

            // Storage for Basis function and solution values
            std::vector<T> bi_dataL(elL.nbasis()); //TODO: move storage declaration out of loop
            std::vector<T> bi_dataR(elR.nbasis());
            std::vector<T> grad_dataL(elL.nbasis() * ndim);
            std::vector<T> grad_dataR(elR.nbasis() * ndim);
            std::vector<T> gradu_dataL(ndim);
            std::vector<T> gradu_dataR(ndim);
            std::vector<T> hess_dataL(elL.nbasis() * ndim * ndim);
            std::vector<T> hess_dataR(elR.nbasis() * ndim * ndim);
            std::vector<T> hessu_dataL(ndim * ndim);
            std::vector<T> hessu_dataR(ndim * ndim);


            // loop over the quadrature points 
            for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                // calculate the riemannian metric tensor root
                // TODO: could maybe reuse the left and righte jacobians because
                // these are needed for the gradients and hessians in the physical domn 
                // but we also need Jfac for unit_normal

                auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                // get the function values
                trace.evalBasisQPL(iqp, bi_dataL.data());
                trace.evalBasisQPR(iqp, bi_dataR.data());

                T value_uL = 0.0, value_uR = 0.0;
                for(std::size_t ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                { value_uL += uL[ibasis, 0] * bi_dataL[ibasis]; }
                for(std::size_t ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                { value_uR += uR[ibasis, 0] * bi_dataR[ibasis]; }

                // get the gradients the physical domain
                auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, grad_dataL.data());
                auto gradBiR = trace.evalPhysGradBasisQPR(iqp, coord, grad_dataR.data());

                auto graduL = uL.contract_mdspan(gradBiL, gradu_dataL.data());
                auto graduR = uR.contract_mdspan(gradBiR, gradu_dataR.data());

                // get the hessians in the physical domain 
                auto hessBiL = trace.evalPhysHessBasisQPL(iqp, coord, hess_dataL.data());
                auto hessBiR = trace.evalPhysHessBasisQPR(iqp, coord, hess_dataR.data());

                auto hessuL = uL.contract_mdspan(hessBiL, hessu_dataL.data());
                auto hessuR = uR.contract_mdspan(hessBiR, hessu_dataR.data());

                // calculate the normal vector 
                auto normal = calc_ortho(Jfac);
                auto unit_normal = normalize(normal);

                // calculate inviscid fluxes
                Tensor<T, ndim> fadv{};
                for(int idim = 0; idim < ndim; ++idim){
                    T fadvL = a[idim] * value_uL + b[idim] * SQUARED(value_uL);
                    T fadvR = a[idim] * value_uR + b[idim] * SQUARED(value_uR);

                    // flux vector splitting
                    T lambdaL = unit_normal[idim] * (a[idim] * 0.5 * b[idim] * value_uL);
                    T lambdaR = unit_normal[idim] * (a[idim] * 0.5 * b[idim] * value_uR);
                    T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
                    T lambda_r_plus = 0.5 * (lambdaR + std::abs(lambdaR));

                    fadv[idim] = value_uL * lambda_l_plus + value_uR * lambda_r_plus;
                }

                // calculate the DDG distance
                MATH::GEOMETRY::Point<T, ndim> phys_pt;
                trace.face->transform(quadpt.abscisse, coord, phys_pt.data());
                T h_ddg = 0;
                for(int idim = 0; idim < ndim; ++idim){
                    h_ddg += unit_normal[idim] * (
                        (phys_pt[idim] - centroidL[idim])
                        + (centroidR[idim] - phys_pt[idim])
                    );
                }
                h_ddg = std::abs(h_ddg);

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

                // switch to interior penalty if set
                if(interior_penalty) beta1 = 0.0;

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

                using namespace MATH::MATRIX_T;
                // calculate the flux weighted by the quadrature and face metric
                T fvisc = mu * dotprod<T, ndim>(grad_ddg, unit_normal.data()) 
                    * quadpt.weight * sqrtg;

                T fadvn = dotprod<T, ndim>(fadv.data(), unit_normal.data())
                    * quadpt.weight * sqrtg;

                // contribution to the residual 
                for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                    resL[itest, 0] += (fvisc - fadvn) * bi_dataL[itest];
                }
                for(std::size_t itest = 0; itest < elR.nbasis(); ++itest){
                    resR[itest, 0] -= (fvisc - fadvn) * bi_dataR[itest];
                }

                // apply the interface correction 
                // NOTE: only works for uniform basis order ?
                if( elL.nbasis() == elR.nbasis() ){

                    T average_gradv[ndim];
                    for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                        // get the average test function gradient
                        for(int idim = 0; idim < ndim; ++idim){
                            average_gradv[idim] = 0.5 * ( gradBiL[itest, idim] + gradBiR[itest, idim] );
                        }

                        // calcualate the interface correction integral contribution
                        T interface_correction = sigma_ic * jumpu * mu * 
                            MATH::MATRIX_T::dotprod<T, ndim>(average_gradv, unit_normal.data())
                            * quadpt.weight * sqrtg;
                        resL[itest, 0] -= interface_correction;
                        resR[itest, 0] -= interface_correction;
                    }
                } else if(sigma_ic != 0) {
                    ICEICLE::UTIL::AnomalyLog::log_anomaly(
                        ICEICLE::UTIL::Anomaly{
                            "DDGIC only works for same basis order throughout",
                            ICEICLE::UTIL::general_anomaly_tag{}}
                    );
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
            FE::dofspan<T, ULayoutPolicy, UAccessorPolicy> uL,
            FE::dofspan<T, ULayoutPolicy, UAccessorPolicy> uR,
            FE::dofspan<T, ResLayoutPolicy> resL
        ) const requires(
            FE::elspan<decltype(uL)> &&
            FE::elspan<decltype(uR)> &&
            FE::elspan<decltype(resL)> 
        ) {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;
            const FiniteElement &elL = trace.elL;

            auto centroidL = elL.geo_el->centroid(coord);

            switch(trace.face->bctype){
                case ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET: 
                {
                    // see Huang, Chen, Li, Yan 2016

                    // loop over quadrature points
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                        const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                        // calculate the normal vector 
                        auto normal = calc_ortho(Jfac);
                        auto unit_normal = normalize(normal);

                        // get the function values
                        std::vector<T> bi_dataL(elL.nbasis());
                        trace.evalBasisQPL(iqp, bi_dataL.data());

                        // get the gradients the physical domain
                        std::vector<T> grad_dataL(elL.nbasis() * ndim);
                        auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, grad_dataL.data());

                        std::vector<T> gradu_dataL(ndim);
                        auto graduL = uL.contract_mdspan(gradBiL, gradu_dataL.data());

                        T value_uL = 0.0;
                        for(std::size_t ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                        { value_uL += uL[ibasis, 0] * bi_dataL[ibasis]; }

                        // Get the value at the boundary 
                        T dirichlet_val;
                        if(trace.face->bcflag < 0){
                            // calback using physical domain location
                            MATH::GEOMETRY::Point<T, ndim> ref_pt, phys_pt;
                            trace.face->transform_xiL(quadpt.abscisse, ref_pt.data());
                            elL.transform(coord, ref_pt, phys_pt);
                            
                            dirichlet_callbacks[-trace.face->bcflag](phys_pt.data(), &dirichlet_val);

                        } else {
                            // just use the value
                            dirichlet_val = dirichlet_values[trace.face->bcflag];
                        }

                        // calculate inviscid fluxes
                        Tensor<T, ndim> fadv{};
                        for(int idim = 0; idim < ndim; ++idim){
                            // flux vector splitting
                            T lambdaL = unit_normal[idim] * (a[idim] * 0.5 * b[idim] * value_uL);
                            T lambdaR = unit_normal[idim] * (a[idim] * 0.5 * b[idim] * dirichlet_val);
                            T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
                            T lambda_r_plus = 0.5 * (lambdaR + std::abs(lambdaR));

                            fadv[idim] = value_uL * lambda_l_plus + dirichlet_val * lambda_r_plus;
                        }

                        // calculate the DDG distance
                        MATH::GEOMETRY::Point<T, ndim> phys_pt;
                        trace.face->transform(quadpt.abscisse, coord, phys_pt.data());
                        T h_ddg = 0; // uses distance to quadpt on boundary face
                        for(int idim = 0; idim < ndim; ++idim){
                            h_ddg += std::abs(unit_normal[idim] * 
                                (phys_pt[idim] - centroidL[idim])
                            );
                        }

                        static constexpr int ieq = 0;
                        // construct the DDG derivatives
                        int max_basis_order = 
                            elL.basis->getPolynomialOrder();

                        // Danis and Yan reccomended for NS
                        T beta0 = std::pow(max_basis_order + 1, 2);
                        T jumpu = dirichlet_val - value_uL;
                        // NOTE: directly constructing the normal gradient as in Huang, Chen, Li, Yan 2016
                        T grad_ddg_n = beta0 * jumpu / h_ddg;
                        for(int idim = 0; idim < ndim; ++idim){
                            grad_ddg_n += graduL[ieq, idim] * unit_normal[idim];
                        }

                        // calculate the flux weighted by the quadrature and face metric
                        T fvisc = (
                            mu * grad_ddg_n
                            + penalty * jumpu
                        ) * quadpt.weight * sqrtg;

                        using namespace MATH::MATRIX_T;
                        T fadvn = dotprod<T, ndim>(fadv.data(), unit_normal.data())
                            * quadpt.weight * sqrtg;

                        for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                            resL[itest, 0] += (fvisc - fadvn) * bi_dataL[itest];
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
                        auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                        // get the basis function values
                        std::vector<T> bi_dataL(elL.nbasis());
                        trace.evalBasisQPL(iqp, bi_dataL.data());

                        // flux contribution weighted by quadrature and face metric 
                        // Li and Tang 2017 sec 9.1.1
                        T flux = mu * neumann_values[trace.face->bcflag]
                            * quadpt.weight * sqrtg;

                        for(std::size_t itest = 0; itest < elL.nbasis(); ++itest){
                            resL[itest, 0] += bi_dataL[itest] * flux;
                        }
                    }
                }
                break;

                default:
                    // assume essential
                    std::cerr << "Warning: assuming essential BC." << std::endl;
                    break;
            }
        }

        void interface_conservation(
            Trace &trace,
            FE::NodalFEFunction<T, ndim> &coord,
            FE::elspan auto unkelL,
            FE::elspan auto &unkelR,
            FE::facspan auto res
        ) const {
            using namespace MATH::MATRIX_T;

            // calculate the centroids of the left and right elements
            // in the physical domain
            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;

            // Storage for Basis function and solution values
            std::vector<T> bi_dataL(elL.nbasis()); //TODO: move storage declaration out of loop
            std::vector<T> bi_dataR(elR.nbasis());
            std::vector<T> bi_trace(trace.nbasis_trace());
            std::vector<T> grad_dataL(elL.nbasis() * ndim);
            std::vector<T> grad_dataR(elR.nbasis() * ndim);
            std::vector<T> gradu_dataL(ndim);
            std::vector<T> gradu_dataR(ndim);
            std::vector<T> hess_dataL(elL.nbasis() * ndim * ndim);
            std::vector<T> hess_dataR(elR.nbasis() * ndim * ndim);
            std::vector<T> hessu_dataL(ndim * ndim);
            std::vector<T> hessu_dataR(ndim * ndim);
            for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                const QUADRATURE::QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                // calcualate Jacobian and metric
                auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                // calculate the normal vector 
                auto normal = calc_ortho(Jfac);
                auto unit_normal = normalize(normal);

                // get the function values
                trace.evalBasisQPL(iqp, bi_dataL.data());
                trace.evalBasisQPR(iqp, bi_dataR.data());

                T uL = 0.0, uR = 0.0;
                for(std::size_t ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                { uL += unkelL[ibasis, 0] * bi_dataL[ibasis]; }
                for(std::size_t ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                { uR += unkelR[ibasis, 0] * bi_dataR[ibasis]; }

                // get the gradients the physical domain
                auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, grad_dataL.data());
                auto gradBiR = trace.evalPhysGradBasisQPR(iqp, coord, grad_dataR.data());

                auto graduL = unkelL.contract_mdspan(gradBiL, gradu_dataL.data());
                auto graduR = unkelR.contract_mdspan(gradBiR, gradu_dataR.data());

                // get the test functions 
                trace.eval_trace_basis_qp(iqp, bi_trace.data());

                // viscous fluxes
                T fvisc_nL = -mu * dotprod<T, ndim>(graduL.data(), unit_normal.data());
                T fvisc_nR = -mu * dotprod<T, ndim>(graduR.data(), unit_normal.data());

                // inviscid fluxes 
                Tensor<T, ndim> fadvL;
                Tensor<T, ndim> fadvR;
                for(int idim = 0; idim < ndim; ++idim){
                    T fadvL[idim] = a[idim] * uL + b[idim] * SQUARED(uL);
                    T fadvR[idim] = a[idim] * uR + b[idim] * SQUARED(uR);
                }

                T fadv_nL = dotprod<T, ndim>(fadvL.data(), unit_normal.data());
                T fadv_nR = dotprod<T, ndim>(fadvR.data(), unit_normal.data());

                // calculate the flux jump
                T jumpF = (fvisc_nR - fvisc_nL) + (fadv_nR - fadv_nL);

                for(int itest = 0; itest < trace.nbasis_trace(); ++itest){
                    // integral contribution of interface conservation
                    T ic_res = jumpF * bi_trace[itest] * sqrtg * quadpt.weight;

                    if constexpr(decltype(res)::static_extent() == ndim){
                        // take the norm of the residual (scalar value in this case)
                        // and multiply by normal vector components 
                        for(IDX idim = 0; idim < ndim; ++idim){
                            res[itest, idim] += ic_res * unit_normal[idim];
                        }
                    } else {
                        // assume there is only one component 
                        res[itest, 0] += ic_res;
                    }
                }
            }

        }
    };
}
