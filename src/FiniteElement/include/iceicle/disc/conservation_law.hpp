#pragma once

#include <type_traits>
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
namespace iceicle {

    /// @brief coefficients to define the burgers equation
    template<class T, int ndim>
    struct BurgersCoefficients {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:
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

    };

    template<
        class T,
        int ndim
    >
    struct BurgersFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;

        BurgersCoefficients<T, ndim>& coeffs;

        /**
         * @brief compute the flux 
         * @param u the value of the scalar solution
         * @param gradu the gradient of the solution 
         * @return the burgers flux function given the value and gradient of u 
         * F = au + 0.5*buu - mu * gradu
         */
        inline constexpr
        auto operator()(
            T u,
            std::span<T, ndim> gradu
        ) const noexcept -> Tensor<T, ndim> 
        {
            Tensor<T, ndim> flux{};
            for(int idim = 0; idim < ndim; ++idim){
                flux[idim] = 
                    coeffs.a[idim] * u                  // linear convection
                    + 0.5 * coeffs.b[idim] * SQUARED(u) // nonlinear convection
                    - coeffs.mu * gradu[idim];          // diffusion
            }
            return flux;
        }
    };

    /// @brief upwind numerical flux for the convective flux in burgers equation
    template <class T, int ndim>
    struct BurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;

        BurgersCoefficients<T, ndim>& coeffs;

        /**
         * @brief compute the convective numerical flux normal to the interface
         * F dot n
         * @param uL the value of the scalar solution at interface for the left element
         * @param uR the value of the scalar solution at interface for the right element
         * @return the upwind convective normal flux for burgers equation
         */
        inline constexpr
        auto operator()(
            T uL,
            T uR,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp>
        {
            T fadvn = 0;
            for(int idim = 0; idim < ndim; ++idim){
                // flux vector splitting
                T lambdaL = unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL);
                T lambdaR = unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR);
                T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
                T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));

                fadvn += uL * lambda_l_plus + uR * lambda_r_plus;
            }
            return std::array<T, nv_comp>{fadvn};
        }
    };

    /// @brief diffusive flux in burgers equation
    template <class T, int ndim>
    struct BurgersDiffusionFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;

        BurgersCoefficients<T, ndim>& coeffs;


        /**
         * @brief compute the diffusive flux normal to the interface
         * F dot n
         * @param u the single valued solution at the interface 
         * @param gradu the single valued gradient at the interface 
         * @param unit normal the unit normal
         * @return the diffusive normal flux for burgers equation
         */
        inline constexpr
        auto operator()(
            T u,
            linalg::in_tensor auto gradu,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp>
        {
            using namespace MATH::MATRIX_T;
            // calculate the flux weighted by the quadrature and face metric
            T fvisc = 0;
            for(int idim = 0; idim < ndim; ++idim){
                fvisc += gradu[0, idim] * unit_normal[idim];
            }
            return std::array<T, nv_comp>{fvisc};
        }
    };

    template<
        class T,
        int ndim
    >
    struct SpacetimeBurgersFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;
        static constexpr int ndim_space = ndim - 1;
        static constexpr int idim_time = ndim - 1;

        BurgersCoefficients<T, ndim-1>& coeffs;

        /**
         * @brief compute the flux 
         * @param u the value of the scalar solution
         * @param gradu the gradient of the solution 
         * @return the burgers flux function given the value and gradient of u 
         * in space dimensions:
         * F = au + 0.5*buu - mu * gradu
         * in time dimension:
         * F = u
         *
         */
        inline constexpr
        auto operator()(
            T u,
            std::span<T, ndim> gradu
        ) const noexcept -> Tensor<T, ndim> 
        {
            Tensor<T, ndim> flux{};
            for(int idim = 0; idim < ndim_space; ++idim){
                flux[idim] = 
                    coeffs.a[idim] * u                  // linear convection
                    + 0.5 * coeffs.b[idim] * SQUARED(u) // nonlinear convection
                    - coeffs.mu * gradu[idim];          // diffusion
            }
            flux[idim_time] = u;
            return flux;
        }
    };

    /// @brief upwind numerical flux for the convective flux in spacetime burgers equation
    template <class T, int ndim>
    struct SpacetimeBurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr int nv_comp = 1;
        static constexpr int ndim_space = ndim - 1;
        static constexpr int idim_time = ndim - 1;

        BurgersCoefficients<T, ndim-1>& coeffs;

        /**
         * @brief compute the convective numerical flux normal to the interface
         * F dot n
         * @param uL the value of the scalar solution at interface for the left element
         * @param uR the value of the scalar solution at interface for the right element
         * @return the upwind convective normal flux for burgers equation
         */
        inline constexpr
        auto operator()(
            T uL,
            T uR,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp> 
        {
            T fadvn = 0;
            for(int idim = 0; idim < ndim_space; ++idim){
                // flux vector splitting
                T lambdaL = unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL);
                T lambdaR = unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR);
                T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
                T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));

                fadvn += uL * lambda_l_plus + uR * lambda_r_plus;
            }
            return std::array<T, nv_comp>{fadvn};
        }
    };

    template<
        typename T,
        typename IDX,
        int ndim,
        class PhysicalFlux,
        class ConvectiveNumericalFlux,
        class DiffusiveFlux
    >
    class ConservationLawDDG {

        // ============
        // = Typedefs =
        // ============

        using value_type = T;
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        PhysicalFlux phys_flux;
        ConvectiveNumericalFlux conv_nflux;
        DiffusiveFlux diff_flux;


        /// @brief switch to use the interior penalty method instead of ddg 
        bool interior_penalty = false;

        /// @brief IC multiplier to get DDGIC
        /// see Danis Yan 2023 Journal of Scientific Computing
        /// DDGIC (sigma = 1)
        /// Default: Standard DDG (sigma = 0)
        T sigma_ic = 0.0;

        // ===============
        // = Constructor =
        // ===============

        /** @brief construct from 
         * and take ownership of the fluxes
         *
         * @param physical_flux the actual discretization flux 
         * F in grad(F) = 0 
         * or for method of lines:
         * du/dt + grad(F) = 0
         *
         * @param convective_numflux the numerical flux for the 
         * convective portion (typically a Riemann Solver or Upwinding method)
         *
         * @param diffusive_numflux the numerical flux for the diffusive portion
         */
        constexpr ConservationLawDDG(
            PhysicalFlux&& physical_flux,
            ConvectiveNumericalFlux&& convective_numflux,
            DiffusiveFlux&& diffusive_flux
        ) noexcept : phys_flux{physical_flux}, conv_nflux{convective_numflux}, 
            diff_flux{diffusive_flux} {}

        // =============
        // = Integrals =
        // =============

        auto domain_integral(
            const FiniteElement<T, IDX, ndim> &el,
            NodeArray<T, ndim>& coord,
            elspan auto unkel,
            elspan auto res
        ) const -> void {
            static constexpr int neq = decltype(unkel)::static_extent;
            static_assert(neq == PhysicalFlux::nv_comp, "Number of equations must match.");
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;

            // basis function scratch space
            std::valarray<T> bi(el.nbasis());
            std::valarray<T> dbdx_data(el.nbasis() * ndim);

            // solution scratch space
            std::array<T, neq> u;
            std::array<T, neq * ndim> gradu_data;

            // loop over the quadrature points
            for(int iqp = 0; iqp < el.nQP(); ++iqp){
                const QuadraturePoint<T, ndim> &quadpt = el.getQP(iqp);

                // calculate the jacobian determinant 
                auto J = el.geo_el->Jacobian(coord, quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // get the basis functions and gradients in the physical domain
                el.evalBasisQP(iqp, bi.data());
                auto gradxBi = el.evalPhysGradBasisQP(iqp, coord, J, dbdx_data.data());

                // construct the solution U at the quadrature point 
                std::ranges::fill(u, 0.0);
                for(int ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    for(int ieq = 0; ieq < neq; ++ieq){
                        u[ieq] += unkel[ibasis, ieq] * bi[ibasis];
                    }
                }

                // construct the gradient of u 
                auto gradu = unkel.contract_mdspan(gradxBi, gradu_data);

                // compute the flux  and scatter to the residual
                if constexpr(neq == 1){
                    // scalar conservation law
                    // don't need neq array dimension
                    Tensor<T, ndim> flux = phys_flux(u[0], gradu_data);

                    // loop over the test functions and construct the residual 
                    for(int itest = 0; itest < el.nbasis(); ++itest){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            res[itest, 0]
                                += flux[jdim] * gradxBi[itest, jdim] * detJ * quadpt.weight;
                        }
                    }
                } else {
                    Tensor<T, neq, ndim> flux = phys_flux(u, gradu);

                    // loop over the test functions and construc the residual
                    for(int itest = 0; itest < el.nbasis(); ++itest){
                        for(int ieq = 0; ieq < neq; ++ieq){
                            for(int jdim = 0; jdim < ndim; ++jdim){
                                res[itest, ieq]
                                    += flux[ieq][jdim] * gradxBi[itest, jdim] * detJ * quadpt.weight;
                            }
                        }
                    }
                }
            }
        }


        template<class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void trace_integral(
            const TraceSpace<T, IDX, ndim> &trace,
            NodeArray<T, ndim> &coord,
            dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelL,
            dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelR,
            dofspan<T, ResLayoutPolicy> resL,
            dofspan<T, ResLayoutPolicy> resR
        ) const requires ( 
            elspan<decltype(unkelL)> && 
            elspan<decltype(unkelR)> && 
            elspan<decltype(resL)> && 
            elspan<decltype(resL)>
        ) {
            static constexpr int neq = ConvectiveNumericalFlux::nv_comp;
            static_assert(neq == decltype(unkelL)::static_extent, "Number of equations must match.");
            static_assert(neq == decltype(unkelR)::static_extent, "Number of equations must match.");
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = FiniteElement<T, IDX, ndim>;

            // calculate the centroids of the left and right elements
            // in the physical domain
            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;
            auto centroidL = elL.geo_el->centroid(coord);
            auto centroidR = elR.geo_el->centroid(coord);

            // Basis function scratch space 
            std::valarray<T> biL(elL.nbasis());
            std::valarray<T> biR(elR.nbasis());
            std::valarray<T> gradbL_data(elL.nbasis() * ndim);
            std::valarray<T> gradbR_data(elR.nbasis() * ndim);
            std::valarray<T> hessbL_data(elL.nbasis * ndim * ndim);
            std::valarray<T> hessbR_data(elR.nbasis * ndim * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq> uR;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim * ndim> hessuL_data;
            std::array<T, neq * ndim * ndim> hessuR_data;


            // loop over the quadrature points 
            for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                const QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                // calculate the riemannian metric tensor root
                auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                // calculate the normal vector 
                auto normal = calc_ortho(Jfac);
                auto unit_normal = normalize(normal);

                // get the basis functions, derivatives, and hessians
                // (derivatives are wrt the physical domain)
                trace.evalBasisQPL(iqp, biL.data());
                trace.evalBasisQPR(iqp, biR.data());
                auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, gradbL_data.data());
                auto gradBiR = trace.evalPhysGradBasisQPR(iqp, coord, gradbR_data.data());
                auto hessBiL = trace.evalPhysHessBasisQPL(iqp, coord, hessbL_data.data());
                auto hessBiR = trace.evalPhysHessBasisQPR(iqp, coord, hessbR_data.data());

                // construct the solution on the left and right
                std::ranges::fill(uL, 0.0);
                std::ranges::fill(uR, 0.0);
                for(int ieq = 0; ieq < neq; ++ieq){
                    for(int ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                        { uL[ieq] += unkelL[ibasis, ieq] * biL[ibasis]; }
                    for(int ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                        { uR[ieq] += unkelR[ibasis, ieq] * biR[ibasis]; }
                }

                // get the solution gradient and hessians
                auto graduL = unkelL.contract_mdspan(gradBiL, graduL_data.data());
                auto graduR = unkelR.contract_mdspan(gradBiR, graduR_data.data());
                auto hessuL = unkelL.contract_mdspan(hessBiL, hessuL_data.data());
                auto hessuR = unkelR.contract_mdspan(hessBiR, hessuR_data.data());

                // compute convective fluxes
                std::array<T, neq> fadvn = conv_nflux(uL, uR, unit_normal);

                // compute a single valued gradient using DDG or IP 

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
                
                int order = std::max(
                    elL.basis->getPolynomialOrder(),
                    elR.basis->getPolynomialOrder()
                );
                // Danis and Yan reccomended for NS
                T beta0 = std::pow(order + 1, 2);
                T beta1 = 1 / std::max((T) (2 * order * (order + 1)), 1.0);

                // switch to interior penalty if set
                if(interior_penalty) beta1 = 0.0;
                for(int ieq = 0; ieq < neq; ++ieq){
                    // construct the DDG derivatives
                    T grad_ddg[ndim];
                    T jumpu = uR - uL;
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
                }

            }
        }

    };
}
