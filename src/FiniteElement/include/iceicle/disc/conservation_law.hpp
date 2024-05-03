#pragma once

#include "Numtool/MathUtils.hpp"
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <cmath>
#include <type_traits>
#include <vector>
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
        static constexpr std::size_t nv_comp = 1;

        BurgersCoefficients<T, ndim>& coeffs;

        mutable T lambda_max = 0.0;

        /**
         * @brief compute the flux 
         * @param u the value of the solution
         * @param gradu the gradient of the solution 
         * @return the burgers flux function given the value and gradient of u 
         * F = au + 0.5*buu - mu * gradu
         */
        inline constexpr
        auto operator()(
            std::array<T, nv_comp> u,
            linalg::in_tensor auto gradu
        ) const noexcept -> Tensor<T, nv_comp, ndim> 
        {
            Tensor<T, nv_comp, ndim> flux{};
            T lambda_norm = 0;
            for(int idim = 0; idim < ndim; ++idim){
                T lambda = 
                    coeffs.a[idim]          // linear advection wavespeed
                    + 0.5 * coeffs.b[idim] * u[0];// nonlinear advection wavespeed
                lambda_norm += lambda * lambda;
                flux[0][idim] = 
                    lambda * u[0]                  // advection
                    - coeffs.mu * gradu[0, idim];  // diffusion
            }
            lambda_norm = std::sqrt(lambda_norm);
            lambda_max = std::max(lambda_max, lambda_norm);
            return flux;
        }

        /**
         * @brief get the timestep from cfl 
         * often this will require data to be set from the domain and boundary integrals 
         * such as wavespeeds, which will arise naturally during residual computation
         * WARNING: does not consider polynomial order of basis functions
         *
         * @param cfl the cfl condition 
         * @param reference_length the size to use for the length of the cfl condition 
         * @return the timestep based on the cfl condition
         */
        inline constexpr
        auto dt_from_cfl(T cfl, T reference_length) const  noexcept -> T {
            T aref = 0;
            aref = lambda_max;
            return (reference_length * cfl) / (coeffs.mu / reference_length + aref);
        }
    };
    template<class T, int ndim>
    BurgersFlux(BurgersCoefficients<T, ndim>) -> BurgersFlux<T, ndim>;

    /// @brief upwind numerical flux for the convective flux in burgers equation
    template <class T, int ndim>
    struct BurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;

        BurgersCoefficients<T, ndim>& coeffs;

        /**
         * @brief compute the convective numerical flux normal to the interface
         * F dot n
         * @param uL the value of the solution at interface for the left element
         * @param uR the value of the solution at interface for the right element
         * @return the upwind convective normal flux for burgers equation
         */
        inline constexpr
        auto operator()(
            std::array<T, nv_comp> uL,
            std::array<T, nv_comp> uR,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp>
        {
            T lambdaL = 0;
            T lambdaR = 0;
            for(int idim = 0; idim < ndim; ++idim){
                lambdaL += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL[0]);
                lambdaR += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR[0]);
            }

            T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
            T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));
            T fadvn = uL[0] * lambda_l_plus + uR[0] * lambda_r_plus;
            return std::array<T, nv_comp>{fadvn};
        }
    };
    template<class T, int ndim>
    BurgersUpwind(BurgersCoefficients<T, ndim>) -> BurgersUpwind<T, ndim>;

    /// @brief diffusive flux in burgers equation
    template <class T, int ndim>
    struct BurgersDiffusionFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

      /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;

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
            std::array<T, nv_comp> u,
            linalg::in_tensor auto gradu,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp>
        {
            using namespace MATH::MATRIX_T;
            // calculate the flux weighted by the quadrature and face metric
            T fvisc = 0;
            for(int idim = 0; idim < ndim; ++idim){
                fvisc += coeffs.mu * gradu[0, idim] * unit_normal[idim];
            }
            return std::array<T, nv_comp>{fvisc};
        }

        /// @brief compute the diffusive flux normal to the interface 
        /// given the prescribed normal gradient
        inline constexpr 
        auto neumann_flux(
            std::array<T, nv_comp> gradn
        ) const noexcept -> std::array<T, nv_comp> {
            return std::array<T, nv_comp>{coeffs.mu * gradn[0]};
        }
    };

    template<class T, int ndim>
    BurgersDiffusionFlux(BurgersCoefficients<T, ndim>) -> BurgersDiffusionFlux<T, ndim>;

    /// ==============================
    /// = Spacetime Burgers Equation =
    /// ==============================

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
        static constexpr std::size_t nv_comp = 1;
        static constexpr int ndim_space = ndim - 1;
        static constexpr int idim_time = ndim - 1;

        BurgersCoefficients<T, ndim_space>& coeffs;

        mutable T lambda_max = 0.0;

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
            std::array<T, nv_comp> u,
            linalg::in_tensor auto gradu
        ) const noexcept -> Tensor<T, nv_comp, ndim> 
        {
            Tensor<T, nv_comp, ndim> flux{};
            T lambda_norm = 0;
            for(int idim = 0; idim < ndim_space; ++idim){
                T lambda = 
                    coeffs.a[idim]          // linear advection wavespeed
                    + 0.5 * coeffs.b[idim] * u[0];// nonlinear advection wavespeed
                lambda_norm += lambda * lambda;
                flux[0][idim] = 
                    lambda * u[0]                  // advection
                    - coeffs.mu * gradu[0, idim];  // diffusion
            }
            flux[0][idim_time] = u[0];
            lambda_norm += SQUARED(u[0]);
            lambda_max = std::max(lambda_max, lambda_norm);
            return flux;
        }


        /**
         * @brief get the timestep from cfl 
         * often this will require data to be set from the domain and boundary integrals 
         * such as wavespeeds, which will arise naturally during residual computation
         * WARNING: does not consider polynomial order of basis functions
         *
         * @param cfl the cfl condition 
         * @param reference_length the size to use for the length of the cfl condition 
         * @return the timestep based on the cfl condition
         */
        inline constexpr
        auto dt_from_cfl(T cfl, T reference_length) const  noexcept -> T {
            T aref = 0;
            aref = lambda_max;
            return (reference_length * cfl) / (coeffs.mu / reference_length + aref);
        }
    };
    template<class T, int ndim_space>
    SpacetimeBurgersFlux(BurgersCoefficients<T, ndim_space>) -> SpacetimeBurgersFlux<T, ndim_space+1>;

    /// @brief upwind numerical flux for the convective flux in spacetime burgers equation
    template <class T, int ndim>
    struct SpacetimeBurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;
        static constexpr int ndim_space = ndim - 1;
        static constexpr int idim_time = ndim - 1;

        BurgersCoefficients<T, ndim_space>& coeffs;

        /**
         * @brief compute the convective numerical flux normal to the interface
         * F dot n
         * @param uL the value of the scalar solution at interface for the left element
         * @param uR the value of the scalar solution at interface for the right element
         * @return the upwind convective normal flux for burgers equation
         */
        inline constexpr
        auto operator()(
            std::array<T, nv_comp> uL,
            std::array<T, nv_comp> uR,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp> 
        {
            T lambdaL = 0;
            T lambdaR = 0;
            for(int idim = 0; idim < ndim_space; ++idim){
                lambdaL += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL[0]);
                lambdaR += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR[0]);
            }
            // time component
            lambdaL += unit_normal[idim_time];
            lambdaR += unit_normal[idim_time];

            T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
            T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));
            T fadvn = uL[0] * lambda_l_plus + uR[0] * lambda_r_plus;
            return std::array<T, nv_comp>{fadvn};
        }
    };

    template<class T, int ndim_space>
    SpacetimeBurgersUpwind(BurgersCoefficients<T, ndim_space>) -> SpacetimeBurgersUpwind<T, ndim_space+1>;

    template<class T, int ndim>
    struct SpacetimeBurgersDiffusion {

        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;
        static constexpr int ndim_space = ndim - 1;
        static constexpr int idim_time = ndim - 1;

        BurgersCoefficients<T, ndim_space>& coeffs;


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
            std::array<T, nv_comp> u,
            linalg::in_tensor auto gradu,
            Tensor<T, ndim> unit_normal
        ) const noexcept -> std::array<T, nv_comp>
        {
            using namespace MATH::MATRIX_T;
            // calculate the flux weighted by the quadrature and face metric
            T fvisc = 0;
            for(int idim = 0; idim < ndim_space; ++idim){
                fvisc += coeffs.mu * gradu[0, idim] * unit_normal[idim];
            }
            return std::array<T, nv_comp>{fvisc};
        }

        /// @brief compute the diffusive flux normal to the interface 
        /// given the prescribed normal gradient
        /// TODO: consider the time dimension somehow (might need normal vector)
        inline constexpr 
        auto neumann_flux(
            std::array<T, nv_comp> gradn
        ) const noexcept -> std::array<T, nv_comp> {
            return std::array<T, nv_comp>{coeffs.mu * gradn[0]};
        }
    };
    template<class T, int ndim_space>
    SpacetimeBurgersDiffusion(BurgersCoefficients<T, ndim_space>) -> SpacetimeBurgersDiffusion<T, ndim_space+1>;


    template<
        typename T,
        int ndim,
        class PhysicalFlux,
        class ConvectiveNumericalFlux,
        class DiffusiveFlux,
        class ST_Info = std::false_type
    >
    class ConservationLawDDG {

        private:
        PhysicalFlux phys_flux;
        ConvectiveNumericalFlux conv_nflux;
        DiffusiveFlux diff_flux;

        public:
        // ============
        // = Typedefs =
        // ============

        using value_type = T;

        // =============
        // = Constants =
        // =============

        /// @brief access the number of dimensions through a public interface
        static constexpr int dimensionality = ndim;

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = PhysicalFlux::nv_comp;
        static constexpr std::size_t dnv_comp = PhysicalFlux::nv_comp;

        // ==================
        // = Public Members =
        // ==================

        /// @brief switch to use the interior penalty method instead of ddg 
        bool interior_penalty = false;

        /// @brief IC multiplier to get DDGIC
        /// see Danis Yan 2023 Journal of Scientific Computing
        /// DDGIC (sigma = 1)
        /// Default: Standard DDG (sigma = 0)
        T sigma_ic = 0.0;

        /// @brief dirichlet value for each bcflag index
        /// as a function callback 
        /// This function will take the physical domain point (size = ndim)
        /// and output neq values in the second argument
        std::vector< std::function<void(const T*, T*)> > dirichlet_callbacks;

        /// @brief neumann value for each bcflag index
        /// as a function callback 
        /// This function will take the physical domain point (size = ndim)
        /// and output neq values in the second argument
        std::vector< std::function<void(const T*, T*)> > neumann_callbacks;

        /// @brief utility for SPACETIME_PAST boundary condition
        ST_Info spacetime_info;

        // ===============
        // = Constructor =
        // ===============
        //
        /** @brief construct from 
         * and take ownership of the fluxes and spacetime_info
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
         *
         * @param spacetime_info is the class that defines the SPACETIME_PAST connection
         * see iceicle::SpacetimeConnection
         */
        constexpr ConservationLawDDG(
            PhysicalFlux&& physical_flux,
            ConvectiveNumericalFlux&& convective_numflux,
            DiffusiveFlux&& diffusive_flux,
            ST_Info&& spacetime_info
        ) noexcept : phys_flux{physical_flux}, conv_nflux{convective_numflux}, 
            diff_flux{diffusive_flux}, spacetime_info{spacetime_info} {}


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


        // ============================
        // = Discretization Interface =
        // ============================

        /**
         * @brief get the timestep from cfl 
         * this takes it from the physical flux
         * often this will require data to be set from the domain and boundary integrals 
         * such as wavespeeds, which will arise naturally during residual computation
         * (WARNING: except for the very first iteration)
         *
         * @param cfl the cfl condition 
         * @param reference_length the size to use for the length of the cfl condition 
         * @return the timestep based on the cfl condition
         */
        T dt_from_cfl(T cfl, T reference_length){
            return phys_flux.dt_from_cfl(cfl, reference_length);
        }

        // =============
        // = Integrals =
        // =============

        template<class IDX>
        auto domain_integral(
            const FiniteElement<T, IDX, ndim> &el,
            NodeArray<T, ndim>& coord,
            elspan auto unkel,
            elspan auto res
        ) const -> void {
            static constexpr int neq = decltype(unkel)::static_extent();
            static_assert(neq == PhysicalFlux::nv_comp, "Number of equations must match.");
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;

            // basis function scratch space
            std::vector<T> bi(el.nbasis());
            std::vector<T> dbdx_data(el.nbasis() * ndim);

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
                auto gradu = unkel.contract_mdspan(gradxBi, gradu_data.data());

                // compute the flux  and scatter to the residual
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


        template<class IDX, class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
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
            static constexpr int neq = nv_comp;
            static_assert(neq == decltype(unkelL)::static_extent(), "Number of equations must match.");
            static_assert(neq == decltype(unkelR)::static_extent(), "Number of equations must match.");
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = FiniteElement<T, IDX, ndim>;

            // calculate the centroids of the left and right elements
            // in the physical domain
            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;
            auto centroidL = elL.geo_el->centroid(coord);
            auto centroidR = elR.geo_el->centroid(coord);

            // Basis function scratch space 
            std::vector<T> biL(elL.nbasis());
            std::vector<T> biR(elR.nbasis());
            std::vector<T> gradbL_data(elL.nbasis() * ndim);
            std::vector<T> gradbR_data(elR.nbasis() * ndim);
            std::vector<T> hessbL_data(elL.nbasis() * ndim * ndim);
            std::vector<T> hessbR_data(elR.nbasis() * ndim * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq> uR;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim> grad_ddg_data;
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
                trace.face->transform(quadpt.abscisse, coord, phys_pt);
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


                std::mdspan<T, std::extents<int, neq, ndim>> grad_ddg{grad_ddg_data.data()};
                for(int ieq = 0; ieq < neq; ++ieq){
                    // construct the DDG derivatives
                    T jumpu = uR[ieq] - uL[ieq];
                    for(int idim = 0; idim < ndim; ++idim){
                        grad_ddg[ieq, idim] = beta0 * jumpu / h_ddg * unit_normal[idim]
                            + 0.5 * (graduL[ieq, idim] + graduR[ieq, idim]);
                        T hessTerm = 0;
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            hessTerm += (hessuR[ieq, jdim, idim] - hessuL[ieq, jdim, idim])
                                * unit_normal[jdim];
                        }
                        grad_ddg[ieq, idim] += beta1 * h_ddg * hessTerm;
                    }
                }

                // construct the viscous fluxes 
                std::array<T, neq> uavg;
                for(int ieq = 0; ieq < neq; ++ieq) uavg[ieq] = 0.5 * (uL[ieq] + uR[ieq]);

                std::array<T, neq> fviscn = diff_flux(uavg, grad_ddg, unit_normal);

                // scale by weight and face metric tensor
                for(int ieq = 0; ieq < neq; ++ieq){
                    fadvn[ieq] *= quadpt.weight * sqrtg;
                    fviscn[ieq] *= quadpt.weight * sqrtg;
                }

                // scatter contribution 
                for(int itest = 0; itest < elL.nbasis(); ++itest){
                    for(int ieq = 0; ieq < neq; ++ieq){
                        resL[itest, 0] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                    }
                }
                for(int itest = 0; itest < elR.nbasis(); ++itest){
                    for(int ieq = 0; ieq < neq; ++ieq){
                        resR[itest, 0] -= (fviscn[ieq] - fadvn[ieq]) * biR[itest];
                    }
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
        template<class IDX, class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
        void boundaryIntegral(
            const TraceSpace<T, IDX, ndim> &trace,
            NodeArray<T, ndim> &coord,
            dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelL,
            dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelR,
            dofspan<T, ResLayoutPolicy> resL
        ) const requires(
            elspan<decltype(unkelL)> &&
            elspan<decltype(unkelR)> &&
            elspan<decltype(resL)> 
        ) {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = FiniteElement<T, IDX, ndim>;
            const FiniteElement &elL = trace.elL;

            static constexpr int neq = nv_comp;

            // Basis function scratch space 
            std::vector<T> biL(elL.nbasis());
            std::vector<T> gradbL_data(elL.nbasis() * ndim);
            std::vector<T> hessbL_data(elL.nbasis() * ndim * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim> grad_ddg_data;
            std::array<T, neq * ndim * ndim> hessuL_data;
            std::array<T, neq * ndim * ndim> hessuR_data;
            auto centroidL = elL.geo_el->centroid(coord);

            switch(trace.face->bctype){
                case BOUNDARY_CONDITIONS::DIRICHLET: 
                {
                    // see Huang, Chen, Li, Yan 2016

                    // loop over quadrature points
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                        const QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                        // calculate the normal vector 
                        auto normal = calc_ortho(Jfac);
                        auto unit_normal = normalize(normal);

                        // calculate the physical domain position
                        MATH::GEOMETRY::Point<T, ndim> phys_pt;
                        trace.face->transform(quadpt.abscisse, coord, phys_pt);

                        // get the function values
                        trace.evalBasisQPL(iqp, biL.data());

                        // get the gradients the physical domain
                        auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, gradbL_data.data());

                        auto graduL = unkelL.contract_mdspan(gradBiL, graduL_data.data());

                        // construct the solution on the left and right
                        std::ranges::fill(uL, 0.0);
                        for(int ieq = 0; ieq < neq; ++ieq){
                            for(int ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                                { uL[ieq] += unkelL[ibasis, ieq] * biL[ibasis]; }
                        }

                        // Get the values at the boundary 
                        std::array<T, nv_comp> dirichlet_vals{};
                        dirichlet_callbacks[trace.face->bcflag](phys_pt.data(), dirichlet_vals.data());

                        // compute convective fluxes
                        std::array<T, neq> fadvn = conv_nflux(uL, dirichlet_vals, unit_normal);

                        // calculate the DDG distance
                        T h_ddg = 0; // uses distance to quadpt on boundary face
                        for(int idim = 0; idim < ndim; ++idim){
                            h_ddg += std::abs(unit_normal[idim] * 
                                (phys_pt[idim] - centroidL[idim])
                            );
                        }

                        // construct the DDG derivatives
                        int order = 
                            elL.basis->getPolynomialOrder();
                        // Danis and Yan reccomended for NS
                        T beta0 = std::pow(order + 1, 2);
                        T beta1 = 1 / std::max((T) (2 * order * (order + 1)), 1.0);

                        std::mdspan<T, std::extents<int, neq, ndim>> grad_ddg{grad_ddg_data.data()};
                        for(int ieq = 0; ieq < neq; ++ieq){
                            // construct the DDG derivatives
                            T jumpu = dirichlet_vals[ieq] - uL[ieq];
                            for(int idim = 0; idim < ndim; ++idim){
                                grad_ddg[ieq, idim] = beta0 * jumpu / h_ddg * unit_normal[idim]
                                    + (graduL[ieq, idim]);
                            }
                        }

                        // construct the viscous fluxes 
                        std::array<T, neq> uavg;
                        for(int ieq = 0; ieq < neq; ++ieq) uavg[ieq] = 0.5 * (uL[ieq] + dirichlet_vals[ieq]);

                        std::array<T, neq> fviscn = diff_flux(uavg, grad_ddg, unit_normal);

                        // scale by weight and face metric tensor
                        for(int ieq = 0; ieq < neq; ++ieq){
                            fadvn[ieq] *= quadpt.weight * sqrtg;
                            fviscn[ieq] *= quadpt.weight * sqrtg;
                        }

                        // scatter contribution 
                        for(int itest = 0; itest < elL.nbasis(); ++itest){
                            for(int ieq = 0; ieq < neq; ++ieq){
                                resL[itest, 0] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                            }
                        }
                    }
                }
                break;

                // NOTE: Neumann Boundary conditions prescribe a solution gradient 
                // For this we only use the diffusive flux 
                // DiffusiveFlux must provide a neumann_flux function that takes the normal gradient
                // If the diffusive flux uses the solution value this will not work 
                // this also ignores the convective flux because hyperbolic problems 
                // dont have a notion of Neumann BC (use an outflow or extrapolation BC instead)
                case BOUNDARY_CONDITIONS::NEUMANN:
                {
                    // loop over quadrature points 
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){

                        const QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                        // calculate the physical domain position
                        MATH::GEOMETRY::Point<T, ndim> phys_pt;
                        trace.face->transform(quadpt.abscisse, coord, phys_pt);

                        // get the basis function values
                        std::vector<T> bi_dataL(elL.nbasis());
                        trace.evalBasisQPL(iqp, bi_dataL.data());

                        // Get the values at the boundary 
                        std::array<T, nv_comp> neumann_vals{};
                        neumann_callbacks[trace.face->bcflag](phys_pt.data(), neumann_vals.data());
                        // flux contribution weighted by quadrature and face metric 
                        // Li and Tang 2017 sec 9.1.1
                        std::array<T, nv_comp> fviscn = diff_flux.neumann_flux(neumann_vals);

                        // scale by weight and face metric tensor
                        for(int ieq = 0; ieq < neq; ++ieq){
                            fviscn[ieq] *= quadpt.weight * sqrtg;
                        }

                        // scatter contribution 
                        for(int itest = 0; itest < elL.nbasis(); ++itest){
                            for(int ieq = 0; ieq < neq; ++ieq){
                                resL[itest, 0] += (fviscn[ieq]) * biL[itest];
                            }
                        }
                    }
                }
                break;

                case BOUNDARY_CONDITIONS::SPACETIME_PAST:
                if constexpr(!std::same_as<ST_Info, std::false_type>){

                    static constexpr int neq = nv_comp;
                    static_assert(neq == decltype(unkelL)::static_extent(), "Number of equations must match.");
                    static_assert(neq == decltype(unkelR)::static_extent(), "Number of equations must match.");
                    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

                    // Get the info from the connection
                    TraceSpace<T, IDX, ndim>& trace_past = spacetime_info.connection_traces[trace.facidx];
                    std::vector<T> unker_past_data{trace_past.elL.nbasis() * neq};
                    auto uR_layout = spacetime_info.u_past.create_element_layout(trace_past.elR.elidx);
                    dofspan unkel_past{unker_past_data, uR_layout};
                    extract_elspan(trace_past.elR.elidx, spacetime_info.u_past, unkel_past);
                    AbstractMesh<T, IDX, ndim>* mesh_past = spacetime_info.fespace_past.meshptr;

                    // calculate the centroids of the left and right elements
                    // in the physical domain
                    const FiniteElement &elL = trace.elL;
                    const FiniteElement &elR = trace_past.elR;
                    auto centroidL = elL.geo_el->centroid(coord);

                    // Basis function scratch space 
                    std::vector<T> biL(elL.nbasis());
                    std::vector<T> biR(elR.nbasis());
                    std::vector<T> gradbL_data(elL.nbasis() * ndim);
                    std::vector<T> gradbR_data(elR.nbasis() * ndim);
                    std::vector<T> hessbL_data(elL.nbasis() * ndim * ndim);
                    std::vector<T> hessbR_data(elR.nbasis() * ndim * ndim);

                    // solution scratch space 
                    std::array<T, neq> uL;
                    std::array<T, neq> uR;
                    std::array<T, neq * ndim> graduL_data;
                    std::array<T, neq * ndim> graduR_data;
                    std::array<T, neq * ndim> grad_ddg_data;
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
                        trace_past.evalBasisQPR(iqp, biR.data());
                        auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, gradbL_data.data());
                        auto gradBiR = trace_past.evalPhysGradBasisQPR(iqp, mesh_past->nodes , gradbR_data.data());
                        auto hessBiL = trace.evalPhysHessBasisQPL(iqp, coord, hessbL_data.data());
                        auto hessBiR = trace_past.evalPhysHessBasisQPR(iqp, mesh_past->nodes, hessbR_data.data());

                        // construct the solution on the left and right
                        std::ranges::fill(uL, 0.0);
                        std::ranges::fill(uR, 0.0);
                        for(int ieq = 0; ieq < neq; ++ieq){
                            for(int ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                                { uL[ieq] += unkelL[ibasis, ieq] * biL[ibasis]; }
                            for(int ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                                { uR[ieq] += unkel_past[ibasis, ieq] * biR[ibasis]; }
                        }

                        // get the solution gradient and hessians
                        auto graduL = unkelL.contract_mdspan(gradBiL, graduL_data.data());
                        auto graduR = unkel_past.contract_mdspan(gradBiR, graduR_data.data());
                        auto hessuL = unkelL.contract_mdspan(hessBiL, hessuL_data.data());
                        auto hessuR = unkel_past.contract_mdspan(hessBiR, hessuR_data.data());

                        // compute convective fluxes
                        std::array<T, neq> fadvn = conv_nflux(uL, uR, unit_normal);

                        // compute a single valued gradient using DDG or IP 

                        // calculate the DDG distance
                        MATH::GEOMETRY::Point<T, ndim> phys_pt;
                        trace.face->transform(quadpt.abscisse, coord, phys_pt);
                        T h_ddg = 0;
                        for(int idim = 0; idim < ndim; ++idim){
                            h_ddg += unit_normal[idim] * (
                                2 * (phys_pt[idim] - centroidL[idim])
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


                        std::mdspan<T, std::extents<int, neq, ndim>> grad_ddg{grad_ddg_data.data()};
                        for(int ieq = 0; ieq < neq; ++ieq){
                            // construct the DDG derivatives
                            T jumpu = uR[ieq] - uL[ieq];
                            for(int idim = 0; idim < ndim; ++idim){
                                grad_ddg[ieq, idim] = beta0 * jumpu / h_ddg * unit_normal[idim]
                                    + 0.5 * (graduL[ieq, idim] + graduR[ieq, idim]);
                                T hessTerm = 0;
                                for(int jdim = 0; jdim < ndim; ++jdim){
                                    hessTerm += (hessuR[ieq, jdim, idim] - hessuL[ieq, jdim, idim])
                                        * unit_normal[jdim];
                                }
                                grad_ddg[ieq, idim] += beta1 * h_ddg * hessTerm;
                            }
                        }

                        // construct the viscous fluxes 
                        std::array<T, neq> uavg;
                        for(int ieq = 0; ieq < neq; ++ieq) uavg[ieq] = 0.5 * (uL[ieq] + uR[ieq]);

                        std::array<T, neq> fviscn = diff_flux(uavg, grad_ddg, unit_normal);

                        // scale by weight and face metric tensor
                        for(int ieq = 0; ieq < neq; ++ieq){
                            fadvn[ieq] *= quadpt.weight * sqrtg;
                            fviscn[ieq] *= quadpt.weight * sqrtg;
                        }

                        // scatter contribution 
                        for(int itest = 0; itest < elL.nbasis(); ++itest){
                            for(int ieq = 0; ieq < neq; ++ieq){
                                resL[itest, 0] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                            }
                        }
                    }
                    
                }
                break;

                case BOUNDARY_CONDITIONS::SPACETIME_FUTURE:
                    // NOTE: Purely upwind so continue on to use EXTRAPOLATION

                // Use only the interior state and assume the exterior state (and gradients) match
                case BOUNDARY_CONDITIONS::EXTRAPOLATION:
                {
                    // loop over quadrature points
                    for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                        const QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                        // calculate the jacobian and riemannian metric root det
                        auto Jfac = trace.face->Jacobian(coord, quadpt.abscisse);
                        T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                        // calculate the normal vector 
                        auto normal = calc_ortho(Jfac);
                        auto unit_normal = normalize(normal);

                        // calculate the physical domain position
                        MATH::GEOMETRY::Point<T, ndim> phys_pt;
                        trace.face->transform(quadpt.abscisse, coord, phys_pt);

                        // get the function values
                        trace.evalBasisQPL(iqp, biL.data());

                        // get the gradients the physical domain
                        auto gradBiL = trace.evalPhysGradBasisQPL(iqp, coord, gradbL_data.data());

                        auto graduL = unkelL.contract_mdspan(gradBiL, graduL_data.data());

                        // construct the solution on the left and right
                        std::ranges::fill(uL, 0.0);
                        for(int ieq = 0; ieq < neq; ++ieq){
                            for(int ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                                { uL[ieq] += unkelL[ibasis, ieq] * biL[ibasis]; }
                        }

                        // compute convective fluxes
                        std::array<T, neq> fadvn = conv_nflux(uL, uL, unit_normal);

                        // construct the DDG derivatives
                        std::mdspan<T, std::extents<int, neq, ndim>> grad_ddg{grad_ddg_data.data()};
                        for(int ieq = 0; ieq < neq; ++ieq){
                            // construct the DDG derivatives ( just match interior gradient )
                            for(int idim = 0; idim < ndim; ++idim){
                                grad_ddg[ieq, idim] = (graduL[ieq, idim]);
                            }
                        }

                        // construct the viscous fluxes 
                        std::array<T, neq> uavg;
                        for(int ieq = 0; ieq < neq; ++ieq) uavg[ieq] = uL[ieq];

                        std::array<T, neq> fviscn = diff_flux(uavg, grad_ddg, unit_normal);

                        // scale by weight and face metric tensor
                        for(int ieq = 0; ieq < neq; ++ieq){
                            fadvn[ieq] *= quadpt.weight * sqrtg;
                            fviscn[ieq] *= quadpt.weight * sqrtg;
                        }

                        // scatter contribution 
                        for(int itest = 0; itest < elL.nbasis(); ++itest){
                            for(int ieq = 0; ieq < neq; ++ieq){
                                resL[itest, 0] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                            }
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

        template<class IDX>
        void interface_conservation(
            const TraceSpace<T, IDX, ndim>& trace,
            NodeArray<T, ndim>& coord,
            elspan auto unkelL,
            elspan auto unkelR,
            facspan auto res
        ) const {
            static constexpr int neq = nv_comp;
            using namespace MATH::MATRIX_T;
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = FiniteElement<T, IDX, ndim>;

            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;

            // Basis function scratch space 
            std::vector<T> biL(elL.nbasis());
            std::vector<T> biR(elR.nbasis());
            std::vector<T> gradbL_data(elL.nbasis() * ndim);
            std::vector<T> gradbR_data(elR.nbasis() * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq> uR;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim> grad_ddg_data;

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

                // get the physical flux on the left and right
                Tensor<T, nv_comp, ndim> fluxL = phys_flux(uL, graduL);
                Tensor<T, nv_comp, ndim> fluxR = phys_flux(uR, graduR);

                // calculate the jump in normal fluxes
                Tensor<T, nv_comp> jumpflux{};
                for(int ieq = 0; ieq < nv_comp; ++ieq) 
                    jumpflux[ieq] = dot(fluxR[ieq], unit_normal) - dot(fluxL[ieq], unit_normal);

                // get the norm and multiply by unit normal vector for square system
                T jumpflux_norm = std::sqrt(norml2(jumpflux));

                // scatter unit normal times interface conservation to residual
                for(int itest = 0; itest < trace.nbasis_trace(); ++itest){
                    T ic_res = jumpflux_norm * sqrtg * quadpt.weight;
                    for(int idim = 0; idim < ndim; ++idim){
                        res[itest, idim] += ic_res * unit_normal[idim];
                    }
                }
            }
        }
    };

    // Deduction Guides
    template<
        class T,
        int ndim,
        template<class T1, int ndim1> class PhysicalFlux,
        template<class T1, int ndim1> class ConvectiveNumericalFlux, 
        template<class T1, int ndim1> class DiffusiveFlux
    >
    ConservationLawDDG(
        PhysicalFlux<T, ndim>&& physical_flux,
        ConvectiveNumericalFlux<T, ndim>&& convective_numflux,
        DiffusiveFlux<T, ndim>&& diffusive_flux
    ) -> ConservationLawDDG<T, ndim, PhysicalFlux<T, ndim>, ConvectiveNumericalFlux<T, ndim>, DiffusiveFlux<T, ndim>>;

    template<
        class T,
        int ndim,
        template<class T1, int ndim1> class PhysicalFlux,
        template<class T1, int ndim1> class ConvectiveNumericalFlux, 
        template<class T1, int ndim1> class DiffusiveFlux,
        class ST_Info
    >
    ConservationLawDDG(
        PhysicalFlux<T, ndim>&& physical_flux,
        ConvectiveNumericalFlux<T, ndim>&& convective_numflux,
        DiffusiveFlux<T, ndim>&& diffusive_flux,
        ST_Info st_info
    ) -> ConservationLawDDG<T, ndim, PhysicalFlux<T, ndim>, ConvectiveNumericalFlux<T, ndim>, DiffusiveFlux<T, ndim>, ST_Info>;
}
