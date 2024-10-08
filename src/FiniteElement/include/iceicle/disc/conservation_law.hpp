#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/point.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <cmath>
#include <string>
#include <type_traits>
#include <vector>
namespace iceicle {

    /// @brief a Physical flux given a vector valued state u 
    /// and the gradient of u 
    /// returns a vector valued flux for each dimension F(u, gradu)
    template<class FluxT>
    concept physical_flux = 
    requires(
        FluxT flux,
        std::array<typename FluxT::value_type, FluxT::nv_comp> u,
        std::mdspan<typename FluxT::value_type, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> gradu
    ) {
        { flux(u, gradu) } -> std::same_as<NUMTOOL::TENSOR::FIXED_SIZE::Tensor<typename FluxT::value_type, FluxT::nv_comp, FluxT::ndim>> ;
    };

    /// @brief Implementation of a numerical flux for convective fluxes 
    ///
    /// given a state uL and uR on either side of an interface and the unit normal vector of the interface 
    /// return the flux
    template<class FluxT>
    concept convective_numerical_flux = requires(
        FluxT flux,
        std::array<typename FluxT::value_type, FluxT::nv_comp> uL,
        std::array<typename FluxT::value_type, FluxT::nv_comp> uR,
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor< typename FluxT::value_type, FluxT::ndim > unit_normal
    ) {
        { flux(uL, uR, unit_normal) } -> std::same_as<std::array<typename FluxT::value_type, FluxT::nv_comp>>;
    };

    /// @brief the diffusive flux normal to the interface 
    ///
    /// given a single valued solution at an interface and the single valued gradient 
    /// compute the flux function for diffusion operators in the normal direction
    ///
    /// NOTE: this will be evaluated separately from ConvectiveNumericalFlux so do not include that here
    template<class FluxT>
    concept diffusion_flux = 
    requires(
        FluxT flux,
        std::array<typename FluxT::value_type, FluxT::nv_comp> u,
        std::mdspan<typename FluxT::value_type, std::extents<std::size_t, std::dynamic_extent, std::dynamic_extent>> gradu,
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor< typename FluxT::value_type, FluxT::ndim > unit_normal
    ) {
        { flux(u, gradu, unit_normal) } -> std::same_as<std::array<typename FluxT::value_type, FluxT::nv_comp>>;
    };

    /// @brief Diffusion fluxes can explicitly define the homogeineity tensor given a state u 
    /// This can be used for interface correction
    template< class FluxT>
    concept computes_homogeneity_tensor = requires(
        FluxT flux,
        std::array<typename FluxT::value_type, FluxT::nv_comp> u
    ) {
        { flux.homogeneity_tensor(u) } -> std::same_as< 
                NUMTOOL::TENSOR::FIXED_SIZE::Tensor<typename FluxT::value_type, FluxT::nv_comp, FluxT::ndim, FluxT::nv_comp, FluxT::ndim>>;
    };

    template<
        typename T,
        int ndim,
        physical_flux PFlux,
        convective_numerical_flux CFlux,
        diffusion_flux DiffusiveFlux,
        class ST_Info = std::false_type
    >
    class ConservationLawDDG {

        public:
        PFlux phys_flux;
        CFlux conv_nflux;
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
        static constexpr std::size_t nv_comp = PFlux::nv_comp;
        static constexpr std::size_t dnv_comp = PFlux::nv_comp;

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

        /// @brief human readable names for each vector component of the variables
        std::vector<std::string> field_names;

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
            PFlux&& physical_flux,
            CFlux&& convective_numflux,
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
            PFlux&& physical_flux,
            CFlux&& convective_numflux,
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
            elspan auto unkel,
            elspan auto res
        ) const -> void {
            static constexpr int neq = decltype(unkel)::static_extent();
            static_assert(neq == PFlux::nv_comp, "Number of equations must match.");
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;

            // basis function scratch space
            std::vector<T> dbdx_data(el.nbasis() * ndim);

            // solution scratch space
            std::array<T, neq> u;
            std::array<T, neq * ndim> gradu_data;

            // loop over the quadrature points
            for(int iqp = 0; iqp < el.nQP(); ++iqp){
                const QuadraturePoint<T, ndim> &quadpt = el.getQP(iqp);

                // calculate the jacobian determinant 
                auto J = el.jacobian(quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // get the basis functions and gradients in the physical domain
                auto bi = el.eval_basis_qp(iqp);
                auto gradxBi = el.eval_phys_grad_basis(quadpt.abscisse, J,
                        el.eval_grad_basis_qp(iqp), dbdx_data.data());

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
                            res[itest, ieq] += flux[ieq][jdim] * gradxBi[itest, jdim] * detJ * quadpt.weight;
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
            auto centroidL = elL.centroid();
            auto centroidR = elR.centroid();

            // Basis function scratch space 
            PhysDomainEvalStorage storageL{elL};
            PhysDomainEvalStorage storageR{elR};

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
                auto biL = trace.qp_evals_l[iqp].bi_span;
                auto biR = trace.qp_evals_r[iqp].bi_span;
                auto xiL = trace.transform_xiL(quadpt.abscisse);
                auto xiR = trace.transform_xiR(quadpt.abscisse);
                PhysDomainEval evalL{storageL, elL, xiL, trace.qp_evals_l[iqp]};
                PhysDomainEval evalR{storageR, elR, xiR, trace.qp_evals_r[iqp]};

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
                auto graduL = unkelL.contract_mdspan(evalL.phys_grad_basis, graduL_data.data());
                auto graduR = unkelR.contract_mdspan(evalR.phys_grad_basis, graduR_data.data());
                auto hessuL = unkelL.contract_mdspan(evalL.phys_hess_basis, hessuL_data.data());
                auto hessuR = unkelR.contract_mdspan(evalR.phys_hess_basis, hessuR_data.data());

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
                h_ddg = std::copysign(std::max(std::abs(h_ddg), std::numeric_limits<T>::epsilon()), h_ddg);
                
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
                        resL[itest, ieq] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                    }
                }
                for(int itest = 0; itest < elR.nbasis(); ++itest){
                    for(int ieq = 0; ieq < neq; ++ieq){
                        resR[itest, ieq] -= (fviscn[ieq] - fadvn[ieq]) * biR[itest];
                    }
                }

                // if applicable: apply the interface correction 
                if constexpr (computes_homogeneity_tensor<DiffusiveFlux>) {
                    auto gradBiL = trace.qp_evals_l[iqp].grad_bi_span;
                    auto gradBiR = trace.qp_evals_r[iqp].grad_bi_span;
                    if(sigma_ic != 0.0){
                        std::array<T, neq> interface_correction;
                        auto Gtensor = diff_flux.homogeneity_tensor(uavg);

                        T average_gradv[ndim];
                        for(int itest = 0; itest < elL.nbasis(); ++itest){
                            // get the average test function gradient
                            for(int idim = 0; idim < ndim; ++idim){
                                average_gradv[idim] = 0.5 * (
                                        gradBiL[itest, idim] + gradBiR[itest, idim] );
                            }

                            std::ranges::fill(interface_correction, 0);
                            for(int ieq = 0; ieq < neq; ++ieq){
                                for(int kdim = 0; kdim < ndim; ++kdim){
                                    for(int req = 0; req < neq; ++req){
                                        T jumpu_r = uR[req] - uL[req];
                                        for(int sdim = 0; sdim < ndim; ++sdim){
                                            
                                            resL[itest, ieq] -= 
                                                Gtensor[ieq][kdim][req][sdim] * unit_normal[kdim] 
                                                * average_gradv[sdim] * jumpu_r
                                                * quadpt.weight * sqrtg;
                                            resR[itest, ieq] -= 
                                                Gtensor[ieq][kdim][req][sdim] * unit_normal[kdim] 
                                                * average_gradv[sdim] * jumpu_r
                                                * quadpt.weight * sqrtg;
                                        }
                                    }
                                }
                            }

                        }
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
        [[gnu::noinline]]
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
            std::vector<T> gradbL_data(elL.nbasis() * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim> grad_ddg_data;
            std::array<T, neq * ndim * ndim> hessuL_data;
            std::array<T, neq * ndim * ndim> hessuR_data;
            auto centroidL = elL.centroid();

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
                        auto biL = trace.qp_evals_l[iqp].bi_span;

                        // get the gradients the physical domain
                        auto gradBiL = trace.eval_phys_grad_basis_l_qp(iqp, gradbL_data.data());
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
                        // std::array<T, neq> fadvn = conv_nflux(dirichlet_vals, dirichlet_vals, unit_normal);

                        // calculate the DDG distance
                        T h_ddg = 0; // uses distance to quadpt on boundary face
                        for(int idim = 0; idim < ndim; ++idim){
                            h_ddg += std::abs(unit_normal[idim] * 
                                (phys_pt[idim] - centroidL[idim])
                            );
                        }
                        h_ddg = std::copysign(std::max(std::abs(h_ddg), std::numeric_limits<T>::epsilon()), h_ddg);

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
                                resL[itest, ieq] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
                            }
                        }

                        // if applicable: apply the interface correction 
                        if constexpr (computes_homogeneity_tensor<DiffusiveFlux>) {
                            if(sigma_ic != 0.0){

                                std::array<T, neq> interface_correction;
                                auto Gtensor = diff_flux.homogeneity_tensor(uavg);

                                T average_gradv[ndim];
                                for(int itest = 0; itest < elL.nbasis(); ++itest){
                                    // get the average test function gradient
                                    for(int idim = 0; idim < ndim; ++idim){
                                        average_gradv[idim] = gradBiL[itest, idim];
                                    }

                                    std::ranges::fill(interface_correction, 0);
                                    for(int ieq = 0; ieq < neq; ++ieq){
                                        for(int kdim = 0; kdim < ndim; ++kdim){
                                            for(int req = 0; req < neq; ++req){
                                                T jumpu_r = dirichlet_vals[req] - uL[req];
                                                for(int sdim = 0; sdim < ndim; ++sdim){
                                                    
                                                    resL[itest, ieq] -= 
                                                        Gtensor[ieq][kdim][req][sdim] * unit_normal[kdim] 
                                                        * average_gradv[sdim] * jumpu_r
                                                        * quadpt.weight * sqrtg;
                                                }
                                            }
                                        }
                                    }

                                }
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
                        auto biL = trace.qp_evals_l[iqp].bi_span;

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
                                resL[itest, ieq] += (fviscn[ieq]) * biL[itest];
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
                    PhysDomainEvalStorage storageL{elL};
                    PhysDomainEvalStorage storageR{elR};

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
                        auto biL = trace.qp_evals_l[iqp].bi_span;
                        auto biR = trace.qp_evals_r[iqp].bi_span;
                        auto xiL = trace.transform_xiL(quadpt.abscisse);
                        auto xiR = trace.transform_xiR(quadpt.abscisse);
                        PhysDomainEval evalL{storageL, elL, xiL, trace.qp_evals_l[iqp]};
                        PhysDomainEval evalR{storageR, elR, xiR, trace.qp_evals_r[iqp]};

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
                        auto graduL = unkelL.contract_mdspan(evalL.phys_grad_basis, graduL_data.data());
                        auto graduR = unkelR.contract_mdspan(evalR.phys_grad_basis, graduR_data.data());
                        auto hessuL = unkelL.contract_mdspan(evalL.phys_hess_basis, hessuL_data.data());
                        auto hessuR = unkelR.contract_mdspan(evalR.phys_hess_basis, hessuR_data.data());

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
                        h_ddg = std::copysign(std::max(std::abs(h_ddg), std::numeric_limits<T>::epsilon()), h_ddg);
                        
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
                                resL[itest, ieq] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
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
                        auto biL = trace.qp_evals_l[iqp].bi_span;

                        // get the gradients the physical domain
                        auto gradBiL = trace.eval_phys_grad_basis_l_qp(iqp, gradbL_data.data());

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
                                resL[itest, ieq] += (fviscn[ieq] - fadvn[ieq]) * biL[itest];
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
            std::vector<T> bitrace(trace.nbasis_trace());
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
                auto biL = trace.eval_basis_l_qp(iqp);
                auto biR = trace.eval_basis_r_qp(iqp);
                trace.eval_trace_basis_qp(iqp, bitrace.data());
                auto gradBiL = trace.eval_phys_grad_basis_l_qp(iqp, gradbL_data.data());
                auto gradBiR = trace.eval_phys_grad_basis_r_qp(iqp, gradbR_data.data());

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

                if(trace.face->bctype != BOUNDARY_CONDITIONS::INTERIOR){
                    switch(trace.face->bctype){
                        case BOUNDARY_CONDITIONS::DIRICHLET:
                            {
                                if(trace.elL.elidx != trace.elR.elidx)
                                    std::cout << "warning: elements do not match" << std::endl;

                                // calculate the physical domain position
                                MATH::GEOMETRY::Point<T, ndim> phys_pt;
                                trace.face->transform(quadpt.abscisse, coord, phys_pt);

                                // Get the values at the boundary 
                                dirichlet_callbacks[trace.face->bcflag](phys_pt.data(), uR.data());

                            } break;
                        default:
                            for(int itest = 0; itest < trace.nbasis_trace(); ++itest){
                                for(int ieq = 0; ieq < neq; ++ieq)
                                    res[itest, ieq] = 0;
                            } return;
                    }
                }

                // get the physical flux on the left and right
                Tensor<T, neq, ndim> fluxL = phys_flux(uL, graduL);
                Tensor<T, neq, ndim> fluxR = phys_flux(uR, graduR);

                // calculate the jump in normal fluxes
                Tensor<T, neq> jumpflux{};
                for(int ieq = 0; ieq < neq; ++ieq) 
                    jumpflux[ieq] = dot(fluxR[ieq], unit_normal) - dot(fluxL[ieq], unit_normal);

                // scatter unit normal times interface conservation to residual
                for(int itest = 0; itest < trace.nbasis_trace(); ++itest){
                    for(int ieq = 0; ieq < neq; ++ieq){
                        T ic_res = jumpflux[ieq] * sqrtg * quadpt.weight;
                        // NOTE: multiplying by signed unit normal 
                        // adds directionality which can allow cancellation error with 
                        // V-shaped interface intersections
                        res[itest, ieq] -= ic_res * bitrace[itest]; 
                    }
                }
            }
        }
    };

    // Deduction Guides
    template<
        class T,
        int ndim,
        template<class T1, int ndim1> class PFlux,
        template<class T1, int ndim1> class CFlux, 
        template<class T1, int ndim1> class DiffusiveFlux
    >
    ConservationLawDDG(
        PFlux<T, ndim>&& physical_flux,
        CFlux<T, ndim>&& convective_numflux,
        DiffusiveFlux<T, ndim>&& diffusive_flux
    ) -> ConservationLawDDG<T, ndim, PFlux<T, ndim>, CFlux<T, ndim>, DiffusiveFlux<T, ndim>>;

    template<
        class T,
        int ndim,
        template<class T1, int ndim1> class PFlux,
        template<class T1, int ndim1> class CFlux, 
        template<class T1, int ndim1> class DiffusiveFlux,
        class ST_Info
    >
    ConservationLawDDG(
        PFlux<T, ndim>&& physical_flux,
        CFlux<T, ndim>&& convective_numflux,
        DiffusiveFlux<T, ndim>&& diffusive_flux,
        ST_Info st_info
    ) -> ConservationLawDDG<T, ndim, PFlux<T, ndim>, CFlux<T, ndim>, DiffusiveFlux<T, ndim>, ST_Info>;
}
