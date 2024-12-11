/// @file burgers.hpp
/// @author Gianni Absillis (gabsill@ncsu.edu)
///
/// @brief Burgers equation implementation

#pragma once

#include "iceicle/linalg/linalg_utils.hpp"
#include "Numtool/MathUtils.hpp"
#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/matrixT.hpp"

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
        int _ndim
    >
    struct BurgersFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        using value_type = T;

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;

        /// @brief the number of dimensions
        static constexpr int ndim = _ndim;

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
    template <class T, int _ndim>
    struct BurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:
        static constexpr int ndim = _ndim;
        using value_type = T;

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
//            T lambdaL = 0;
//            T lambdaR = 0;
//            for(int idim = 0; idim < ndim; ++idim){
//                lambdaL += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL[0]);
//                lambdaR += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR[0]);
//            }
//
//            T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
//            T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));
//            T fadvn = uL[0] * lambda_l_plus + uR[0] * lambda_r_plus;
//            return std::array<T, nv_comp>{fadvn};
            T u_avg = 0.5 * (uL[0] + uR[0]);
            T P = 0;
            for(int idim = 0; idim < ndim; ++idim){
                P += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * u_avg);
            }

            T u_upwind = (P > 0) ? uL[0] : uR[0];

            std::array<T, nv_comp> fadvn = {0};
            for(int idim = 0; idim < ndim; ++idim){
                fadvn[0] += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * u_upwind);
            }
            fadvn[0] *= u_upwind; // factor out u from the flux to reduce computation

            return fadvn;
        }
    };
    template<class T, int ndim>
    BurgersUpwind(BurgersCoefficients<T, ndim>) -> BurgersUpwind<T, ndim>;

    /// @brief diffusive flux in burgers equation
    template <class T, int _ndim>
    struct BurgersDiffusionFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        static constexpr int ndim = _ndim;

        /// @the real number type
        using value_type = T;

        /// @brief the number of vector components
        static constexpr std::size_t nv_comp = 1;

        /// @brief get the number of equations
        static constexpr 
        auto neq() -> std::size_t { return nv_comp; }

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

        /// @brief compute the homogeneity tensor
        /// which is just the identity times the viscosity
        inline constexpr 
        auto homogeneity_tensor(
            std::array<T, nv_comp> u
        ) const noexcept -> Tensor<T, nv_comp, ndim, nv_comp, ndim>
        {
            Tensor<T, nv_comp, ndim, nv_comp, ndim> G;
            for(int kdim = 0; kdim < ndim; ++kdim){
                for(int sdim = 0; sdim < ndim; ++sdim){
                    if(sdim == kdim){
                        G[0][kdim][0][sdim] = coeffs.mu;
                    } else {
                        G[0][kdim][0][sdim] = 0.0;
                    }
                }
            }
            return G;
        }
    };



    template<class T, int ndim>
    BurgersDiffusionFlux(BurgersCoefficients<T, ndim>) -> BurgersDiffusionFlux<T, ndim>;

    /// ==============================
    /// = Spacetime Burgers Equation =
    /// ==============================

    template<
        class T,
        int _ndim
    >
    struct SpacetimeBurgersFlux {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        /// @brief the real number type
        using value_type = T;

        /// @brief the number of dimensions
        static constexpr int ndim = _ndim;

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
            return (reference_length * cfl) / 
                (coeffs.mu / std::max(std::numeric_limits<T>::epsilon(), reference_length) + aref);
        }
    };
    template<class T, int ndim_space>
    SpacetimeBurgersFlux(BurgersCoefficients<T, ndim_space>) -> SpacetimeBurgersFlux<T, ndim_space+1>;

    /// @brief upwind numerical flux for the convective flux in spacetime burgers equation
    template <class T, int _ndim>
    struct SpacetimeBurgersUpwind {
        private:
        template<class T2, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

        public:

        static constexpr int ndim = _ndim;
        using value_type = T;

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
            // Dolejsi, Feistaur Discontinuous Galerkin Method pp. 121
            //
            // TODO: increase efficiency by computing p from lambdaL + lambdaR
            T u_avg = 0.5 * (uL[0] + uR[0]);
            T P = 0;
            for(int idim = 0; idim < ndim_space; ++idim){
                P += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * u_avg);
            }
            P += unit_normal[idim_time];

            T u_upwind = (P > 0) ? uL[0] : uR[0];

            std::array<T, nv_comp> fadvn = {0};
            for(int idim = 0; idim < ndim_space; ++idim){
                fadvn[0] += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * u_upwind);
            }
            fadvn[0] += unit_normal[idim_time];
            fadvn[0] *= u_upwind; // factor out u from the flux to reduce computation

            return fadvn;

//            T lambdaL = 0;
//            T lambdaR = 0;
//            for(int idim = 0; idim < ndim_space; ++idim){
//                lambdaL += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uL[0]);
//                lambdaR += unit_normal[idim] * (coeffs.a[idim] + 0.5 * coeffs.b[idim] * uR[0]);
//            }
//            // time component
//            lambdaL += unit_normal[idim_time];
//            lambdaR += unit_normal[idim_time];
//
//            T lambda_l_plus = 0.5 * (lambdaL + std::abs(lambdaL));
//            T lambda_r_plus = 0.5 * (lambdaR - std::abs(lambdaR));
//            T fadvn = uL[0] * lambda_l_plus + uR[0] * lambda_r_plus;
//            return std::array<T, nv_comp>{fadvn};

            // TODO: examine Osher flux (see Toro Riemann Solvers and Numerical Methods pp.384)
        }
    };

    template<class T, int ndim_space>
    SpacetimeBurgersUpwind(BurgersCoefficients<T, ndim_space>) -> SpacetimeBurgersUpwind<T, ndim_space+1>;

    template<class T, int _ndim>
    struct SpacetimeBurgersDiffusion {

        static constexpr int ndim = _ndim;
        using value_type = T;

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
}
