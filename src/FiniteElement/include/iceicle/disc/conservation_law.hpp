#pragma once

#include <type_traits>
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
namespace DISC {

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
            ICEICLE::LINALG::in_tensor auto gradu
        ) const noexcept -> Tensor<T, ndim> 
        {
            Tensor<T, ndim> flux_adv{};
            for(int idim = 0; idim < ndim; ++idim){
                flux_adv[idim] = 
                    a[idim] * u                  // linear convection
                    + 0.5 * b[idim] * SQUARED(u) // nonlinear convection
                    - mu * gradu[idim];          // diffusion
            }

        }


    };

    template<
        typename T,
        typename IDX,
        int ndim,
        int neq,
        class PhysicalFlux,
        class ConvectiveNumericalFlux,
        class DiffusiveNumericalFlux
    >
    struct ConservationLaw {

        // ============
        // = Typedefs =
        // ============

        using value_type = T;
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;



    };
}
