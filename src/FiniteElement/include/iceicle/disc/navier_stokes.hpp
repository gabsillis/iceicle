#include "iceicle/anomaly_log.hpp"
#include "iceicle/geometry/face.hpp"
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/linalg/linalg_utils.hpp>

namespace iceicle {
    namespace navier_stokes {

        /// @brief all of the flow state quantities necessary to compute fluxes
        template<class T, int ndim>
        struct FlowState {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;

            /// @brief the density of the fluid
            T density;

            /// @brief the velocity of the fluid
            Vector velocity;

            /// @brief the momentum of the fluid 
            Vector momentum;

            /// @brief the square magnitude of the velocity (v * v)
            T velocity_magnitude_squared;

            /// @brief the pressure
            T pressure;

            /// @brief speed of sound
            T csound;

            /// the energy
            T rhoe;

            /// The temperature
            T temp;

            /// The viscosity
            T mu;
        };

        /// @brief the gradients of flow state quantities
        template< class T, int ndim >
        struct FlowStateGradients {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;

            /// @brief velocity gradients du_i / dx_j
            Tensor velocity_gradient;

            /// @brief gradient of the temperature
            Vector temp_gradient;

            /// @brief the shear stress tensor
            Tensor tau;
        };
        
        /// @brief Handles the physics relations and quantities for thermodynamic state 
        ///
        /// @tparam T the real number type 
        /// @tparam ndim the number of dimensions 
        template<class T, int ndim>
        struct Physics {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;

            /// @brief floor for the pressure so we don't have negative pressures
            static constexpr T MIN_PRESSURE = 1e-8;

            /// @brief index of density
            static constexpr int irho = 0;

            /// @brief index of x momentum
            static constexpr int irhou = 1;

            /// @brief index of energy density
            static constexpr int irhoe = ndim + 1;


            /// @brief the ratio of specific heats ( cp / cv )
            T gamma = 1.4;

            /// @brief Specific gas constant, difference of spcific heats  ( cp - cv )
            T R = 287.052874; // J / (kg * K)

            /// @brief Prandtl number 
            T Pr = 0.72;

            /// @brief reference viscosity
            T mu_0 = 1.716e-5; // Pa * s

            // @brief reference temperature
            T T_0 = 273.1; // K

            /// @brief Sutherlands law temperature
            T T_s = 110.5; // K

            std::vector<T> isothermal_temperatures{};


            /// @brief given the conservative variables, compute the flow state 
            /// @param u the conservative variables
            inline constexpr
            auto calc_flow_state(std::array<T, nv_comp> u) const noexcept
            -> FlowState<T, ndim> {
                T density = u[0];

                // compute the square velocity magnitude 
                Vector v;
                T vv = 0;
                for(int idim = 0; idim < ndim; ++idim){
                    v[idim] = u[irhou + idim] / density;
                    vv += v[idim] * v[idim];
                    for(int jdim = 0; jdim < ndim; ++jdim){

                    }
                }

                // copy the momentum 
                Vector momentum;
                for(int idim = 0; idim < ndim; ++idim)
                    momentum[idim] = u[idim + 1];

                // compute the pressure
                T rhoe = u[irhoe];
                T pressure = std::max(MIN_PRESSURE, (gamma - 1) * (rhoe - 0.5 * density * vv));

                // compute the speed of sound
                T csound = std::sqrt(gamma * pressure / density);

                // compute the temperature from ideal gas law
                T temp = pressure / (density * R);

                // compute the viscosity using Sutherland's Law
                T mu = mu_0 * (T_0 + T_s) / (temp + T_s) * std::pow(temp / T_s, 1.5);

                return FlowState<T, ndim>{density, v, momentum, vv, pressure, csound, rhoe, temp, mu};
            }

            // @brief given the conservative variable gradients and current flow state 
            // compute the gradients of the flow state 
            //
            // @param state the flow state 
            // @param gradu the gradients of the conservative variables 
            // NOTE: the output gradients will be with respect to the same coordinates as gradu 
            //
            // @return the gradients of the flow state
            inline constexpr 
            auto calc_flow_state_gradients(const FlowState<T, ndim>& state, linalg::in_tensor auto gradu) const noexcept 
            -> FlowStateGradients<T, ndim>
            {
                Tensor grad_vel;

                // calculate velocity gradients
                for(int idim = 0; idim < ndim; ++idim){
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        grad_vel[idim][jdim] = gradu[irhou + idim, jdim] / state.density
                            - state.velocity[idim] / state.density * gradu[irho, jdim];
                    }
                }

                // calculate temperature gradient 
                Vector grad_temp;
                for(int jdim = 0; jdim < ndim; ++jdim){
                    grad_temp[jdim] = ( 
                            (gamma - 1) / R * gradu[irhoe, jdim]
                            - gradu[irho, jdim] * state.temp
                    ) / state.density;

                }

                // calculate shear stress tensor
                Tensor tau;
                T partial_uk_partial_xk = 0;
                for(int kdim = 0; kdim < ndim; ++kdim){
                    partial_uk_partial_xk += grad_vel[kdim][kdim];
                }

                for(int idim = 0; idim < ndim; ++idim){
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        tau[idim][jdim] = state.mu * (grad_vel[idim][jdim] + grad_vel[jdim][idim]);
                    }
                }

                // kronecker
                for(int ij = 0; ij < ndim; ++ij){
                    tau[ij][ij] -= 2.0 / 3.0 * state.mu * partial_uk_partial_xk;
                }

                return FlowStateGradients<T, ndim>{grad_vel, grad_temp, tau};
            }
        };

        /// @brief Van Leer flux
        /// implementation reference: http://www.chimeracfd.com/programming/gryphon/fluxvanleer.html
        template< class T, int _ndim >
        struct VanLeer {

            static constexpr int ndim = _ndim;
            using value_type = T;

            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;

            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;
            static constexpr int neq = nv_comp;

            Physics<T, ndim> physics;

            inline constexpr
            auto operator()(
                std::array<T, nv_comp> uL,
                std::array<T, nv_comp> uR,
                Vector unit_normal
            ) const noexcept -> std::array<T, neq> {
                using namespace NUMTOOL::TENSOR::FIXED_SIZE;
                FlowState<T, ndim> stateL = physics.calc_flow_state(uL);
                FlowState<T, ndim> stateR = physics.calc_flow_state(uR);

                // normal velocity 
                T vnormalL = dot(stateL.velocity, unit_normal);
                T vnormalR = dot(stateR.velocity, unit_normal);

                // normal mach numbers 
                T machL = vnormalL / stateL.csound;
                T machR = vnormalR / stateR.csound;

                std::array<T, neq> flux;

                // compute positive fluxes and add contribution
                if(machL > 1){
                    // supersonic left 
                    flux[0] = stateL.density * vnormalL;
                    for(int idim = 0; idim < ndim; ++idim)
                        flux[1 + idim] = stateL.momentum[idim] * vnormalL + stateL.pressure * unit_normal[idim];
                    flux[ndim + 1] = vnormalL * (stateL.rhoe + stateL.pressure);
                } else if(machL < -1) {
                    std::ranges::fill(flux, 0);
                } else {
                    T fmL = stateL.density * stateL.csound * SQUARED(machL + 1) / 4.0;
                    flux[0] = fmL;
                    for(int idim = 0; idim < ndim; ++idim){
                        flux[1 + idim] = fmL * (
                            stateL.velocity[idim] 
                            + unit_normal[idim] * (-vnormalL + 2 * stateL.csound) / physics.gamma 
                        );
                    }
                    flux[ndim + 1] = fmL * (
                        (stateL.velocity_magnitude_squared - SQUARED(vnormalL)) / 2
                        + SQUARED( (physics.gamma - 1) * vnormalL + 2 * stateL.csound)
                        / (2 * (SQUARED(physics.gamma) - 1))
                    );
                }

                // compute negative fluxes and add contribution
                if(machR < -1){
                    flux[0] += stateR.density * vnormalR;
                    for(int idim = 0; idim < ndim; ++idim)
                        flux[1 + idim] += stateR.momentum[idim] * vnormalR + stateR.pressure * unit_normal[idim];
                    flux[ndim + 1] += vnormalR * (stateR.rhoe + stateR.pressure);
                } else if (machR <= 1) {
                    T fmR = -stateR.density * stateR.csound * SQUARED(machR - 1) / 4.0;
                    flux[0] += fmR;
                    for(int idim = 0; idim < ndim; ++idim){
                        flux[1 + idim] += fmR * (
                            stateR.velocity[idim] 
                            + unit_normal[idim] * (-vnormalR - 2 * stateR.csound) / physics.gamma 
                        );
                    }
                    flux[ndim + 1] += fmR * (
                        (stateR.velocity_magnitude_squared - SQUARED(vnormalR)) / 2
                        + SQUARED( (physics.gamma - 1) * vnormalR - 2 * stateR.csound)
                        / (2 * (SQUARED(physics.gamma) - 1))
                    );
                } // else 0
                
                return flux;
            }
        };

        // @brief the physical flux for navier stokes equations
        // @tparam T the real number type
        // @tparam _ndim the number of dimensions 
        // @tparam euler if true, limits computations for euler equations (don't compute viscous terms)
        template< class T, int _ndim, bool euler = true >
        struct Flux {

            template<class T2, std::size_t... sizes>
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, _ndim>;

            /// @brief the real number type
            using value_type = T;

            /// @brief the number of dimensions
            static constexpr int ndim = _ndim;

            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;

            /// @brief number of equations
            static constexpr int neq = nv_comp;

            /// @brief index of density equation
            static constexpr int irho = 0;

            /// @brief index of x momentum equation
            static constexpr int irhou = 1;

            /// @brief index of energy density equation
            static constexpr int irhoe = ndim + 1;

            Physics<T, ndim> physics;

            mutable T lambda_max = 0.0;

            inline constexpr 
            auto operator()(
                std::array<T, nv_comp> u,
                linalg::in_tensor auto gradu
            ) const noexcept -> Tensor<T, neq, ndim> {

                FlowState<T, ndim> state = physics.calc_flow_state(u);

                lambda_max = state.csound + std::sqrt(state.velocity_magnitude_squared);

                Tensor<T, neq, ndim> flux;
                // loop over the flux direction j
                for(int jdim = 0; jdim < ndim; ++jdim) {
                    flux[0][jdim] = u[1 + jdim];
                    for(int idim = 0; idim < ndim; ++idim)
                        flux[1 + idim][jdim] = state.momentum[idim] * state.velocity[jdim];
                    flux[1 + jdim][jdim] += state.pressure;
                    flux[1 + ndim][jdim] = state.velocity[jdim] * (state.rhoe + state.pressure);
                }

                if(!euler) {
                    // get gradients of state
                    FlowStateGradients<T, ndim> state_grads{physics.calc_flow_state_gradients(state, gradu)};

                    // calculate the heat conductivity coefficient
                    T kappa = physics.gamma * physics.R * state.mu / (physics.Pr * ( physics.gamma - 1));

                    // subtract viscous fluxes
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            flux[irhou + idim][jdim] -= state_grads.tau[idim][jdim];
                            flux[irhoe][jdim] -= state.velocity[idim] * state_grads.tau[idim][jdim];
                        }
                        flux[irhoe][idim] -= state_grads.temp_gradient[idim];
                    }
                }
                
                return flux;
            }

            inline constexpr 
            auto apply_bc(
                std::array<T, neq> uL,
                linalg::in_tensor auto graduL,
                Vector unit_normal,
                BOUNDARY_CONDITIONS bctype,
                int bcflag
            ) const noexcept {
                using namespace NUMTOOL::TENSOR::FIXED_SIZE;

                // outputs
                std::array<T, neq> uR{};
                Tensor<T, neq, ndim> graduR{};

                switch(bctype) {
                    case BOUNDARY_CONDITIONS::WALL_GENERAL:
                        if(euler){
                            goto euler_wall_bc_tag;
                        } else {
                            goto ns_wall_bc_tag;
                        }
                        break;
                    case BOUNDARY_CONDITIONS::SLIP_WALL: 
euler_wall_bc_tag:
                    {
                        FlowState<T, ndim> stateL = physics.calc_flow_state(uL);

                        // density and energy are the same
                        uR[0] = uL[0];
                        uR[1 + ndim] = uL[1 + ndim];

                        // flip velocity over the normal
                        T mom_n = dot(stateL.momentum, unit_normal);
                        Vector mom_R = stateL.momentum;
                        axpy(-2 * mom_n, unit_normal, mom_R);

                        for(int idim = 0; idim < ndim; ++idim){
                            uR[1 + idim] = mom_R[idim];
                        }

                        // exterior state gradients equal interor state gradients 
                        linalg::copy(graduL, linalg::as_mdspan(graduR));

                    } break;
                    case BOUNDARY_CONDITIONS::NO_SLIP_ISOTHERMAL:
ns_wall_bc_tag:
                    {
                        FlowState<T, ndim> stateL = physics.calc_flow_state(uL);

                        // density is the same
                        uR[irho] = stateL.density;

                        // velocity, and therefore momentum, is zero
                        for(int idim = 0; idim < ndim; ++idim){
                            uR[irhou + idim] = 0;
                        }

                        // compute energy from Twall
                        T Twall = physics.isothermal_temperatures[bcflag];
                        T pressure = stateL.density * physics.R * Twall;
                        // NOTE: kinetic energy term is zero because momentum is zero
                        T E = pressure / (physics.gamma - 1) / stateL.density;
                        uR[irhoe] = stateL.density * E;

                        // exterior state gradients equal interor state gradients 
                        linalg::copy(graduL, linalg::as_mdspan(graduR));
                    } break;
                    default:
                    util::AnomalyLog::log_anomaly(util::Anomaly{"Unsupported BC",
                            util::general_anomaly_tag{}});
                }

                return std::pair{uR, graduR};
            }

            inline constexpr 
            auto dt_from_cfl(T cfl, T reference_length) const noexcept -> T {
                return (reference_length * cfl) / (lambda_max);
            }
        };

        template< class T, int _ndim, bool euler = true >
        struct DiffusionFlux {

            static constexpr int ndim = _ndim;
            using value_type = T;

            template<class T2, std::size_t... sizes>
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;

            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;
            static constexpr int neq = nv_comp;

            /// @brief index of density equation
            static constexpr int irho = 0;

            /// @brief index of x momentum equation
            static constexpr int irhou = 1;

            /// @brief index of energy density equation
            static constexpr int irhoe = ndim + 1;

            Physics<T, ndim> physics;


            inline constexpr
            auto operator()(
                std::array<T, nv_comp> u,
                linalg::in_tensor auto gradu,
                Tensor<T, ndim> unit_normal
            ) const noexcept -> std::array<T, neq>
            {
                std::array<T, neq> flux;
                std::ranges::fill(flux, 0.0);

                if(!euler) {

                    // compute the state
                    FlowState<T, ndim> state = physics.calc_flow_state(u);

                    // get gradients of state
                    FlowStateGradients<T, ndim> state_grads{physics.calc_flow_state_gradients(state, gradu)};

                    // calculate the heat conductivity coefficient
                    T kappa = physics.gamma * physics.R * state.mu / (physics.Pr * ( physics.gamma - 1));

                    // subtract viscous fluxes
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            flux[irhou + idim] += state_grads.tau[idim][jdim] * unit_normal[jdim];
                            flux[irhoe] += state.velocity[idim] * state_grads.tau[idim][jdim] * unit_normal[jdim];
                        }
                        flux[irhoe] += state_grads.temp_gradient[idim] * unit_normal[idim];
                    }
                }
                return flux;
            }

            /// @brief compute the diffusive flux normal to the interface 
            /// given the prescribed normal gradient
            inline constexpr 
            auto neumann_flux(
                std::array<T, nv_comp> gradn
            ) const noexcept -> std::array<T, neq> {
                std::array<T, nv_comp> flux{};
                std::ranges::fill(flux, 0.0);
                return flux;
            }
        };
    }
}
