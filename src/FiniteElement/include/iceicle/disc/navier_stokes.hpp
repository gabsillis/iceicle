#pragma once
#include "Numtool/MathUtils.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/geometry/face.hpp"
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/linalg/linalg_utils.hpp>

namespace iceicle {
    namespace navier_stokes {

        /// @brief the variable sets
        enum class VARSET {
            CONSERVATIVE,
            RHO_U_T,
            RHO_U_P, /// FUN3D primitive
        };

        /// @brief The thermodynamic state of the fluid
        template<class real, int ndim>
        struct ThermodynamicState {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, ndim>;
            static constexpr int nv_comp = ndim + 2;

            /// @brief the density of the fluid
            real rho;

            /// @brief the momentum of the fluid  (rhou)
            Vector momentum;

            /// @brief the energy density including kinetic energy
            real rhoE;

            /// @brief the ratio of specific heats 
            real gamma;

            /// @brief the specific heat under constant pressure process
            real cp;

            /// @brief the velocity of the fluid
            Vector velocity;

            /// The temperature
            real T;

            /// @brief the square magnitude of the velocity (v * v)
            real vv;

            ///@brief the pressure
            real p;

            /// @brief speed of sound
            real csound;

            /// the specific internal energy 
            real e;

            /// The specific total energy (included kinetic energy)
            real E;

            /// @brief the enthalpy (E + p / rho)
            real H;

            /// @brief get the state vector of the given variable set 
            /// @tparam variable_set which variables to get 
            /// @return a std::array with the given variable_set
            template<VARSET variable_set>
            [[nodiscard]] inline constexpr
            auto get_state_vector() const noexcept -> std::array<real, nv_comp>
            {
                std::array<real, nv_comp>out;
                if constexpr(variable_set == VARSET::CONSERVATIVE) {
                    out[0] = rho;
                    for(int idim = 0; idim < ndim; ++idim)
                        out[1 + idim] = momentum[idim];
                    out[1 + ndim] = rhoE;
                } else if constexpr(variable_set == VARSET::RHO_U_T) {
                    out[0] = rho;
                    for(int idim = 0; idim < ndim; ++idim)
                        out[1 + idim] = velocity[idim];
                    out[1 + ndim] = T;
                } else { // VARSET::RHO_U_P
                    out[0] = rho;
                    for(int idim = 0; idim < ndim; ++idim)
                        out[1 + idim] = velocity[idim];
                    out[1 + ndim] = p;

                }
                return out;
            }
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
        };

        /// @brief The free stream state 
        template< class T, int ndim >
        struct FreeStream {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;

            /// free stream density
            T rho_inf;

            /// free stream velocity magnitude
            T u_inf;

            /// free stream velocity unit direction 
            Vector u_direction;

            /// free stream temperature
            T temp_inf;

            /// free stream viscosity
            T mu_inf;

            /// reference length
            T l_ref;
        };
        
        /// @brief the set of reference parameters to nondimensionalize the flow quantities with
        /// Default initialization is to set all parameters to unity 
        /// which will result in the same scaling as the dimensional form.
        template<class real>
        struct ReferenceParameters {
            /// @bref reference time 
            real t = 1;

            /// @brief reference length 
            real l = 1;

            /// @brief reference density
            real rho = 1;

            /// @brief reference velocity 
            real u = 1;

            /// @brief reference internal energy
            real e = 1;

            /// @brief reference pressure
            real p = 1;

            /// @brief reference temperature
            real T = 1;

            /// @brief reference viscosity 
            real mu = 1;
        };

        /// @brief construct reference parameters from free stream parameters 
        /// Chooses reference pressure and energy to ensure that Euler number and 
        /// energy coefficient are unity 
        /// Chooses reference time so that Strouhal number is unity
        ///
        /// 4.14.2 in Katate Masatsuka I do like CFD, Vol. 1
        template<class real>
        [[nodiscard]] inline constexpr
        auto FreeStreamReference(
                real rho_inf,    /// @brief free_stream density
                real u_inf,      /// @brief the free stream velocity magnitude
                real temp_inf,   /// @brief the free stream temperature
                real mu_inf,     /// @brief the free stream viscosity
                real l_ref = 1.0 /// @brief the reference length (default = 1)
        ) -> ReferenceParameters<real>
        {
            return ReferenceParameters<real>{
                .t = l_ref / u_inf,
                .l = l_ref,
                .rho = rho_inf,
                .u = u_inf,
                .e = u_inf * u_inf,
                .p = rho_inf * u_inf * u_inf
                .T = 1,
                .mu = 1
            };
        }

        template< class real>
        struct Nondimensionalization {
            /// @brief reference Reynolds number
            real Re;

            /// @brief the reference Euler number
            real Eu;

            /// @brief the reference Strouhal number
            real Sr;

            /// @brief the dimensionless energy coefficient
            real e_coeff;
        };

        /// @brief Given the reference parameters and specific gas constant 
        /// Compute the derived dimensionless quantities necessary for NS 
        ///
        /// @param ref the reference parameters 
        template< class real >
        auto create_nondim(const ReferenceParameters<real>& ref)
        -> Nondimensionalization<real>
        {
            real Re = ref.rho * ref.u * ref.l / ref.mu;
            real Eu = ref.p / (ref.rho * ref.u * ref.u);
            real Sr = ref.l / (ref.u * ref.t);
            real e_coeff = ref.u * ref.u / ref.e;
            return Nondimensionalization{Re, Eu, Sr, e_coeff};
        };

        /// @brief Dimensional form of Sutherlands Law for temperature dependence of viscosity
        /// Temperatures are in degrees Kelvin (K)
        template< class real >
        struct Sutherlands {
            /// @brief reference viscosity
            real mu_0 = 1.716e-5; // Pa * s

            // @brief reference temperature
            real T_0 = 273.1; // K

            /// @brief Sutherlands law temperature
            real T_s = 110.5; // K
           
            [[nodiscard]] inline constexpr
            auto operator()(real temp) const noexcept
            -> real {
                return mu_0 * std::pow(temp / T_0, 1.5) * (T_0 + T_s) / (temp + T_s);
            }
        };

        /// @brief dimensionless form of Sutherlands law that allows
        /// a mismatch of reference temperature and reference viscosity
        template< class real >
        struct DimensionlessSutherlands {
            /// @brief reference viscosity
            real mu_0 = 1.716e-5; // Pa * s

            // @brief reference temperature
            real T_0 = 273.1; // K

            /// @brief Sutherlands law temperature
            real T_s = 110.5; // K

            real mu_ratio; // @brief ratio mu_0 to reference viscosity

            real T_0_ratio; /// @brief ratio of T_0 to T_ref

            real T_s_ratio; // T_s / T_ref

            /// @brief construct coefficients based on reference parameters
            DimensionlessSutherlands(ReferenceParameters<real> ref)
            : mu_ratio{mu_0 / ref.mu}, T_0_ratio{T_0 / ref.T}, T_s_ratio{T_s / ref.T}{}

            [[nodiscard]] inline constexpr
            auto operator()(real That) const noexcept
            -> real {
                return mu_ratio * std::pow(That / T_0_ratio, 1.5)
                    * (T_0_ratio + T_s_ratio) / (That + T_s_ratio);
            }
        };

        /// @brief a type is an equation of state if it 
        /// calculates ThermodynamicState from a given state vector and 
        /// reference parameters
        ///
        /// this represents the closure of thermodynamic state
        template< class eos_t >
        concept is_eos = requires(
            const eos_t eos,
            const std::array<typename eos_t::real_t, eos_t::nv_comp> u,
            const ThermodynamicState<typename eos_t::real_t, eos_t::ndim> state,
            const std::mdspan<typename eos_t::real_t, std::extents<int, (std::size_t) eos_t::nv_comp,
                (std::size_t) eos_t::ndim> > gradu,
            const ReferenceParameters<typename eos_t::real_t> ref,
            const Nondimensionalization<typename eos_t::real_t> nondim
        ) {
            { eos.template calc_thermo_state<VARSET::CONSERVATIVE>(u, ref, nondim) }
                -> std::same_as<ThermodynamicState<typename eos_t::real_t, eos_t::ndim>>;
            { eos.template calc_thermo_state<VARSET::RHO_U_T>(u, ref, nondim) }
                -> std::same_as<ThermodynamicState<typename eos_t::real_t, eos_t::ndim>>;
            { eos.template calc_thermo_state<VARSET::RHO_U_P>(u, ref, nondim) }
                -> std::same_as<ThermodynamicState<typename eos_t::real_t, eos_t::ndim>>;
            { eos.template calc_state_gradients<VARSET::CONSERVATIVE>(state, gradu, ref, nondim) }
                -> std::same_as<FlowStateGradients<typename eos_t::real_t, eos_t::ndim>>;
            { eos.template calc_state_gradients<VARSET::RHO_U_T>(state, gradu, ref, nondim) }
                -> std::same_as<FlowStateGradients<typename eos_t::real_t, eos_t::ndim>>;
            { eos.template calc_state_gradients<VARSET::RHO_U_P>(state, gradu, ref, nondim) }
                -> std::same_as<FlowStateGradients<typename eos_t::real_t, eos_t::ndim>>;
        };

        template< class real, int _ndim >
        struct CaloricallyPerfectEoS {
            using real_t = real;
            static constexpr int ndim = _ndim;
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, ndim>;
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, ndim, ndim>;
            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;

            /// @brief floor for the pressure so we don't have negative pressures
            static constexpr real MIN_PRESSURE = 1e-12;

            static constexpr real MIN_DENSITY = 1e-14;


            real gamma = 1.4;

            real Rgas = 287.052874;

            /// @brief convert from one variable set to another 
            /// @tparam From the variable set given 
            /// @tparam To the variable set to go to 
            /// @param u_in the values in the From variable set 
            /// @return the values converted to the To variable set
            template<VARSET From, VARSET To>
            [[nodiscard]] inline constexpr 
            auto convert_varset(
                const std::array<real, nv_comp> u_in,
                const ReferenceParameters<real> ref,
                const Nondimensionalization<real> nondim
            )
            const noexcept -> std::array<real, nv_comp> 
            {
                if constexpr(From == To) { return u_in; }
                if constexpr (From == VARSET::CONSERVATIVE && To == VARSET::RHO_U_T){
                    static constexpr int irho = 0;
                    static constexpr int irhou = 1;
                    static constexpr int irhoe = ndim + 1;
                    std::array<real, nv_comp> u_out;
                    real rho = u_in[0];
                    real rhoE = u_in[irhoe];
                    real E = rhoE / rho;
                    real vv = 0;
                    Vector velocity;
                    for(int idim = 0; idim < ndim; ++idim) {
                        velocity[idim] = u_in[irhou + idim] / rho;
                        vv += velocity[idim] * velocity[idim];
                    }
                    real e = E - 0.5 * nondim.e_coeff * vv;
                    real p = (gamma - 1) / (nondim.e_coeff * nondim.Eu) * rho * e;
                    real cp = (gamma) / (gamma - 1) * Rgas;

                    // temperature coefficient
                    real T_coeff = cp * ref.T * ref.rho / ref.p;
                    real T = p / rho * gamma / (gamma - 1) / T_coeff;

                    u_out[0] = rho;
                    for(int idim = 0; idim < ndim; ++idim){
                        u_out[1 + idim] = u_in[1 + idim] / u_in[0];
                    }
                    u_out[1 + ndim] = T;
                    return u_out;
                }
                // OPTIMIZE: the default method computes all the quantities
                // specialize below
                ThermodynamicState<real, ndim> state = calc_thermo_state<From>(u_in, ref, nondim);
                return state.template get_state_vector<To>();
            }

           
            /// @brief given a state vector of a given variable set 
            /// compute the dimensionless thermodynamic quantities
            /// @tparam variable_set which variable set is stored in the u array 
            /// @param u the dimensionless state vector of the given variable set 
            /// @param ref the reference parameters
            /// @param nondim reference dimensionless parameters
            template<VARSET variable_set>
            [[nodiscard]] inline constexpr
            auto calc_thermo_state(
                const std::array<real, nv_comp> u,
                const ReferenceParameters<real> ref,
                const Nondimensionalization<real> nondim
            ) const noexcept -> ThermodynamicState<real, ndim>
            {
                if constexpr(variable_set == VARSET::CONSERVATIVE) {
                    static constexpr int irho = 0;
                    static constexpr int irhou = 1;
                    static constexpr int irhoe = ndim + 1;

                    real rho = std::max(MIN_DENSITY, u[0]);
                    Vector momentum, velocity;
                    real vv = 0;
                    for(int idim = 0; idim < ndim; ++idim) {
                        momentum[idim] = u[irhou + idim];
                        velocity[idim] = u[irhou + idim] / rho;
                        vv += velocity[idim] * velocity[idim];
                    }
                    real rhoE = u[irhoe];
                    real cp = (gamma) / (gamma - 1) * Rgas;

                    // temperature coefficient
                    real T_coeff = cp * ref.T * ref.rho / ref.p;

                    real E = rhoE / rho;
                    real e = E - 0.5 * nondim.e_coeff * vv;

                    real p = std::max(MIN_PRESSURE, 
                            (gamma - 1) / (nondim.e_coeff * nondim.Eu) * rho * e);

                    real T = p / rho * gamma / (gamma - 1) / T_coeff;

                    real csound = sqrt((gamma * nondim.Eu * p) / rho);

                    real H = E + p / rho;
                    return ThermodynamicState<real, ndim>{
                        .rho = rho,
                        .momentum = momentum,
                        .rhoE = rhoE,
                        .gamma = gamma,
                        .cp = cp,
                        .velocity = velocity,
                        .T = T,
                        .vv = vv,
                        .p = p,
                        .csound = csound,
                        .e = e,
                        .E = e,
                        .H = H
                    };
                } else if constexpr (variable_set == VARSET::RHO_U_T) {
                    static constexpr int irho = 0;
                    static constexpr int iu = 1;
                    static constexpr int iT = ndim + 1;

                    real rho = std::max(MIN_DENSITY, u[irho]);
                    Vector momentum, velocity;
                    real vv = 0;
                    for(int idim = 0; idim < ndim; ++idim) {
                        velocity[idim] = u[iu + idim];
                        momentum[idim] = u[iu + idim] * rho;
                        vv += velocity[idim] * velocity[idim];
                    }
                    real T = u[iT];
                    real cp = (gamma) / (gamma - 1) * Rgas;

                    // temperature coefficient
                    real T_coeff = cp * ref.T * ref.rho / ref.p;
                    real p = std::max(MIN_PRESSURE, 
                            rho * (gamma - 1) / gamma * T_coeff * T);
                    real e = p * nondim.e_coeff * nondim.Eu / (gamma - 1) / rho;
                    real E = e + 0.5 * nondim.e_coeff * vv;
                    real rhoE = rho * E;
                    real csound = sqrt((gamma * nondim.Eu * p) / rho);
                    real H = E + p / rho;

                    return ThermodynamicState<real, ndim>{
                        .rho = rho,
                        .momentum = momentum,
                        .rhoE = rhoE,
                        .gamma = gamma,
                        .cp = cp,
                        .velocity = velocity,
                        .T = T,
                        .vv = vv,
                        .p = p,
                        .csound = csound,
                        .e = e,
                        .E = e,
                        .H = H
                    };
                } else { // variable_set == VARSET::RHO_U_P
                    static constexpr int irho = 0;
                    static constexpr int iu = 1;
                    static constexpr int ip = ndim + 1;

                    real rho = std::max(MIN_DENSITY, u[irho]);
                    real p = std::max(MIN_PRESSURE, u[ip]);
                    real cp = (gamma) / (gamma - 1) * Rgas;
                    // temperature coefficient
                    real T_coeff = cp * ref.T * ref.rho / ref.p;
                    real T = p / rho * gamma / (gamma - 1) / T_coeff;

                    Vector momentum, velocity;
                    real vv = 0;
                    for(int idim = 0; idim < ndim; ++idim) {
                        velocity[idim] = u[iu + idim];
                        momentum[idim] = u[iu + idim] * rho;
                        vv += velocity[idim] * velocity[idim];
                    }
                    real e = p * nondim.e_coeff * nondim.Eu / (gamma - 1) / rho;
                    real E = e + 0.5 * nondim.e_coeff * vv;
                    real rhoE = rho * E;
                    real csound = sqrt((gamma * nondim.Eu * p) / rho);
                    real H = E + p / rho;

                    return ThermodynamicState<real, ndim>{
                        .rho = rho,
                        .momentum = momentum,
                        .rhoE = rhoE,
                        .gamma = gamma,
                        .cp = cp,
                        .velocity = velocity,
                        .T = T,
                        .vv = vv,
                        .p = p,
                        .csound = csound,
                        .e = e,
                        .E = e,
                        .H = H
                    };

                }
            }


            /// @brief given a state vector of a given variable set 
            /// compute the dimensionless thermodynamic quantities
            /// @tparam variable_set which variable set is stored in the u array 
            /// @param state the thermodynamic state
            /// @param gradu the dimensionless state vector gradients of the given variable set 
            // NOTE: the output gradients will be with respect to the same coordinates as gradu 
            //
            /// @param ref the reference parameters
            /// @param nondim reference dimensionless parameters
            template<VARSET variable_set>
            [[nodiscard]] inline constexpr
            auto calc_state_gradients(
                const ThermodynamicState<real, ndim> state,
                const linalg::in_tensor auto gradu,
                const ReferenceParameters<real> ref,
                const Nondimensionalization<real> nondim
            ) const noexcept -> FlowStateGradients<real, ndim>
            {
                if constexpr(variable_set == VARSET::CONSERVATIVE){
                    static constexpr int irho = 0;
                    static constexpr int irhou = 1;
                    static constexpr int irhoe = ndim + 1;

                    Tensor grad_vel;

                    // calculate velocity gradients
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            grad_vel[idim][jdim] = gradu[irhou + idim, jdim] / state.rho
                                - state.velocity[idim] / state.rho * gradu[irho, jdim];
                        }
                    }

                    Vector grad_E;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        grad_E[jdim] = (
                            gradu[irhoe, jdim] - state.E * gradu[irho, jdim]
                        ) / state.rho;
                    }

                    real T_coeff = state.cp * ref.T * ref.rho / ref.p;
                    // calculate temperature gradient 
                    Vector grad_temp;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        real mult = (state.gamma / T_coeff / nondim.Eu); 
                        grad_temp[jdim] = 
                            mult *  grad_E[jdim] / (state.rho * nondim.e_coeff);
                        for(int kdim = 0; kdim < ndim; ++kdim){
                            grad_temp += mult * state.velocity[kdim] * grad_vel[kdim][jdim];
                        }
                    }
                } else if constexpr (variable_set == VARSET::RHO_U_T) {
                    static constexpr int irho = 0;
                    static constexpr int iu = 1;
                    static constexpr int iT = ndim + 1;
                    Tensor grad_vel;
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            grad_vel[idim][jdim] = gradu[iu + idim, jdim];
                        }
                    }

                    // calculate temperature gradient 
                    Vector grad_temp;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        grad_temp[jdim] = gradu[iT, jdim];
                    }
                } else { // variable_set = VARSET::RHO_U_P
                    static constexpr int irho = 0;
                    static constexpr int iu = 1;
                    static constexpr int ip = ndim + 1;
                    Tensor grad_vel;
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            grad_vel[idim][jdim] = gradu[iu + idim, jdim];
                        }
                    }

                    real T_coeff = state.cp * ref.T * ref.rho / ref.p;
                    Vector grad_temp;
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        // quotient rule
                        grad_temp[jdim] = (gradu[ip, jdim] * state.rho - gradu[irho] * state.p)
                            / SQUARED(state.rho) * gamma / (gamma - 1) / T_coeff;
                    }
                }
            }
        };

        static_assert(is_eos<CaloricallyPerfectEoS<double, 3>>, "Must satisfy EoS");
        /// @brief Handles the physics relations and quantities for thermodynamic state 
        ///
        /// @tparam real the real number type 
        /// @tparam ndim the number of dimensions 
        /// @tparam EoS the equation of state
        /// @tparam varset the variable set for the state vector
        template<class real, int ndim, is_eos EoS, VARSET varset = VARSET::CONSERVATIVE>
        struct Physics {
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, ndim>;
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, ndim, ndim>;
            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;

            /// @brief Prandtl number 
            const real Pr;

            /// @brief reference parameters
            const ReferenceParameters<real> ref;

            /// @brief Sutherlands law for viscosity temperature closure in dimensionless form
            const DimensionlessSutherlands<real> viscosity;

            /// @brief the nondimensionalization
            const Nondimensionalization<real> nondim;

            const EoS eos;

            /// @brief the temperatures for isothermal boundary conditions
            std::vector<real> isothermal_temperatures{};

            /// @brief the free stream state
            FreeStream<real, ndim> free_stream{};

            /// @brief Constructor
            /// Set up all 
            Physics(
                ReferenceParameters<real> ref, /// @param reference parameters for nondimensionalization
                EoS eos,          /// @param the equation of state
                real Pr = 0.72       /// @param Prandtl number
            ) : Pr{Pr}, ref{ref}, viscosity{ref},
                nondim{create_nondim(ref)}, eos{eos}
            {}

            /// @brief calculate the shear stress given the thermodynamic state and 
            /// flow gradiens 
            /// @param state the thermodynamic stae 
            /// @param grads the flow state gradients
            [[nodiscard]] inline constexpr 
            auto calc_shear_stress(
                    ThermodynamicState<real, ndim>& state, FlowStateGradients<real, ndim>& grads)
            const noexcept -> Tensor {
                // get the viscosity
                real mu = viscosity(state.T);

                const auto& dudx = grads.velocity_gradient;

                Tensor tau;
                real dukduk = 0;
                for(int k = 0; k < ndim; ++k){
                    dukduk += dudx[k, k];
                }
                for(int idim = 0; idim < ndim; ++idim){
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        tau[idim, jdim] = mu * (dudx[idim, jdim] + dudx[jdim, idim]);
                    }
                }
                for(int i = 0; i < ndim; ++i){
                    tau[i, i] -= 2.0 / 3.0 * mu * dukduk;
                }

                return tau;
            }

            // @brief given a state vector in the native variable set (varset)
            // get the thermodynamic state from the EoS
            // @param u the state vector 
            // @return the thermodynamic state
            [[nodiscard]] inline constexpr 
            auto calc_thermo_state(std::array<real, nv_comp> u)
            const noexcept -> ThermodynamicState<real, ndim>
            { return eos.template calc_thermo_state<varset>(u, ref, nondim); }
        };

        template<class real, class EoS>
        Physics(ReferenceParameters<real>, EoS) -> Physics<real, EoS::ndim, EoS>;

        /// @brief Van Leer flux
        /// implementation reference: http://www.chimeracfd.com/programming/gryphon/fluxvanleer.html
        template< class T, int _ndim, is_eos EoS, VARSET varset>
        struct VanLeer {

            static constexpr int ndim = _ndim;
            using value_type = T;

            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim>;

            /// @brief number of variables
            static constexpr int nv_comp = ndim + 2;
            static constexpr int neq = nv_comp;

            Physics<T, ndim, EoS, varset> physics;

            inline constexpr
            auto operator()(
                std::array<T, nv_comp> uL,
                std::array<T, nv_comp> uR,
                Vector unit_normal
            ) const noexcept -> std::array<T, neq> {
                using namespace NUMTOOL::TENSOR::FIXED_SIZE;
                ThermodynamicState<T, ndim> stateL = physics.calc_thermo_state(uL);
                ThermodynamicState<T, ndim> stateR = physics.calc_thermo_state(uR);

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
                        flux[1 + idim] = stateL.momentum[idim] * vnormalL 
                            + physics.nondim.Eu * stateL.pressure * unit_normal[idim];
                    flux[ndim + 1] = vnormalL * (stateL.rhoe + physics.nondim.Eu * physics.nondim.e_coeff * stateL.pressure);
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
                        flux[1 + idim] += stateR.momentum[idim] * vnormalR 
                            + physics.nondim.Eu * stateR.pressure * unit_normal[idim];
                    flux[ndim + 1] += vnormalR * (stateR.rhoe + physics.nondim.Eu * physics.nondim.e_coeff * stateR.pressure);
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
        template< class T, int _ndim, is_eos EoS, VARSET varset>
        VanLeer(Physics<T, _ndim, EoS, varset>) -> VanLeer<T, _ndim, EoS, varset>;

        // @brief the physical flux for navier stokes equations
        // @tparam T the real number type
        // @tparam _ndim the number of dimensions 
        // @tparam euler if true, limits computations for euler equations (don't compute viscous terms)
        template< class real, int _ndim, is_eos EoS, VARSET varset, bool euler = true >
        struct Flux {

            template<class T2, std::size_t... sizes>
            using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;
            using Vector = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<real, _ndim>;

            /// @brief the real number type
            using value_type = real;

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

            Physics<real, ndim, EoS, varset> physics;

            mutable real lambda_max = 0.0;

            mutable real visc_max = 0.0;

            inline constexpr 
            auto operator()(
                std::array<real, nv_comp> u,
                linalg::in_tensor auto gradu
            ) const noexcept -> Tensor<real, neq, ndim> {

                ThermodynamicState<real, ndim> state = physics.calc_thermo_state(u);

                // nondimensional quantities
                real Re = physics.nondim.Re;
                real e_coeff = physics.nondim.e_coeff;
                real gamma = state.gamma;
                real Eu = physics.nondim.Eu;
                real Pr = physics.Pr;

                lambda_max = state.csound + std::sqrt(state.vv);

                Tensor<real, neq, ndim> flux;
                // loop over the flux direction j
                for(int jdim = 0; jdim < ndim; ++jdim) {
                    flux[0][jdim] = u[1 + jdim];
                    for(int idim = 0; idim < ndim; ++idim)
                        flux[1 + idim][jdim] = state.momentum[idim] * state.velocity[jdim];
                    flux[1 + jdim][jdim] += Eu * state.p;
                    flux[1 + ndim][jdim] = state.velocity[jdim] * (state.rhoe + Eu * e_coeff * state.pressure);
                }

                if(!euler) {
                    // get gradients of state
                    FlowStateGradients<real, ndim> state_grads{physics.calc_thermo_state_gradients(state, gradu)};

                    // get the temperature coefficient
                    real T_coeff = state.cp * physics.ref.T * physics.ref.rho / physics.ref.p;

                    // get the shear stress
                    Tensor tau = physics.calc_shear_stress(state, state_grads);

                    real mu = physics.viscosity(state.T);

                    // subtract viscous fluxes
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = 0; jdim < ndim; ++jdim){
                            flux[irhou + idim][jdim] -= tau[idim][jdim] / physics.nondim.Re;
                            flux[irhoe][jdim] -= state.velocity[idim] * state_grads.tau[idim][jdim] 
                                * e_coeff / Re;
                        }
                        flux[irhoe][idim] -= state_grads.temp_gradient[idim] 
                            * state.mu * e_coeff / Re * gamma * R_ref * Eu / (gamma - 1) / Pr;
                    }

                    visc_max = std::max(visc_max, mu);
                }
                
                return flux;
            }

            inline constexpr 
            auto apply_bc(
                std::array<real, neq> &uL,
                linalg::in_tensor auto graduL,
                Vector unit_normal,
                BOUNDARY_CONDITIONS bctype,
                int bcflag
            ) const noexcept {
                using namespace NUMTOOL::TENSOR::FIXED_SIZE;

                // outputs
                std::array<real, neq> uR;
                Tensor<real, neq, ndim> graduR;

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
                        ThermodynamicState<real, ndim> stateL = physics.calc_thermo_state(uL);

                        // density and energy are the same
                        uR[0] = uL[0];
                        uR[1 + ndim] = uL[1 + ndim];

                        // flip velocity over the normal
                        real mom_n = dot(stateL.momentum, unit_normal);
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
                        ThermodynamicState<real, ndim> stateL = physics.calc_thermo_state(uL);

                        constexpr VARSET prim_varset = VARSET::RHO_U_T;
                        std::array<real, neq> prim;
                        // density is the same
                        prim[0] = stateL.density;

                        // velocity, and therefore momentum, is zero
                        for(int idim = 0; idim < ndim; ++idim){
                            prim[1 + idim] = 0;
                        }

                        // compute energy from Twall 
                        prim[1 + ndim] = physics.isothermal_temperatures[bcflag];

                        // convert
                        uR = physics.eos.template convert_varset<prim_varset, varset>(prim);

                        // exterior state gradients equal interor state gradients 
                        linalg::copy(graduL, linalg::as_mdspan(graduR));
                    } break;
                    case BOUNDARY_CONDITIONS::RIEMANN:
                    {
                        // Carlson "Inflow/Outflow Boundary Conditions with Application to FUN3D"
                        // NASA/TM-2011-217181

                        // TODO: optimize over branches to remove unused calculations
                        // in supersonic cases


                        ThermodynamicState<real, ndim> stateL = physics.calc_thermo_state(uL);
                        real gamma = stateL.gamma; // used a ton, make more concise
                        FreeStream<real, ndim> free_stream = physics.free_stream;
                        std::array<real, neq> fs_vec;
                        fs_vec[0] = free_stream.rho_inf;
                        for(int idim = 0; idim < ndim; ++idim)
                            fs_vec[1 + idim] = free_stream.u_direction[idim] * free_stream.u_inf;
                        fs_vec[1 + ndim] = free_stream.temp_inf;
                        ThermodynamicState<real, ndim> state_freestream 
                            = physics.eos.template calc_thermo_state<VARSET::RHO_U_T>(fs_vec);
                        real normal_uadv_i = dot(stateL.velocity, unit_normal);
                        real normal_uadv_o = dot(state_freestream.velocity, unit_normal);
                        real normal_mach = normal_uadv_i / stateL.csound;

                        // eq 8
                        real Rplus = normal_uadv_i + 2 * stateL.csound / (gamma - 1);
                        real Rminus = normal_uadv_o - 2 * state_freestream.csound 
                            / (gamma - 1);

                        if(normal_mach > 0) {
                            // supersonic outflow 
                            real Rminus = normal_uadv_i - 2 * stateL.csound 
                                / (gamma - 1);
                        } else if (normal_mach < 0){
                            // supersonic inflow
                            Rplus = normal_uadv_o + 2 * state_freestream.csound 
                                / (gamma - 1);
                        }

                        real Ub = 0.5 * (Rplus + Rminus);
                        // NOTE: paper says 4* but Rodriguez et al
                        //"Formulation and Implementation of 
                        // Inflow/Outflow Boundary Conditions to Simulate 
                        // Propulsive Effects"
                        // says (gamma - 1) / 4 
                        // which would correctly recover the supersonic sound speeds
                        // so this is most likely a typo
                        real cb = 0.25 * (gamma - 1) * (Rplus - Rminus);

                        // eq 15 and 16
                        Vector uadvB;
                        real sb;
                        if(Ub > 0) {
                            // outflow
                            uadvB = stateL.velocity;
                            axpy(Ub - normal_uadv_i, unit_normal, uadvB);
                            sb = SQUARED(stateL.csound) / 
                                (gamma * std::pow(stateL.density, gamma - 1));
                        } else {
                            // inflow
                            uadvB = stateL.velocity;
                            axpy(Ub - normal_uadv_i, unit_normal, uadvB);
                            sb = SQUARED(stateL.csound) / 
                                (gamma * std::pow(stateL.density, gamma - 1));
                        }

                        real rhoR = std::pow(SQUARED(cb) / (gamma * sb), 1.0 / (gamma - 1));
                        real pR = rhoR * SQUARED(cb) / gamma;

                        // convert to conservative variables
                        uR = physics.prim_to_cons(rhoR, uadvB, pR);
                        
                        // exterior state gradients equal interor state gradients 
                        linalg::copy(graduL, linalg::as_mdspan(graduR));

//                            // subsonic outflow alternate
//                            // Rodgriguez et al. "Formulation and Implementation of 
//                            // Inflow/Outflow Boundary Conditions to Simulate 
//                            // Propulsive Effects"
//                            
//                            // TODO: prestore?
//                            FlowState<T, ndim> state_freestream 
//                                = physics.calc_thermo_state(physics.free_stream);
//
//                            // Entropy (eq. 3)
//                            T SR = stateL.pressure / stateL.density;
//
//                            // Riemann invariant (eq. 4)
//                            T JL = -normal_mach * stateL.csound
//                                + 2 * stateL.csound / (physics.gamma - 1);
//
//                            // speed of sound on the right (eq. 5)
//                            T csoundR = std::sqrt(
//                                physics.gamma * std::pow(state_freestream.pressure,
//                                    (physics.gamma - 1) / physics.gamma)
//                                * std::pow(SR, 1.0 / physics.gamma)
//                            );
//
//                            // Riemann invariant on the right (eq. 7)
//                            T JR = JL - 4.0 / (physics.gamma - 1) * csoundR;
//
//                            // density on the right 
//                            T rhoR = std::pow( 
//                                csoundR * csoundR / (physics.gamma * SR),
//                                1.0 / (physics.gamma - 1)
//                            );
//
//                            // Tangential and normal internal velocities
//                            // (vector quantitites)
//                            Vector VtL, VnL;
//                            VnL = unit_normal;
//                            for(int i = 0; i < ndim; ++i) 
//                                VnL[i] *= normal_mach * stateL.csound;
//                            VtL = stateL.velocity;
//                            axpy(-1.0, VnL, VtL);
//
//                            Vector uadvR = VtL;
//                            // add in normal component (eq. 9);
//                            axpy(-0.5 * (JL + JR), unit_normal, uadvR);
//
//                            // build the right state
//                            uR[irho] = rhoR;
//                            for(int idim = 0; idim < ndim; ++idim){
//                                uR[irhou + idim] = rhoR * uadvR[idim];
//                            }
//                            uR[irhoe] = state_freestream.pressure / (physics.gamma - 1)
//                                + 0.5 * rhoR * dot(uadvR, uadvR);
//                            // exterior state gradients equal interor state gradients 
//                            linalg::copy(graduL, linalg::as_mdspan(graduR));
                    } break;
                    default:
                    util::AnomalyLog::log_anomaly(util::Anomaly{"Unsupported BC",
                            util::general_anomaly_tag{}});


                }

                return std::pair{uR, graduR};
            }

            inline constexpr 
            auto dt_from_cfl(real cfl, real reference_length) const noexcept -> real {
                real dt = (reference_length * cfl) / lambda_max;
                if(!euler){
                    dt = std::min(dt, SQUARED(reference_length) * cfl / visc_max);
                }
                // reset maximums
                lambda_max = 0;
                visc_max = 0;
                return dt;
            }
        };
        template< class T, int _ndim, is_eos EoS, VARSET varset>
        Flux(Physics<T, _ndim, EoS, varset>) -> Flux<T, _ndim, EoS, varset>;

        template< class T, int _ndim, is_eos EoS, VARSET varset, bool euler = true >
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

            Physics<T, ndim, EoS, varset> physics;

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
                    ThermodynamicState<T, ndim> state = physics.calc_thermo_state(u);

                    // get gradients of state
                    FlowStateGradients<T, ndim> state_grads{physics.calc_thermo_state_gradients(state, gradu)};

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
