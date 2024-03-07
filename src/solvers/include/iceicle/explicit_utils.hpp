/**
 * @brief Common utilities for explicit timestepping 
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include <variant>
namespace ICEICLE::SOLVERS{

    // ==========================
    // = Timestep Determination =
    // ==========================

    /**
     * @brief The timestep determination concept 
     * This describes a class that can be used to determine 
     * what timestep to use given the finte element space,
     * discretization, and current solution state
     */
    template<
        class TimestepClass,
        class T,
        class IDX,
        int ndim,
        class disc_T,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    concept TimestepT = requires(
        const TimestepClass& timestep,
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_T disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u
        ){
        {timestep(fespace, disc, u)} -> std::same_as<T>;
    };

    /**
     * @brief a Fixed value timestep 
     * uses the provided value 
     */
    template<class T, class IDX>
    struct FixedTimestep {
        T dt;

        template< int ndim, class disc_T, class LayoutPolicy, class AccessorPolicy >
        inline T operator()(
            FE::FESpace<T, IDX, ndim> &fespace,
            disc_T &disc,
            FE::fespan<T, LayoutPolicy, AccessorPolicy> u
        ) const noexcept { return dt; }
    };

    /**
     * @brief determine the timestep based on the CFL condition 
     * of the discretization
     */
    template<class T, class IDX>
    struct CFLTimestep {
        T cfl = 0.3;

        template< int ndim, class disc_T, class LayoutPolicy, class AccessorPolicy >
        inline T operator()(
            FE::FESpace<T, IDX, ndim> &fespace,
            disc_T &disc,
            FE::fespan<T, LayoutPolicy, AccessorPolicy> u
        ) const noexcept {

            // first: reference length is the minimum diagonal entry of the jacobian at the cell center
            T reflen = 1e8;
            int Pn_max = 1;
            for(const ELEMENT::FiniteElement<T, IDX, ndim> &el : fespace.elements){
                MATH::GEOMETRY::Point<T, ndim> center_xi = el.geo_el->centroid_ref();
                auto J = el.geo_el->Jacobian(fespace.meshptr->nodes, center_xi);
                for(int idim = 0; idim < ndim; ++idim){
                    reflen = std::min(reflen, J[idim][idim]);
                }

                Pn_max = std::max(Pn_max, el.basis->getPolynomialOrder());
            }

            // calculate the timestep from the CFL condition 
            return disc.dt_from_cfl(cfl, reflen) / (2 * (Pn_max + 1));
        }
    };

    /** @brief a variant of all the timestep classes */
    template<class T, class IDX>
    using TimestepVariant = std::variant<FixedTimestep<T, IDX>, CFLTimestep<T, IDX>>;

    // ==========================
    // = Termination Conditions =
    // ==========================

    /**
     * @brief The condition on which to terminate iterations
     * specifies termination based on the timestep and time 
     * also has the ability to limit the timestep to 
     *  do things like exactly matching a final time
     */
    template<typename TConditionClass>
    concept TerminationCondition = requires (
        const TConditionClass &tcondition,
        TConditionClass::index_type itime,
        TConditionClass::value_type time,
        TConditionClass::value_type dt
    ){
        {tcondition(itime, time)} -> std::same_as<bool>;
        {tcondition.limit_dt(dt, time)} -> std::same_as<typename TConditionClass::value_type>;
    };

    /**
     * @brief A termination condition based on the number of timesteps
     * Once the set number of timesteps is reached, this terminates the solve
     */
    template<class T, class IDX>
    struct TimestepTermination {
        using value_type = T;
        using index_type = IDX;

        /// @brief the number of timesteps to terminate at
        IDX ntime;
       
        /** @brief terminate if the timesteps have reached ntime */
        inline bool operator()(int itime, T time) const noexcept{
            return itime >= ntime;
        }

        /** @brief this termination condition imposes no timestep limit */
        inline T limit_dt(T dt, T time) const noexcept { return dt; }
    };

    /**
     * @brief A termination condition based on the final time 
     * terminates when the final time is met
     */
    template<class T, class IDX>
    struct TfinalTermination {
        using value_type = T;
        using index_type = IDX;

        /// @brief the final time 
        T tfinal = 1.0;

        /** @brief terminate if the time has reached tfinal */
        inline bool operator()(int itime, T time) const noexcept {
            return time >= tfinal;
        }

        /** 
         * @brief this termination condition limits timestep 
         * to not go over tfinal 
         */
        inline T limit_dt(T dt, T time) const noexcept {
            return std::min(dt, tfinal - time);
        }
    };

    /** @brief a variant of all the termination classes */
    template<class T, class IDX>
    using TerminationVariant = std::variant<TimestepTermination<T, IDX>, TfinalTermination<T, IDX> >;
}
