/**
 * @brief Explicit Euler (1-stage Runge Kutta) time integration
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "Numtool/matrix/decomposition/decomp_lu.hpp"
#include "Numtool/matrix/dense_matrix.hpp"
#include "Numtool/matrix/permutation_matrix.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/form_residual.hpp"
#include "iceicle/explicit_utils.hpp"

#include <iostream>
#include <iomanip>
namespace ICEICLE::SOLVERS {

template< class T, class IDX, class TimestepClass, class StopCondition>
class ExplicitEuler {

public: 

    /** The timestep determination */
    TimestepClass timestep;

    /** The termination criteria */
    StopCondition stop_condition;

    /// @brief the residual data array
    std::vector<T> res_data;

    /// @brief the current timestep 
    IDX itime = 0;

    /// @brief the current time 
    T time = 0.0;

    /// @brief the callback function for visualization during solve()
    /// is given a reference to this when called 
    /// default is to print out a l2 norm of the residual data array
    std::function<void(ExplicitEuler &)> vis_callback = [](ExplicitEuler &disc){
        T sum = 0.0;
        for(int i = 0; i < disc.res_data.size(); ++i){
            sum += SQUARED(disc.res_data[i]);
        }
        std::cout << std::setprecision(8);
        std::cout << "itime: " << std::setw(6) << disc.itime 
            << " | t: " << std::setw(14) << disc.time
            << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
            << std::endl;
    };

    /// @brief if this is a positive integer 
    /// then the vis_callback will be called every ivis timesteps 
    /// (itime % ivis == 0)
    IDX ivis = -1;

    /**
     * @brief create a ExplicitEuler solver 
     * initializes the residual data vector
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param timestep the class that determines the 
     */
    template<int ndim, class disc_class>
    ExplicitEuler(
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        const TimestepClass &timestep,
        const StopCondition &stop_condition
    )
    requires specifies_ncomp<disc_class> && TerminationCondition<StopCondition>
    : res_data(fespace.dg_offsets.calculate_size_requirement(disc_class::dnv_comp))
    {}

    /**
     * @brief perform a single timestep 
     * NOTE: this chooses the same layout for res based on u
     *
     * @param [in] fespace the finite element space 
     * @param [in] disc the discretization 
     * @param [in/out] u the solution as an fespan view
     */
    template<int ndim, class disc_class, class LayoutPolicy, class uAccessorPolicy>
    void step(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc, FE::fespan<T, LayoutPolicy, uAccessorPolicy> u)
    requires TimestepT<TimestepClass, T, IDX, ndim, disc_class, LayoutPolicy, uAccessorPolicy>
    {
       
        // calculate the timestep 
        T dt = timestep(fespace, disc, u);

        // Termination Condiditon restrictions on dt 
        dt = stop_condition.limit_dt(dt, time);

        // create view of the residual using the same Layout as u 
        FE::fespan res{res_data.data(), u.get_layout()};

        // get the rhs
        form_residual(fespace, disc, u, res);

        // storage for rhs of mass matrix equation
        int max_ndof = fespace.dg_offsets.max_el_size_reqirement(1);
        std::vector<T> b(max_ndof);
        std::vector<T> du(max_ndof);

        // TODO: prestore mass matrix with the reference element 
        // TODO: need to build a global mass matrix if doing CG (but not for DG)
        for(ELEMENT::FiniteElement<T, IDX, ndim> &el : fespace.elements){
            using namespace MATH::MATRIX;
            using namespace MATH::MATRIX::SOLVERS;
            DenseMatrix<T> mass = ELEMENT::calculate_mass_matrix(el, fespace.meshptr->nodes);
            PermutationMatrix<unsigned int> pi = decompose_lu(mass);

            const std::size_t ndof = el.nbasis();
            for(std::size_t ieqn = 0; ieqn < disc_class::dnv_comp; ++ieqn){

                // copy the residual for each degree of freedom to the rhs 
                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                    b[idof] = res[FE::fe_index{(std::size_t) el.elidx, idof, ieqn}];
                }

                // solve the matrix equation 
                sub_lu(mass, pi, b.data(), du.data());

                // TODO: add to u 
                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                    u[FE::fe_index{(std::size_t) el.elidx, idof, ieqn}] += dt * du[idof];
                }
            }
        }

        // update the timestep and time
        itime++;
        time += dt;
    }

    /**
     * @brief perform timesteps until the stop condition is reached 
     * either itime reaches ntime (ntime >= 0)
     * or the time value reaches tfinal
     *
     * NOTE: this chooses the same layout for res based on u
     *
     * @param [in] fespace the finite element space 
     * @param [in] disc the discretization 
     * @param [in/out] u the solution as an fespan view
     */
    template<int ndim, class disc_class, class LayoutPolicy, class uAccessorPolicy>
    void solve(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc, FE::fespan<T, LayoutPolicy, uAccessorPolicy> u) {

        // visualization callback on initial state (0 % anything == 0) 
        vis_callback(*this);

        // timestep loop
        while(!stop_condition(itime, time)){
            step(fespace, disc, u);
            if(itime % ivis == 0){
                vis_callback(*this);
            }
        }
    }
};

// template argument deduction
template<class T, class IDX, int ndim, class disc_class, class TimestepClass, class StopCondition>
ExplicitEuler(FE::FESpace<T, IDX, ndim> &, disc_class &,
    const TimestepClass &, const StopCondition &) -> ExplicitEuler<T, IDX, TimestepClass, StopCondition>;

}
