/**
 * @brief Explicit Euler (1-stage Runge Kutta) time integration
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/fespace/fespace.hpp"
#include "iceicle/form_residual.hpp"
namespace ICEICLE::SOLVERS {

template< class T, class IDX >
class ExplicitEuler {

public: 

    /// @brief the cfl condition number to determine the timestep
    T cfl;

    /// @brief the number of timesteps to run if calling solve()
    /// if this is negative, then tfinal is used
    IDX ntime;

    /// @brief the final time to terminate solve()
    T tfinal;

    /// @brief the residual data array
    std::vector<T> res_data;

    /**
     * @brief create a ExplicitEuler solver 
     * initializes the residual data vector
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param cfl the cfl condition 
     * @param ntime the number of timesteps or -1 to use tfinal instead
     * @tparam tfinal the final time to reach
     */
    template<int ndim, class disc_class>
    ExplicitEuler(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc,
        T cfl = 0.3, IDX ntime = -1, T tfinal = 1.0)
    requires specifies_ncomp<disc_class>
    : cfl(cfl), ntime(ntime), tfinal(tfinal), 
      res_data(fespace.dg_offsets.calculate_size_requirement(disc_class::dnv_comp))
    {}

    /**
     * @brief perform a single timestep 
     *
     * @param [in] fespace the finite element space 
     * @param [in] disc the discretization 
     * @param [in/out] udata the solution data, gets updated
     */
    template<int ndim, class disc_class>
    T step(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc, std::span<T> u_data){
        
        // get the rhs
        form_residual(fespace, disc, u_data, std::span{res_data});

        // TODO: solve the M du/dt = rhs
    }

};


}
