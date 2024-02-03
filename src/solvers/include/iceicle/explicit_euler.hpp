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

    /// @brief a constant timestep value set to a non-negative number to use 
    /// a set timestep instead of computing it
    T dt_fixed;

    /// @brief the residual data array
    std::vector<T> res_data;

    /// @brief the current timestep 
    IDX itime = 0;

    /// @brief the current time 
    T time = 0.0;

    /**
     * @brief create a ExplicitEuler solver 
     * initializes the residual data vector
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param cfl the cfl condition 
     * @param ntime the number of timesteps or -1 to use tfinal instead
     * @tparam tfinal the final time to reach
     * @param dt if non-negative, this is the fixed timestep to use
     */
    template<int ndim, class disc_class>
    ExplicitEuler(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc,
        T cfl = 0.3, IDX ntime = -1, T tfinal = 1.0, T dt = -1.0)
    requires specifies_ncomp<disc_class>
    : cfl(cfl), ntime(ntime), tfinal(tfinal), dt_fixed(dt),
      res_data(fespace.dg_offsets.calculate_size_requirement(disc_class::dnv_comp))
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
    T step(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc, FE::fespan<T, LayoutPolicy, uAccessorPolicy> u){
       
        // calculate the timestep 
        T dt;
        if(dt_fixed > 0){
            dt = dt_fixed;
        } else {
            // get timestep from cfl condition 

            // first: reference length is the minimum diagonal entry of the jacobian at the cell center
            T reflen = 1e8;
            for(const ELEMENT::FiniteElement<T, IDX, ndim> &el : fespace.elements){
                MATH::GEOMETRY::Point<T, ndim> center_xi = el.geo_el.centroid_ref();
                auto J = el.geo_el->Jacobian(fespace.meshptr->nodes, center_xi);
                for(int idim = 0; idim < ndim; ++idim){
                    reflen = std::min(reflen, J[idim][idim]);
                }
            }

            // calculate the timestep from the CFL condition 
            disc.dt_from_cfl(cfl, reflen);
        }

        // if final time is non-negative make sure we don't go over the final time 
        if(tfinal > 0){
            dt = std::min(dt, tfinal - time);
        }

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

            const int ndof = el.nbasis();
            for(int ieqn = 0; ieqn < disc_class::dnv_comp; ++ieqn){

                // copy the residual for each degree of freedom to the rhs 
                for(int idof = 0; idof < el.nbasis(); ++idof){
                    b[idof] = res[FE::fe_index{el.elidx, idof, ieqn}];
                }

                // solve the matrix equation 
                sub_lu(mass, pi, b.data(), du.data());

                // TODO: add to u 
                for(int idof = 0; idof < el.nbasis(); ++idof){
                    u[FE::fe_index{el.elidx, idof, ieqn}] += dt * du[idof];
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
    T solve(FE::FESpace<T, IDX, ndim> &fespace, disc_class &disc, FE::fespan<T, LayoutPolicy, uAccessorPolicy> u) {

        // stop conditions
        auto done = [&]() -> bool {
            if(ntime > 0){
                return itime >= ntime;
            } else {
                return time >= tfinal;
            }
        };

        while(!done()){
            step(fespace, disc, u);
        }
    }
};


}
