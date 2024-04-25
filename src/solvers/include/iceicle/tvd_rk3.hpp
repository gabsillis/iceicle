/**
 * @brief strong stability preserving Runge-Kutta scheme
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "Numtool/matrix/decomposition/decomp_lu.hpp"
#include "Numtool/matrix/dense_matrix.hpp"
#include "Numtool/matrix/permutation_matrix.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/form_residual.hpp"
#include "iceicle/explicit_utils.hpp"

#include <iostream>
#include <iomanip>
namespace iceicle::solvers {

/**
 * @brief Explicit 3-stage Total Variation Diminishing Runge-Kutta
 *
 * TODO:
 * Butcher tableau
 *
 * 0   | 0    0    0
 * 1   | 1    0    0
 * 1/2 | 1/4  1/4  0
 * ----+---------------
 *     | 1/6  1/6  2/3
 */
template< class T, class IDX, class TimestepClass, class StopCondition>
class RK3TVD {

public: 

    /** The timestep determination */
    TimestepClass timestep;

    /** The termination criteria */
    StopCondition stop_condition;

    /// @brief the residual data array
    std::vector<T> res_data;

    /// @brief intermediate step data arrays
    std::vector<T> res1_data, res2_data, res3_data, u_stage_data;

    /// @brief the current timestep 
    IDX itime = 0;

    /// @brief the current time 
    T time = 0.0;

    /// @brief the callback function for visualization during solve()
    /// is given a reference to this when called 
    /// default is to print out a l2 norm of the residual data array
    std::function<void(RK3TVD &)> vis_callback = [](RK3TVD &disc){
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
     * @brief create a RK3TVD solver 
     * initializes the residual data vector
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param timestep the class that determines the 
     */
    template<int ndim, class disc_class>
    RK3TVD(
        FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        const TimestepClass &timestep,
        const StopCondition &stop_condition
    )
    requires specifies_ncomp<disc_class> && TerminationCondition<StopCondition>
    : res_data(fespace.dg_map.calculate_size_requirement(disc_class::nv_comp)),
      res1_data(fespace.dg_map.calculate_size_requirement(disc_class::nv_comp)),
      res2_data(fespace.dg_map.calculate_size_requirement(disc_class::nv_comp)),
      res3_data(fespace.dg_map.calculate_size_requirement(disc_class::nv_comp)),
      u_stage_data(fespace.dg_map.calculate_size_requirement(disc_class::nv_comp)),
      timestep{timestep}, stop_condition{stop_condition}
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
    void step(FESpace<T, IDX, ndim> &fespace, disc_class &disc, fespan<T, LayoutPolicy, uAccessorPolicy> u)
    requires TimestepT<TimestepClass, T, IDX, ndim, disc_class, LayoutPolicy, uAccessorPolicy>
    {
       using namespace NUMTOOL::TENSOR::FIXED_SIZE;

       // RK coefficients by number of stages 
       // [nstage-1, istage]
       
       // alfa multiplies uold
       static Tensor<T, 5, 5> valfa{{
           {0.0,  0.0,       0.0,       0.0, 0.0},
           {0.0,  0.5,       0.0,       0.0, 0.0},
           {0.0,  0.75,      1.0 / 3.0, 0.0, 0.0},
           {0.25, 1.0 / 3.0, 0.5,       1.0, 0.0},
           {0.25, 1.0 / 6.0, 3.0 / 8.0, 0.5, 1.0}
       }};

        // beta multiplies current stage u
        static Tensor<T, 5, 5> vbeta{{
            {1.0,  0.0,       0.0,       0.0, 0.0},
            {1.0,  0.5,       0.0,       0.0, 0.0},
            {1.0,  0.25,      2.0 / 3.0, 0.0, 0.0},
            {0.25, 1.0 / 3.0, 0.5,       1.0, 0.0},
            {0.25, 1.0 / 6.0, 0.375,     0.5, 1.0}
        }};

        // number of TVD Stages
        static constexpr int nstag = 3;

        // calculate the timestep 
        T dt = timestep(fespace, disc, u);

        // Termination Condiditon restrictions on dt 
        dt = stop_condition.limit_dt(dt, time);

        // create view of the residual using the same Layout as u 
        fespan res{res_data.data(), u.get_layout()};

        // storage for rhs of mass matrix equation
        int max_ndof = fespace.dg_map.max_el_size_reqirement(1);
        std::vector<T> b(max_ndof);
        std::vector<T> du(max_ndof);

        // function to get the residual for a single stage
        auto stage_residual = [&](fespan<T, LayoutPolicy> u_stage, fespan<T, LayoutPolicy> res_stage){
            // zero out
            res_stage = 0;

            // get the rhs
            form_residual(fespace, disc, u, res);

            // invert mass matrices
            // TODO: prestore mass matrix with the reference element 
            // TODO: need to build a global mass matrix if doing CG (but not for DG)
            for(const FiniteElement<T, IDX, ndim> &el : fespace.elements){
                using namespace MATH::MATRIX;
                using namespace MATH::MATRIX::SOLVERS;
                DenseMatrix<T> mass = calculate_mass_matrix(el, fespace.meshptr->nodes);
                PermutationMatrix<unsigned int> pi = decompose_lu(mass);

                const std::size_t ndof = el.nbasis();
                for(IDX ieqn = 0; ieqn < disc_class::nv_comp; ++ieqn){

                    // copy the residual for each degree of freedom to the rhs 
                    for(IDX idof = 0; idof < ndof; ++idof){
                        b[idof] = res[el.elidx, idof, ieqn];
                    }

                    // solve the matrix equation 
                    sub_lu(mass, pi, b.data(), du.data());

                    for(IDX idof = 0; idof < ndof; ++idof){
                        res_stage[el.elidx, idof, ieqn] += du[idof];
                    }
                }
            }
        };

        // describe fespans for intermediate states 
        fespan res1{res1_data.data(), u.get_layout()};
        fespan res2{res2_data.data(), u.get_layout()};
        fespan res3{res3_data.data(), u.get_layout()};
        fespan u_old{u_stage_data.data(), u.get_layout()};


        copy_fespan(u, u_old);
       
        for(int istage = 0; istage < nstag; ++istage){
            // get Minv * r for the stage u
            stage_residual(u, res1);

            // update from coefficients
            axpby(valfa[nstag-1][istage], u_old, vbeta[nstag-1][istage], u); 

            // add residual dt contribution
            axpy(vbeta[nstag-1][istage] * dt, res1, u);
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
    void solve(FESpace<T, IDX, ndim> &fespace, disc_class &disc, fespan<T, LayoutPolicy, uAccessorPolicy> u) {

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
RK3TVD(FESpace<T, IDX, ndim> &, disc_class &,
    const TimestepClass &, const StopCondition &) -> RK3TVD<T, IDX, TimestepClass, StopCondition>;

}
