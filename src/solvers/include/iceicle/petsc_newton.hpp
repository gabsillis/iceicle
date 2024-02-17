/**
 * @brief newton's method solvers that use petsc as a backend
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/nonlinear_solver_utils.hpp"
#include <functional>
#include <iomanip>
#include <iostream>
#include <petscmat.h>
#include <petscsys.h>
#include <petscvec.h>

namespace ICEICLE::SOLVERS {

    /**
     * @brief Newton solver that uses Petsc for linear solvers 
     * @tparam T the floating point type 
     * @tparam IDX the index type
     * @tparam ndim the number of dimensions
     * @tparam disc_class the discretization
     */
    template<class T, class IDX, int ndim, class disc_class>
    class PetscNewton {

        // ================
        // = Data Members =
        // ================
        private:
        /// @brief the Jacobian Matrix
        Mat jac;

        /// @brief the storage for the residual vector
        Vec res_data;

        /// @brief storage for the solution update vector
        Vec du_data;

        /// @brief store a reference to the fespace being used 
        FE::FESpace<T, IDX, ndim> &fespace;

        /// @brief store a reference to the discretization being solved
        disc_class &disc;

        public:
        /// @brief the convergence Criteria
        /// determines whether the solver should terminate
        ConvergenceCriteria<T, IDX> conv_criteria;

        /// @brief diagnostics function 
        /// very minimal by default other options are defined in this header
        /// or a custom function can be made 
        std::function<void(PetscNewton &)> diag_callback = [](PetscNewton &solver) {
        
        };

        /// @brief the callback function for visualization during solve()
        /// is given a reference to this when called 
        /// default is to print out a l2 norm of the residual data array
        std::function<void(PetscNewton &)> vis_callback = [](PetscNewton &disc){
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

        // ================
        // = Constructors =
        // ================

        /**
         * @brief Construct the Newton Solver 
         * @param fespace the finite element space
         * @param disc the discretization
         * @param conv_criteria the convergence criteria for terminating the solve 
         * @param jac (optional) give a already set up matrix to use for jacobian storage 
         * @param comm (optional) the MPI Communicator defaults to MPI_COMM_WORLD
         */
        PetscNewton(
            FE::FESpace<T, IDX, ndim> &fespace,
            disc_class &disc,
            const ConvergenceCriteria<T, IDX> &conv_criteria,
            Mat jac = nullptr,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : fespace(fespace), disc(disc), conv_criteria{conv_criteria}, jac{jac}
        {
            std::size_t local_res_size = fespace.dg_offsets.calculate_size_requirement(disc_class::dnv_comp);
            std::size_t local_u_size = local_res_size;
            // Create and set up the matrix if not given 
            if(jac == nullptr){
                MatCreate(comm, &jac);
                MatSetSizes(jac, local_res_size, local_res_size, PETSC_DETERMINE, PETSC_DETERMINE);
                MatSetFromOptions(jac);
            }

            // Create and set up the vectors
            VecCreate(comm, &res_data);
            VecSetSizes(res_data, local_res_size, PETSC_DETERMINE);
            VecSetFromOptions(res_data);
            

            VecCreate(comm, &du_data);
            VecSetSizes(du_data, local_u_size, PETSC_DETERMINE);
            VecSetFromOptions(du_data);
        }

    };

    /// Deduction guides
    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &) -> PetscNewton<T, IDX, ndim, disc_class>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat) -> PetscNewton<T, IDX, ndim, disc_class>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat, MPI_Comm) -> PetscNewton<T, IDX, ndim, disc_class>;
}
