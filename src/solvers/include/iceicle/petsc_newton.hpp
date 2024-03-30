/**
 * @brief newton's method solvers that use petsc as a backend
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/form_petsc_jacobian.hpp"
#include "iceicle/form_residual.hpp"
#include "iceicle/nonlinear_solver_utils.hpp"
#include "iceicle/petsc_interface.hpp"
#include <functional>
#include <iomanip>
#include <iostream>
#include <mpi_proto.h>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscvec.h>
#include <petscviewer.h>

namespace ICEICLE::SOLVERS {

    /**
     * @brief Newton solver that uses Petsc for linear solvers 
     * @tparam T the floating point type 
     * @tparam IDX the index type
     * @tparam ndim the number of dimensions
     * @tparam disc_class the discretization
     * @tparam ls_type the linesearch type to use
     */
    template<class T, class IDX, int ndim, class disc_class, class ls_type = no_linesearch<T, IDX>>
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

        /// @brief the linear solver 
        KSP ksp;

        /// @brief the preconditioner
        PC pc;

        /// @brief the set of selected nodes to do mdg with 
        /// Activates mdg mode if specified 
        /// Mdg mode adds node dofs 
        /// and interface conservation equations
        std::optional<FE::nodeset_dof_map<IDX>> mdg_nodeset;

        public:
        /// @brief store a reference to the fespace being used 
        FE::FESpace<T, IDX, ndim> &fespace;

        /// @brief store a reference to the discretization being solved
        disc_class &disc;

        /// @brief the linesearch strategy
        ls_type linesearch;

        /// @brief store the MPI communicator used at construction
        MPI_Comm comm;

        /// @brief the convergence Criteria
        /// determines whether the solver should terminate
        ConvergenceCriteria<T, IDX> conv_criteria;

        /// @brief if this is a positive integer 
        /// Then the diagnostics callback will be called every idiag timesteps
        /// (k % idiag == 0)
        IDX idiag = -1;

        /// @brief diagnostics function 
        /// very minimal by default other options are defined in this header
        /// or a custom function can be made 
        ///
        /// Passes a reference to this, the current iteration number, the residual vector, and the du vector
        std::function<void(PetscNewton &, IDX, Vec, Vec)> diag_callback = []
            (PetscNewton &solver, IDX k, Vec res_data, Vec du_data)
        {
            int iproc;
            MPI_Comm_rank(solver.comm, &iproc);
            if(iproc == 0){
                std::cout << "Diagnostics for iteration: " << k << std::endl;
            }
            if(iproc == 0) std::cout << "Residual: " << std::endl;
            PetscCallAbort(solver.comm, VecView(res_data, PETSC_VIEWER_STDOUT_WORLD));
            if(iproc == 0) std::cout << std::endl << "du: " << std::endl;
            PetscCallAbort(solver.comm, VecView(du_data, PETSC_VIEWER_STDOUT_WORLD));
            if(iproc == 0) std::cout << "------------------------------------------" << std::endl << std::endl; 
        };

        /// @brief if this is a positive integer 
        /// Then the diagnostics callback will be called every ivis timesteps
        /// (k % ivis == 0)        
        IDX ivis = -1;

        /// @brief the callback function for visualization during solve()
        /// is given a reference to this when called 
        /// default is to print out a l2 norm of the residual data array
        /// Passes a reference to this, the current iteration number, the residual vector, and the du vector
        std::function<void(PetscNewton &, IDX, Vec, Vec)> vis_callback = []
            (PetscNewton &solver, IDX k, Vec res_data, Vec du_data)
        {
            T res_norm;
            PetscCallAbort(solver.comm, VecNorm(res_data, NORM_2, &res_norm));
            std::cout << std::setprecision(8);
            std::cout << "itime: " << std::setw(6) << k
                << " | residual l2: " << std::setw(14) << res_norm
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
         *            NOTE: this takes ownership and will destroy with destructor
         *
         * @param comm (optional) the MPI Communicator defaults to MPI_COMM_WORLD
         */
        PetscNewton(
            FE::FESpace<T, IDX, ndim> &fespace,
            disc_class &disc,
            const ConvergenceCriteria<T, IDX> &conv_criteria,
            const ls_type& linesearch,
            Mat jac = nullptr,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : fespace(fespace), disc(disc), linesearch{linesearch}, comm(comm), conv_criteria{conv_criteria}, jac{jac}
        {
            PetscInt local_res_size = fespace.dg_map.calculate_size_requirement(disc_class::dnv_comp);
            PetscInt local_u_size = local_res_size;
            // Create and set up the matrix if not given 
            if(jac == nullptr){
                MatCreate(comm, &(this->jac));
                MatSetSizes(this->jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
                MatSetFromOptions(this->jac);
            }

            // Create and set up the vectors
            VecCreate(comm, &res_data);
            VecSetSizes(res_data, local_res_size, PETSC_DETERMINE);
            VecSetFromOptions(res_data);
            

            VecCreate(comm, &du_data);
            VecSetSizes(du_data, local_u_size, PETSC_DETERMINE);
            VecSetFromOptions(du_data);

            // Create the linear solver and preconditioner
            PetscCallAbort(comm, KSPCreate(comm, &ksp));

            // default to sor preconditioner
            PetscCallAbort(comm, KSPGetPC(ksp, &pc));
            PCSetType(pc, PCSOR);

            // Get user input (can override defaults set above)
            PetscCallAbort(comm, KSPSetFromOptions(ksp));
        }

        PetscNewton(
            FE::FESpace<T, IDX, ndim> &fespace,
            disc_class &disc,
            const ConvergenceCriteria<T, IDX> &conv_criteria,
            Mat jac = nullptr,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : PetscNewton(fespace, disc, conv_criteria, no_linesearch<T, IDX>{}, jac, comm) {}
        // ====================
        // = Member Functions =
        // ====================

        /**
         * @brief solve the nonlinear pde defined by disc and fespace 
         * @tparam uLayoutPolicy the layout of the input solution 
         * NOTE: since u is modified it must use the default accessor policy
         *
         * @param [in/out] u the discretized solution coefficients. 
         * The given values are used as the initial guess to the newton method.
         * After this function, this holds the solution 
         */
        template<class uLayoutPolicy>
        void solve(FE::fespan<T, uLayoutPolicy> u){

            // get the initial residual and jacobian
            {
                PETSC::VecSpan res_view{res_data};
                FE::fespan res{res_view.data(), u.get_layout()};
                form_petsc_jacobian_fd(fespace, disc, u, res, jac);
//                std::cout << "res_initial" << std::endl;
//                std::cout << res;
            } // end scope of res_view

            // set the initial residual norm
            PetscCallAbort(comm, VecNorm(res_data, NORM_2, &(conv_criteria.r0)));

            for(IDX k = 0; k < conv_criteria.kmax; ++k){

                // solve for du 
                MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
//                MatView(jac, PETSC_VIEWER_STDOUT_WORLD); // for debug purposes
                PetscCallAbort(comm, KSPSetOperators(ksp, this->jac, this->jac));
                PetscCallAbort(comm, KSPSolve(ksp, res_data, du_data));

                // update u
                if constexpr (std::is_same_v<ls_type, no_linesearch<T, IDX>>){
                    PETSC::VecSpan du_view{du_data};
                    FE::fespan du{du_view.data(), u.get_layout()};
                    FE::axpy(-1.0, du, u);
                } else {
                    // its linesearchin time!
                    PETSC::VecSpan du_view{du_data};
                    FE::fespan du{du_view.data(), u.get_layout()};
                    std::vector<T> u_step_storage(u.size());
                    FE::fespan u_step{u_step_storage.data(), u.get_layout()};
                    FE::copy_fespan(u, u_step);

                    std::vector<T> r_work_storage(u.size());
                    FE::fespan res_work{r_work_storage.data(), u.get_layout()};

                    T alpha = linesearch([&](T alpha_arg){
                        static constexpr T BIG_RESIDUAL = 1e9;
                        FE::copy_fespan(u, u_step);
                        FE::axpy(-alpha_arg, du, u_step);
                        form_residual(fespace, disc, u_step, res_work);
                        // safeguard the cost function for linesearch 
                        T rnorm = res_work.vector_norm();
                        if(std::isfinite(rnorm)){
                            return rnorm;
                        } else {
                            return BIG_RESIDUAL;
                        }
                    });

                    FE::axpy(-alpha, du, u);
                }

                // Get the new residual and Jacobian (for the next step)
                {
                    PETSC::VecSpan res_view{res_data};
                    FE::fespan res{res_view.data(), u.get_layout()};
                    MatZeroEntries(jac); // zero out the jacobian
                    form_petsc_jacobian_fd(fespace, disc, u, res, jac);
                } // end scope of res_view

                // get the residual norm
                T rk;
                PetscCallAbort(comm, VecNorm(res_data, NORM_2, &rk));
                
                // Diagnostics 
                if(idiag > 0 && k % idiag == 0) {
                    diag_callback(*this, k, res_data, du_data);
                }

                // visualization
                if(ivis > 0 && k % idiag == 0) {
                    vis_callback(*this, k, res_data, du_data);
                }

                // test convergence
                if(conv_criteria.done_callback(rk)) break;

            }
        }

        ~PetscNewton(){
            VecDestroy(&res_data);
            VecDestroy(&du_data);
            MatDestroy(&jac);
        }

    };

    /// Deduction guides
    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&, Mat) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&, Mat, MPI_Comm) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    /// Deduction guides
    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FE::FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat, MPI_Comm) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

}
