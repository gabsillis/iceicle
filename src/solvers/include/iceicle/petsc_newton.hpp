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
#include "iceicle/mdg_utils.hpp"
#include <format>
#include <functional>
#include <iomanip>
#include <iostream>
#include <petscerror.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>
#include <petscpctypes.h>
#include <petscsys.h>
#include <petscsystypes.h>
#include <petscvec.h>
#include <petscviewer.h>

namespace iceicle::solvers {

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

        public:

        // ============
        // = Typedefs =
        // ============
        using value_type = T;
        using index_type = IDX;

        /// @brief store a reference to the fespace being used 
        FESpace<T, IDX, ndim> &fespace;

        /// @brief store a reference to the discretization being solved
        disc_class &disc;

        /// @brief the linesearch strategy
        const ls_type& linesearch;

        /// @brief the convergence Criteria
        /// determines whether the solver should terminate
        ConvergenceCriteria<T, IDX> conv_criteria;

        /// @brief if this is a positive integer 
        /// Then the diagnostics callback will be called every idiag timesteps
        /// (k % idiag == 0)
        IDX idiag = -1;

        /// @brief set the verbosity level to print out different diagnostic information
        ///
        /// Level 0: no extra output
        /// Level 1: 
        ///   - linesearch iterations and multiplier
        /// Level 2:
        /// Level 3:
        /// Level 4: 
        ///   - x_step for each dof and the corresponnding global node
        IDX verbosity = 0;

        /// @brief diagnostics function 
        /// very minimal by default other options are defined in this header
        /// or a custom function can be made 
        ///
        /// Passes a reference to this, the current iteration number, the residual vector, and the du vector
        std::function<void(IDX, Vec, Vec)> diag_callback = []
            (IDX k, Vec res_data, Vec du_data)
        {
            int iproc;
            MPI_Comm_rank(PETSC_COMM_WORLD, &iproc);
            if(iproc == 0){
                std::cout << "Diagnostics for iteration: " << k << std::endl;
            }
            if(iproc == 0) std::cout << "Residual: " << std::endl;
            PetscCallAbort(PETSC_COMM_WORLD, VecView(res_data, PETSC_VIEWER_STDOUT_WORLD));
            if(iproc == 0) std::cout << std::endl << "du: " << std::endl;
            PetscCallAbort(PETSC_COMM_WORLD, VecView(du_data, PETSC_VIEWER_STDOUT_WORLD));
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
        std::function<void(IDX, Vec, Vec)> vis_callback = []
            (IDX k, Vec res_data, Vec du_data)
        {
            T res_norm;
            PetscCallAbort(PETSC_COMM_WORLD, VecNorm(res_data, NORM_2, &res_norm));
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
         * in standard (non-mdg mode) - the nodeset option is empty
         *
         * @param fespace the finite element space
         * @param disc the discretization
         * @param conv_criteria the convergence criteria for terminating the solve 
         */
        PetscNewton(
            FESpace<T, IDX, ndim> &fespace,
            disc_class &disc,
            const ConvergenceCriteria<T, IDX> &conv_criteria,
            const ls_type& linesearch
        ) : fespace(fespace), disc(disc), linesearch{linesearch}, 
            conv_criteria{conv_criteria} 
        {
            PetscInt local_res_size = fespace.dg_map.calculate_size_requirement(disc_class::dnv_comp);
            PetscInt local_u_size = local_res_size;
            // Create and set up the matrix if not given 
            MatCreate(PETSC_COMM_WORLD, &(this->jac));
            MatSetSizes(this->jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetFromOptions(this->jac);
            MatSetUp(this->jac);

            // Create and set up the vectors
            VecCreate(PETSC_COMM_WORLD, &res_data);
            VecSetSizes(res_data, local_res_size, PETSC_DETERMINE);
            VecSetFromOptions(res_data);
            

            VecCreate(PETSC_COMM_WORLD, &du_data);
            VecSetSizes(du_data, local_u_size, PETSC_DETERMINE);
            VecSetFromOptions(du_data);

            // Create the linear solver and preconditioner
            PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));

            // default to sor preconditioner
            PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
            PCSetType(pc, PCSOR);

            // Get user input (can override defaults set above)
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));
        }

        PetscNewton(
            FESpace<T, IDX, ndim> &fespace,
            disc_class &disc,
            const ConvergenceCriteria<T, IDX> &conv_criteria
        ) : PetscNewton(fespace, disc, conv_criteria, no_linesearch<T, IDX>{}) {}

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
         * @return the number of iterations performed (can discard)
         */
        template<class uLayoutPolicy>
        auto solve(fespan<T, uLayoutPolicy> u) -> IDX {

            // TODO: find a cleaner way to do this


            // get the initial residual and jacobian
            {
                petsc::VecSpan res_view{res_data};
                fespan res{res_view.data(), u.get_layout()};
                form_petsc_jacobian_fd(fespace, disc, u, res, jac);
//                std::cout << "res_initial" << std::endl;
//                std::cout << res;
            } // end scope of res_view

            // set the initial residual norm
            PetscCallAbort(PETSC_COMM_WORLD, VecNorm(res_data, NORM_2, &(conv_criteria.r0)));

            IDX k;
            for(k = 0; k < conv_criteria.kmax; ++k){

                // get node radii 
                std::vector<T> node_radii{node_freedom_radii(fespace)};

                // solve for du 
                MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);

                // view jacobian matrix
                if(verbosity >= 4){
                    PetscViewer jacobian_viewer;
                    PetscViewerASCIIOpen(PETSC_COMM_WORLD, ("iceicle_data/jacobian_view" + std::to_string(k) + ".dat").c_str(), &jacobian_viewer);
                    PetscViewerPushFormat(jacobian_viewer, PETSC_VIEWER_ASCII_DENSE);
                    MatView(jac, jacobian_viewer);
                    PetscViewerDestroy(&jacobian_viewer);
    //                MatView(jac, PETSC_VIEWER_STDOUT_WORLD); // for debug purposes
                }

                PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, this->jac, this->jac));
                PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp, res_data, du_data));

                // update u
                if constexpr (std::is_same_v<ls_type, no_linesearch<T, IDX>>){
                    petsc::VecSpan du_view{du_data};
                    fespan du{du_view.data(), u.get_layout()};
                    axpy(-1.0, du, u);
                } else {
                    // its linesearchin time!

                    // view into the calculated newton step for u and x
                    petsc::VecSpan du_view{du_data};
                    fespan du{du_view.data(), u.get_layout()};

                    // u step for linesearch
                    std::vector<T> u_step_storage(u.size());
                    fespan u_step{u_step_storage.data(), u.get_layout()};
                    copy_fespan(u, u_step);

                    // working array for linesearch residuals
                    std::vector<T> r_work_storage(u.size());
                    fespan res_work{r_work_storage.data(), u.get_layout()};

                    std::vector<T> r_mdg_work_storage{};


                    T alpha = linesearch([&](T alpha_arg){
                        static constexpr T BIG_RESIDUAL = 1e9;

                        // apply the step scaled by linesearch param
                        copy_fespan(u, u_step);
                        axpy(-alpha_arg, du, u_step);

                        form_residual(fespace, disc, u_step, res_work);
                        T rnorm = res_work.vector_norm();

                        // verbose output
                        if(verbosity >= 1){
                            std::cout << "linesearch: alpha = " << alpha_arg << " | linesearch residual = " << rnorm << std::endl;
                        }

                        // safeguard the cost function for linesearch 
                        if(std::isfinite(rnorm)){
                            return rnorm;
                        } else {
                            if(verbosity >= 1) {
                                std::cout << "linesearch: non-finite residual" << std::endl;
                            }
                            return BIG_RESIDUAL;
                        }
                    });

                    if(verbosity >= 1) std::cout << "linesearch: selected alpha = " << alpha << std::endl;

                    // apply the step times linesearch multiplier to u and x
                    axpy(-alpha, du, u);
                }

                // Get the new residual and Jacobian (for the next step)
                {
                    petsc::VecSpan res_view{res_data};
                    fespan res{res_view.data(), u.get_layout()};
                    MatZeroEntries(jac); // zero out the jacobian
                    form_petsc_jacobian_fd(fespace, disc, u, res, jac);
                } // end scope of res_view

                // get the residual norm
                T rk;
                PetscCallAbort(PETSC_COMM_WORLD, VecNorm(res_data, NORM_2, &rk));
                
                // Diagnostics 
                if(idiag > 0 && k % idiag == 0) {
                    diag_callback(k, res_data, du_data);
                }

                // visualization
                if(ivis > 0 && k % ivis == 0) {
                    vis_callback(k, res_data, du_data);
                }

                // test convergence
                if(conv_criteria.done_callback(rk)) break;

            }
            return k;
        }

        ~PetscNewton(){
            VecDestroy(&res_data);
            VecDestroy(&du_data);
            MatDestroy(&jac);
        }

    };

    /// Deduction guides
    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&, Mat) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    template<class T, class IDX, int ndim, class disc_class, class ls_type>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, const ls_type&, Mat, MPI_Comm) -> PetscNewton<T, IDX, ndim, disc_class, ls_type>;

    /// Deduction guides
    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

    template<class T, class IDX, int ndim, class disc_class>
    PetscNewton(FESpace<T, IDX, ndim> &, disc_class &,
        const ConvergenceCriteria<T, IDX> &, Mat, MPI_Comm) -> PetscNewton<T, IDX, ndim, disc_class, no_linesearch<T, IDX>>;

}
