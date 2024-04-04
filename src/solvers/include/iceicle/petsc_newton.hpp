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
        /// @brief the set of selected nodes to do mdg with 
        /// Activates mdg mode if specified 
        /// Mdg mode adds node dofs 
        /// and interface conservation equations
        FE::nodeset_dof_map<IDX> mdg_nodeset;

        bool mdg_mode;

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

        /// @brief multiplier for node movement of the node radius
        T node_radius_mult = 0.1;

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
         * in standard (non-mdg mode) - the nodeset option is empty
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
        ) : fespace(fespace), disc(disc), linesearch{linesearch}, mdg_nodeset{}, mdg_mode{false},
            comm(comm), conv_criteria{conv_criteria}, jac{jac}
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

        /**
         * @brief Construct the Newton Solver in MDG mode 
         * NOTE: MDG Mode:
         * - operates on a set of nodes, which is a subset of all the nodes 
         *   this nodeset provided as an argument is copied into an optional parameter internally
         * - will solve for the node locations using additional interface conservation equations
         * - sets the number of interface conservation equations to ndim to recover a square system
         *   (for nonsquare systems we need a nonlinear optimization solver instead of newton-ls)
         * - requires a linesearch to be specified - because MDG is generally to poorly conditioned for 
         *   unlimited newton steps 
         *
         *  @param fespace the finite element space 
         *  @param disc the discretization 
         *  @param conv_criteria the convergence criteria for terminating the solve 
         *  @param nodeset the subset of nodes to use for mdg dofs 
         */
        PetscNewton(
            FE::FESpace<T, IDX, ndim>& fespace,
            disc_class& disc,
            const ConvergenceCriteria<T, IDX>& conv_criteria,
            const ls_type& linesearch,
            const FE::nodeset_dof_map<IDX>& nodeset
        ) : fespace{fespace}, disc{disc}, conv_criteria{conv_criteria}, linesearch{linesearch},
            mdg_nodeset{nodeset}, mdg_mode{true}, comm{MPI_COMM_WORLD}
        {
            PetscInt local_res_size = fespace.dg_map.calculate_size_requirement(disc_class::dnv_comp)
                + nodeset.selected_nodes.size() * ndim;
            PetscInt local_u_size = local_res_size;
            // Create and set up the matrix if not given 
            MatCreate(comm, &(this->jac));
            MatSetSizes(this->jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetFromOptions(this->jac);

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
        auto solve(FE::fespan<T, uLayoutPolicy> u) -> IDX {

            // TODO: find a cleaner way to do this

            // Node selection layout for MDG (empty unless mdg active)
            FE::node_selection_layout<IDX, ndim> mdg_layout{mdg_nodeset};
            // copy out the initial nodes
            std::vector<T> current_x_storage(mdg_layout.size());
            FE::dofspan current_x{current_x_storage, mdg_layout};
            FE::extract_node_selection_span(fespace.meshptr->nodes, current_x);

            // get the initial residual and jacobian
            {
                PETSC::VecSpan res_view{res_data};
                FE::fespan res{res_view.data(), u.get_layout()};
                form_petsc_jacobian_fd(fespace, disc, u, res, jac);
                if(mdg_mode){
                    // MDG Mode 

                    // MDG residual starts after fe residual
                    FE::dofspan mdg_res{res_view.data() + res.size(), mdg_layout};

                    form_petsc_mdg_jacobian_fd(fespace, disc, u, mdg_res, jac);
                }
//                std::cout << "res_initial" << std::endl;
//                std::cout << res;
            } // end scope of res_view

            // set the initial residual norm
            PetscCallAbort(comm, VecNorm(res_data, NORM_2, &(conv_criteria.r0)));

            IDX k;
            for(k = 0; k < conv_criteria.kmax; ++k){

                // keep around the current node locations
                FE::extract_node_selection_span(fespace.meshptr->nodes, current_x);

                // get node radii 
                std::vector<T> node_radii{FE::node_freedom_radii(fespace)};

                // solve for du 
                MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);

                PetscViewer jacobian_viewer;
                PetscViewerASCIIOpen(comm, ("jacobian_view" + std::to_string(k) + ".dat").c_str(), &jacobian_viewer);
                PetscViewerPushFormat(jacobian_viewer, PETSC_VIEWER_ASCII_DENSE);
                MatView(jac, jacobian_viewer);
                PetscViewerDestroy(&jacobian_viewer);
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

                    // view into the calculated newton step for u and x
                    const FE::nodeset_dof_map<IDX> &nodeset = mdg_layout.nodeset;
                    PETSC::VecSpan du_view{du_data};
                    FE::fespan du{du_view.data(), u.get_layout()};
                    FE::dofspan dx{du_view.data() + u.size(), mdg_layout};

                    // print out dx if verbosity is 4 or more
                    if(verbosity >= 4){
                        for(int idof = 0; idof < dx.ndof(); ++idof){
                            T node_limit = node_radius_mult * node_radii[nodeset.selected_nodes[idof]];
                            std::cout << "nodeset_dof: " << std::format("{:<4d}", idof)
                                << " | node_index: " << std::format("{:<4d}", nodeset.selected_nodes[idof])
                                << " | step_limit: " << std::format("{:>8f}", node_limit)
                                << " | dx: ";
                            for(int idim = 0; idim < ndim; ++idim)
                                std::cout << std::format("{:<16f}", dx[idof, idim]) << " ";
                            std::cout << std::endl;

                        }

                    }

                    // restrict dx by node_radius
//                    for(IDX idof = 0; idof < dx.ndof(); ++idof){
//                        T node_limit = node_radius_mult * node_radii[nodeset.selected_nodes[idof]];
//
//                        for(IDX iv = 0; iv < dx.nv(); ++iv){
//                            dx[idof, iv] = std::copysign(
//                                std::min(std::abs(dx[idof, iv]), node_limit),
//                                dx[idof, iv]
//                            );
//                        }
//                    }

                    
                    // compute a linesearch restriction by node radius
                    // with no linesearch you'll just be YOLOing 
                    // when it comes to node movement i guess
                    if constexpr (variable_alpha_ls<ls_type>){
                        T alpha_node_limit = linesearch.alpha_max;
                        for(int idof = 0; idof < dx.ndof(); ++idof){
                            T node_limit = node_radius_mult * node_radii[nodeset.selected_nodes[idof]];
                            alpha_node_limit = std::min(alpha_node_limit, node_limit);
                        }

                        linesearch.alpha_max = alpha_node_limit;
                        // start off at half of the max
                        linesearch.alpha_initial = 0.5 * alpha_node_limit;
                    }

                    // u step for linesearch
                    std::vector<T> u_step_storage(u.size());
                    FE::fespan u_step{u_step_storage.data(), u.get_layout()};
                    FE::copy_fespan(u, u_step);

                    // x step for linesearch
                    std::vector<T> x_step_storage(mdg_layout.size());
                    FE::dofspan x_step{x_step_storage, mdg_layout};

                    // working array for linesearch residuals
                    std::vector<T> r_work_storage(u.size());
                    FE::fespan res_work{r_work_storage.data(), u.get_layout()};

                    std::vector<T> r_mdg_work_storage{};

                    T alpha = linesearch([&](T alpha_arg){
                        static constexpr T BIG_RESIDUAL = 1e9;

                        // apply the step scaled by linesearch param
                        FE::copy_fespan(u, u_step);
                        FE::axpy(-alpha_arg, du, u_step);
                        FE::extract_node_selection_span(fespace.meshptr->nodes, x_step);
                        FE::axpy(-alpha_arg, dx, x_step);

                        // copy moved nodes 
                        FE::scatter_node_selection_span(1.0, x_step, 0.0, fespace.meshptr->nodes);

                        form_residual(fespace, disc, u_step, res_work);
                        T rnorm = res_work.vector_norm();

                        // Add MDG contribution if applicable
                        if(mdg_mode){
                            // MDG Mode 
                            r_mdg_work_storage.resize(mdg_layout.size());

                            // set the x points from x_step

                            FE::dofspan mdg_res{r_mdg_work_storage, mdg_layout};

                            form_mdg_residual(fespace, disc, u_step, mdg_res);
                            rnorm += mdg_res.vector_norm();
                        }

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

                        // revert nodes
                        FE::scatter_node_selection_span(1.0, current_x, 0.0, fespace.meshptr->nodes);
                    });

                    if(verbosity >= 1) std::cout << "linesearch: selected alpha = " << alpha << std::endl;

                    // apply the step times linesearch multiplier to u and x
                    FE::axpy(-alpha, du, u);
                    FE::extract_node_selection_span(fespace.meshptr->nodes, x_step);
                    FE::axpy(-alpha, dx, x_step);
                    FE::scatter_node_selection_span(1.0, x_step, 0.0, fespace.meshptr->nodes);
                }

                // Get the new residual and Jacobian (for the next step)
                {
                    PETSC::VecSpan res_view{res_data};
                    FE::fespan res{res_view.data(), u.get_layout()};
                    MatZeroEntries(jac); // zero out the jacobian
                    form_petsc_jacobian_fd(fespace, disc, u, res, jac);
                    if(mdg_mode){
                        // MDG Mode 
                        FE::node_selection_layout<IDX, ndim> mdg_res_layout{mdg_nodeset};

                        // MDG residual starts after fe residual
                        FE::dofspan mdg_res{res_view.data() + res.size(), mdg_res_layout};

                        form_petsc_mdg_jacobian_fd(fespace, disc, u, mdg_res, jac);
                    }
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
