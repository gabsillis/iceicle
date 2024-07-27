/// @brief Levenberg-Marquardt implementation following Ching et al. 
/// The moving discontinuous Galerkin method with interface condition enforcement for the simulation of hypersonic, viscous flows
/// Computer Methods in Applied Mechanics and Engineering 2024
///
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/fe_function/component_span.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/nonlinear_solver_utils.hpp"
#include "iceicle/petsc_interface.hpp"

#include <petsc.h>

#include <iostream>
#include <petscsys.h>

namespace iceicle::solvers {
    
    template<class T, class IDX, int ndim, class disc_class, class ls_type = no_linesearch<T, IDX>>
    class CorriganLM {
        public:

        
        // ================
        // = Data Members =
        // ================

        /// @brief reference to the fespace to use
        FESpace<T, IDX, ndim>& fespace;

        /// @brief reference to the discretization to use
        disc_class& disc;

        /// @brief the convergence crieria
        ///
        /// determines whether the solver should terminate
        const ConvergenceCriteria<T, IDX>& conv_criteria;

        /// @brief the linesearch strategy
        const ls_type& linesearch;

        /// @brief map of geometry dofs to consider for interface conservation enforcement
        const geo_dof_map<T, IDX, ndim>& geo_map;

        // === Petsc Data Members ===

        /// @brief the Jacobian Matrix
        Mat jac;

        /// @brief the residual vector 
        Vec res_data;

        /// @brief the solution update
        Vec du_data;

        /// @brief J transpose times r 
        Vec Jtr;

        /// @brief krylov solver 
        KSP ksp;

        /// @brief preconditioner
        PC pc;

        // === Nonlinear Solver Behavior ===

        /// @brief if this is a positive integer 
        /// Then the diagnostics callback will be called every idiag timesteps
        /// (k % idiag == 0)
        IDX idiag = -1;

        /// @brief diagnostics function 
        ///
        /// very minimal by default other options are defined in this header
        /// or a custom function can be made 
        ///
        /// Passes the current iteration number, the residual vector, and the du vector
        std::function<void(IDX, Vec, Vec)> diag_callback = []
            (IDX k, Vec res_data, Vec du_data)
        {
            int iproc;
            MPI_Comm_rank(PETSC_COMM_WORLD, &iproc); if(iproc == 0){
                std::cout << "Diagnostics for iteration: " << k << std::endl;
            }
            if(iproc == 0) std::cout << "Residual: " << std::endl;
            PetscCallAbort(PETSC_COMM_WORLD, VecView(res_data, PETSC_VIEWER_STDOUT_WORLD));
            if(iproc == 0) std::cout << std::endl << "du: " << std::endl;
            PetscCallAbort(PETSC_COMM_WORLD, VecView(du_data, PETSC_VIEWER_STDOUT_WORLD));
            if(iproc == 0) std::cout << "------------------------------------------" << std::endl << std::endl; 
        };

        /// @brief set to true if you want to explicitly form the J^TJ + lambda * I matrix
        ///
        /// This may greatly reduce the sparsity
        const bool explicitly_form_subproblem;

        // === Regularization Parameters ===
        
        /// @brief regularization for pde dofs
        T lambda_u = 1e-7;

        /// @brief anisotropic lagrangian regularization 
        T lambda_lag = 1e-5;

        /// @brief Curvature penalization
        T lambda_1 = 1e-3;

        /// @brief Grid pentalty regularization 
        T lambda_b = 1e-2;

        /// @brief power of anisotropic metric
        T alpha = -1;

        /// @brief power for principle stretching magnitude
        T beta = 3;

        /// @brief the minimum allowable jacobian determinant
        T J_min = 1e-10;

        public:

        // ================
        // = Constructors =
        // ================
        
        CorriganLM(
            FESpace<T, IDX, ndim>& fespace,
            disc_class& disc, 
            const ConvergenceCriteria<T, IDX>& conv_criteria,
            const ls_type& linesearch,
            const geo_dof_map<T, IDX, ndim>& geo_map,
            bool explicitly_form_subproblem = false
        ) : fespace{fespace}, disc{disc}, conv_criteria{conv_criteria}, 
            linesearch{linesearch}, geo_map{geo_map},
            explicitly_form_subproblem{explicitly_form_subproblem}
        {
            static constexpr int neq = disc_class::nv_comp;

            // define data layouts
            fe_layout_right u_layout{fespace.dg_map, std::integral_constant<std::size_t, neq>{}};
            geo_data_layout geo_layout{geo_map};
            ic_residual_layout<T, IDX, ndim, neq> ic_layout{geo_map};

            // determine the system sizes on the local processor
            PetscInt local_u_size = u_layout.size() + geo_layout.size();
            PetscInt local_res_size = u_layout.size() + ic_layout.size();

            // Create and set up the jacobian matrix 
            MatCreate(PETSC_COMM_WORLD, &jac);
            MatSetSizes(jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetFromOptions(jac);
            MatSetUp(jac);

            // Create and set up vectors
            // Create and set up the vectors
            VecCreate(PETSC_COMM_WORLD, &res_data);
            VecSetSizes(res_data, local_res_size, PETSC_DETERMINE);
            VecSetFromOptions(res_data);
            

            VecCreate(PETSC_COMM_WORLD, &Jtr);
            VecSetSizes(Jtr, local_u_size, PETSC_DETERMINE);
            VecSetFromOptions(Jtr);

            VecCreate(PETSC_COMM_WORLD, &du_data);
            VecSetSizes(du_data, local_u_size, PETSC_DETERMINE);
            VecSetFromOptions(du_data);

            // Create the linear solver and preconditioner
            PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));

            // default to sor preconditioner
            PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
            PCSetType(pc, PCNONE);

            // Get user input (can override defaults set above)
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));
        }

        // ====================
        // = Member Functions =
        // ====================

        template<class uLayoutPolicy>
        auto solve(fespan<T, uLayoutPolicy> u) -> IDX {

            static constexpr int neq = disc_class::nv_comp;

            // define data layouts
            fe_layout_right u_layout{fespace.dg_map, std::integral_constant<std::size_t, neq>{}};
            geo_data_layout geo_layout{geo_map};
            ic_residual_layout<T, IDX, ndim, neq> ic_layout{geo_map};

            // get the current coordinate data
            std::vector<T> coord_data(geo_layout.size());
            component_span coord{coord_data, geo_layout};
            extract_geospan(*(fespace.meshptr), coord);

            // get initial residual and jacobian 
            {
                petsc::VecSpan res_view{res_data};
                fespan res{res_view.data(), u.get_layout()};
                form_petsc_jacobian_fd(fespace, disc, u, res, jac);
                dofspan mdg_res{res_view.data() + u_layout.size(), ic_layout};
                form_petsc_mdg_jacobian_fd(fespace, disc, u, coord, mdg_res, jac);
            }


        }
    };

}
