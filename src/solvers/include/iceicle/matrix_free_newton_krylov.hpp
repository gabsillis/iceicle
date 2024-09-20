/// @brief Matrix-free Newton-Krylov solver
///
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/fespace/fespace.hpp"
#include <iceicle/nonlinear_solver_utils.hpp>
#include <limits>
#include <cmath>
#include <iomanip>
#include <petsc.h>
#include <petscmat.h>
#include <petscsys.h>
#include "iceicle/form_residual.hpp"
#include "iceicle/petsc_interface.hpp"
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/component_span.hpp"

namespace iceicle::solvers {


    namespace impl {

        /// @brief Context for calculating the jacobian with directional derivatives
        template<class T, class IDX, int ndim, class disc_class, class uLayoutPolicy>
        struct JacobianContext {
            /// @brief the finite element space
            FESpace<T, IDX, ndim>& fespace;

            /// @brief the discretization
            disc_class& disc;

            /// @brief the geometry mapping
            const geo_dof_map<T, IDX, ndim>& geo_map;

            /// @brief the current state vector
            fespan<T, uLayoutPolicy> u;

            /// @brief a span over the calculated full residual
            Vec res;
        };

        template<class T, class IDX, int ndim, class disc_class, class uLayoutPolicy>
        inline 
        auto jacobian_mf_op(Mat A, Vec p, Vec y) 
        -> PetscErrorCode 
        {
            static constexpr T epsilon = std::sqrt(std::numeric_limits<T>::epsilon());

            // Set up the context
            JacobianContext<T, IDX, ndim, disc_class, uLayoutPolicy> *ctx;
            PetscFunctionBeginUser;
            PetscCall(MatShellGetContext(A, &ctx));

            FESpace<T, IDX, ndim>& fespace = ctx->fespace;
            disc_class& disc = ctx->disc;
            const geo_dof_map<T, IDX, ndim>& geo_map = ctx->geo_map;
            fespan<T, uLayoutPolicy> u = ctx->u;
            Vec res = ctx->res;

            // create all the layouts
            fe_layout_right dg_layout{fespace.dg_map, tmp::to_size<disc_class::nv_comp>()};
            geo_data_layout x_layout{geo_map};
            ic_residual_layout<T, IDX, ndim, disc_class::nv_comp> ic_layout{geo_map};

            // get the geometric parameterization
            std::vector<T> xdata(x_layout.size());
            component_span x{xdata, x_layout};
            extract_geospan(*(fespace.meshptr), x);

            // setup peturbed residuals 
            std::vector<T> resp(dg_layout.size() + ic_layout.size());
            fespan res_dg{resp, dg_layout};
            dofspan res_mdg{std::span{resp.begin() + dg_layout.size(), resp.end()}, ic_layout};

            // perform the peturbation
            std::vector<T> xdata_peturb = xdata;
            std::vector<T> udata_peturb(dg_layout.size());
            fespan up{udata_peturb, dg_layout};
            copy_fespan(u, up);
            component_span xp{xdata_peturb, x_layout};
            {
                petsc::VecSpan pview{p};
                fespan du{pview, dg_layout};
                component_span dx{pview.data() + dg_layout.size(), x_layout};
                axpy(epsilon, du, up);
                axpy(epsilon, dx, xp);
            }


            // apply the peturbed geometric parameterization to the mesh 
            update_mesh(xp, *(fespace.meshptr));

            // form the peturbed residual
            form_residual(fespace, disc, up, res_dg);
            form_mdg_residual(fespace, disc, up, geo_map, res_mdg);

            // directional derivative
            {
                petsc::VecSpan resview{res};
                petsc::VecSpan yview{y};
                for(IDX i = 0; i < resp.size(); ++i){
                    yview[i] = (resp[i] - resview[i]) / epsilon;
                }
            }

            // revert mesh peturbation
            update_mesh(x, *(fespace.meshptr));
            PetscFunctionReturn(EXIT_SUCCESS);
        }
    }

    template<class T, class IDX, int ndim, class disc_class, class ls_type = no_linesearch<T, IDX>>
    struct MFNK {

        /// @brief the finite element space
        FESpace<T, IDX, ndim>& fespace;

        /// @brief the discretization
        disc_class& disc;

        /// @brief the convergence criteria
        ConvergenceCriteria<T, IDX> conv_criteria;

        /// @brief the linesearch 
        const ls_type& linesearch;

        /// @brief the geometry mapping
        const geo_dof_map<T, IDX, ndim>& geo_map;

        /// @brief if this is a positive integer 
        /// Then the diagnostics callback will be called every idiag timesteps
        /// (k % idiag == 0)
        IDX idiag = -1;

        /// @brief set the verbosity level to print out different diagnostic information
        ///
        /// Level 0: no extra output
        /// Level 1: 
        /// Level 2:
        /// Level 3:
        /// Level 4: 
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

        MFNK(
            FESpace<T, IDX, ndim>& fespace,
            disc_class& disc,
            ConvergenceCriteria<T, IDX> conv_criteria,
            const ls_type& linesearch,
            const geo_dof_map<T, IDX, ndim>& geo_map
        ) : fespace(fespace), disc(disc), conv_criteria(conv_criteria),
            linesearch(linesearch), geo_map(geo_map)
        {}

        /// @brief solve the PDE
        template<class uLayoutPolicy>
        auto solve(fespan<T, uLayoutPolicy> u)
        -> IDX 
        {
            static constexpr int neq = disc_class::nv_comp;

            // define data layouts
            fe_layout_right u_layout{fespace.dg_map, std::integral_constant<std::size_t, neq>{}};
            geo_data_layout geo_layout{geo_map};
            ic_residual_layout<T, IDX, ndim, neq> ic_layout{geo_map};

            // setup petsc matrix and residual
            Vec r, du;
            VecCreate(PETSC_COMM_WORLD, &r);
            VecSetSizes(r, u_layout.size() + ic_layout.size(), PETSC_DETERMINE);
            VecSetFromOptions(r);

            VecCreate(PETSC_COMM_WORLD, &du);
            VecSetSizes(du, u_layout.size() + geo_layout.size(), PETSC_DETERMINE);
            VecSetFromOptions(du);

            Mat J;
            impl::JacobianContext<T, IDX, ndim, disc_class, uLayoutPolicy> j_ctx{
                .fespace = fespace,
                .disc = disc,
                .geo_map = geo_map,
                .u = u,
                .res = r
            };

            MatCreate(PETSC_COMM_WORLD, &J);
            MatSetSizes(J, u_layout.size() + geo_layout.size(), u_layout.size() + ic_layout.size(),
                    PETSC_DETERMINE, PETSC_DETERMINE);
            MatSetType(J, MATSHELL);
            MatSetUp(J);
            MatShellSetOperation(J, MATOP_MULT, 
                    (void (*)()) impl::jacobian_mf_op<T, IDX, ndim, disc_class, uLayoutPolicy>);
            MatShellSetContext(J, (void *) &j_ctx);

            // Create the linear solver and preconditioner
            KSP ksp;
            PC pc;
            PetscCallAbort(PETSC_COMM_WORLD, KSPCreate(PETSC_COMM_WORLD, &ksp));
            PetscCallAbort(PETSC_COMM_WORLD, KSPSetFromOptions(ksp));

            // default preconditioner
            PetscCallAbort(PETSC_COMM_WORLD, KSPGetPC(ksp, &pc));
            PCSetType(pc, PCNONE);

            { // get the initial residual 
                petsc::VecSpan resview{r};
                fespan res_pde{resview, u_layout};
                dofspan res_mdg{resview.data() + u_layout.size(), ic_layout};

                form_residual(fespace, disc, u, res_pde);
                form_mdg_residual(fespace, disc, u, geo_map, res_mdg);
            }

            // =======================
            // = Main Iteration Loop =
            // =======================
            IDX k;
            for(k = 0; k < conv_criteria.kmax; ++k){

                // Solve for the step
                MatAssemblyBegin(J, MAT_FINAL_ASSEMBLY);
                MatAssemblyEnd(J, MAT_FINAL_ASSEMBLY);
                PetscCallAbort(PETSC_COMM_WORLD, KSPSetOperators(ksp, J, J));
                PetscCallAbort(PETSC_COMM_WORLD, KSPSolve(ksp, r, du));

                // keep the old geometry data around
                std::vector<T> xdata(geo_layout.size());
                component_span x{xdata, geo_layout};
                extract_geospan(*(fespace.meshptr), x);

                T rnorm_step;
                // linesearchin time!
                T alpha = linesearch([&](T alpha_arg){
                    
                        // === Set up working arrays for linesearch update ===

                        // u step for linesearch
                        std::vector<T> u_step_storage(u.size());
                        fespan u_step{u_step_storage.data(), u.get_layout()};
                        copy_fespan(u, u_step);

                        // x step for linesearch
                        std::vector<T> x_step_storage = xdata; // copy over the coordinates as well
                        component_span x_step{x_step_storage, geo_layout};

                        // working array for linesearch residuals
                        std::vector<T> r_work_storage(u.size());
                        fespan res_work{r_work_storage.data(), u.get_layout()};

                        std::vector<T> r_mdg_work_storage(ic_layout.size());
                        dofspan mdg_res{r_mdg_work_storage, ic_layout};

                        // === Compute the Update ===

                        petsc::VecSpan du_view{du};
                        fespan du{du_view.data(), u.get_layout()};
                        axpy(-alpha_arg, du, u_step);

                        // x update
                        component_span dx{du_view.data() + u.size(), geo_layout};
                        axpy(-alpha_arg, dx, x);
                        // apply the x coordinates to the mesh
                        update_mesh(x, *(fespace.meshptr));

                        // === Get the residuals ===
                        form_residual(fespace, disc, u_step, res_work);
                        form_mdg_residual(fespace, disc, u_step, geo_map, mdg_res);
                        T rnorm = res_work.vector_norm() + mdg_res.vector_norm();
                        if(!std::isfinite(rnorm)) return 1e100;
                        rnorm_step = rnorm; // extract it

                        // Add any penalties
                        return rnorm;
                });

                // perform the update 
                petsc::VecSpan du_view{du};
                fespan duspan{du_view, u.get_layout()};
                axpy(-alpha, duspan, u);
                component_span dx{du_view.data() + u.size(), geo_layout};
                axpy(-alpha, dx, x);
                // apply the x coordinates to the mesh
                update_mesh(x, *(fespace.meshptr));

                { // get the updated residual 
                    petsc::VecSpan resview{r};
                    fespan res_pde{resview, u_layout};
                    dofspan res_mdg{resview.data() + u_layout.size(), ic_layout};

                    form_residual(fespace, disc, u, res_pde);
                    form_mdg_residual(fespace, disc, u, geo_map, res_mdg);
                }

                // get the residual norm
                T rk;
                PetscCallAbort(PETSC_COMM_WORLD, VecNorm(r, NORM_2, &rk));

                // Diagnostics 
                if(idiag > 0 && k % idiag == 0) {
                    diag_callback(k, r, du);
                }

                // visualization
                if(ivis > 0 && k % ivis == 0) {
                    vis_callback(k, r, du);
                }

                // test convergence
                if(conv_criteria.done_callback(rk)) break;
            }

            // ===========
            // = Cleanup =
            // ===========
            VecDestroy(&r);
            VecDestroy(&du);
            MatDestroy(&J);
            KSPDestroy(&ksp);

            return k;
        }


    };
}
