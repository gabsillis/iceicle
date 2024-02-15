#include "HYPRE_parcsr_ls.h"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/hypre_interface.hpp"
#include "iceicle/petsc_interface.hpp"
#include "mdspan/mdspan.hpp"
#include "gtest/gtest.h"

TEST(test_hypre_interface, test_bounds){
    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 1;
    static constexpr int pn_basis = 1;
    static constexpr int neq = 1;

    // create a uniform mesh
    int nx = 2;
    int ny = 2;
    MESH::AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {nx, ny}, pn_geo);
    mesh.nodes.random_perturb(-0.4 * 1.0 / std::max(nx, ny), 0.4*1.0/std::max(nx, ny));

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, FE::FESPACE_ENUMS::LAGRANGE,
        FE::FESPACE_ENUMS::GAUSS_LEGENDRE, 
        ICEICLE::TMP::compile_int<pn_basis>()
    };

    using namespace ICEICLE::SOLVERS;

    auto comm = MPI_COMM_WORLD;
    // MPI info
    int nproc;
    int proc_id;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &proc_id);

    int ncomp = 2;
    HypreBounds bounds = compute_hypre_bounds(fespace, ncomp);
    int expected_size = ncomp * std::pow(pn_basis + 1, ndim) * nx * ny;
    ASSERT_EQ(bounds.ilower, proc_id * expected_size);
    ASSERT_EQ(bounds.iupper, (proc_id + 1) * expected_size - 1);
}

TEST(test_hypre_interface, test_hypre_vec){
    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 1;
    static constexpr int pn_basis = 1;
    static constexpr int neq = 1;

    // create a uniform mesh
    int nx = 2;
    int ny = 2;
    MESH::AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {nx, ny}, pn_geo);
    mesh.nodes.random_perturb(-0.4 * 1.0 / std::max(nx, ny), 0.4*1.0/std::max(nx, ny));

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, FE::FESPACE_ENUMS::LAGRANGE,
        FE::FESPACE_ENUMS::GAUSS_LEGENDRE, 
        ICEICLE::TMP::compile_int<pn_basis>()
    };

    using namespace ICEICLE::SOLVERS;

    auto comm = MPI_COMM_WORLD;
    // MPI info
    int nproc;
    int proc_id;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &proc_id);
    static constexpr int ncomp = 2;

    int total_components = fespace.ndof_dg() * ncomp;
    std::vector<T> u_data(fespace.ndof_dg() * ncomp); 
    std::iota(u_data.begin(), u_data.end(), 0);
    FE::fespan<T, FE::dg_layout<T, ncomp>> u(u_data.data(), fespace.dg_offsets);

    // hypre should now have the std::iota data
    HypreVec u_hypre{u};

    std::vector<T> u2_data(fespace.ndof_dg() * ncomp, 0); // 0 initialized
    FE::fespan<T, FE::dg_layout<T, ncomp>> u2(u2_data.data(), fespace.dg_offsets);

    // extract the data to u2
    u_hypre.extract_data(u2);

    for(int i = 0; i < total_components; ++i){
        ASSERT_EQ((T) i, u2.data()[i]);
    }
}

namespace stdex = std::experimental;

TEST(test_hypre_interface, test_hypre_mat){
    using namespace ICEICLE::SOLVERS;

    auto comm = MPI_COMM_WORLD;
    // MPI info
    int nproc;
    int proc_id;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &proc_id);
    HYPRE_BigInt sz = 6;
    HypreBounds mat_bounds{.ilower = sz * proc_id, .iupper = sz * (proc_id + 1) - 1};
    HypreMat hmat{mat_bounds};

    std::array vals1{
        0.0, 0.5, 1.0,
        3.0, 3.0, 4.0,
        2.0, 7.0, 6.0
    };

    stdex::mdspan arr1{vals1.data(), stdex::extents{3, 3}};

    std::array vals2 {
        2.0, 1.0,
        7.0, 2.0,
        1.0, 4.0
    };
    stdex::mdspan arr2{vals2.data(), stdex::extents{3, 2}};

    hmat.set_values(0, 0, arr1);
    hmat.add_values(2, 1, arr2);
    hmat.add_values(3, 3, arr1);
    HYPRE_IJMatrixAssemble(hmat);
// A =
//
//     0     5     1     0     0     0
//     3     3     4     0     0     0
//     2     9     7     0     0     0
//     0     7     2     0     5     1
//     0     1     4     3     3     4
//     0     0     0     2     7     6

    HypreVec xvec{mat_bounds};
    HypreVec bvec{mat_bounds};

    std::array bvals{
        1.0, 2.0, 2.0, 1.0, 3.0, 4.0
    };
    bvec.set_values(0, std::span{bvals});

    HYPRE_IJMatrixPrint(hmat, "test_matrix");

    HYPRE_Solver solver, precond;
    /* Run info - needed logging turned on */
    int num_iterations;
    double final_res_norm;
    int restart = 30;
      /* Create solver */
      HYPRE_ParCSRFlexGMRESCreate(MPI_COMM_WORLD, &solver);

      /* Set some parameters (See Reference Manual for more parameters) */
      HYPRE_FlexGMRESSetKDim(solver, restart);
      HYPRE_FlexGMRESSetMaxIter(solver, 5000); /* max iterations */
      HYPRE_FlexGMRESSetTol(solver, 1e-7); /* conv. tolerance */
      HYPRE_FlexGMRESSetPrintLevel(solver, 2); /* print solve info */
      HYPRE_FlexGMRESSetLogging(solver, 1); /* needed to get run info later */


      /* Now set up the AMG preconditioner and specify any parameters */
      HYPRE_BoomerAMGCreate(&precond);
      HYPRE_BoomerAMGSetPrintLevel(precond, 1); /* print amg solution info */
      HYPRE_BoomerAMGSetCoarsenType(precond, 1);
      HYPRE_BoomerAMGSetOldDefault(precond);
      HYPRE_BoomerAMGSetRelaxType(precond, 0); /* Jacobi */
      HYPRE_BoomerAMGSetNumSweeps(precond, 1);
      HYPRE_BoomerAMGSetTol(precond, 0.0); /* conv. tolerance zero */
      HYPRE_BoomerAMGSetMaxIter(precond, 1); /* do only one iteration! */

      /* Set the FlexGMRES preconditioner */
      HYPRE_FlexGMRESSetPrecond(solver, (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSolve,
                                (HYPRE_PtrToSolverFcn) HYPRE_BoomerAMGSetup, precond);

      /* Now setup and solve! */
      HYPRE_ParCSRFlexGMRESSetup(solver, hmat, bvec, xvec);
      HYPRE_ParCSRFlexGMRESSolve(solver, hmat, bvec, xvec);

      /* Run info - needed logging turned on */
      HYPRE_FlexGMRESGetNumIterations(solver, &num_iterations);
      HYPRE_FlexGMRESGetFinalRelativeResidualNorm(solver, &final_res_norm);


    if (proc_id == 0)
    {
        printf("\n");
        printf("Iterations = %d\n", num_iterations);
        printf("Final Relative Residual Norm = %e\n", final_res_norm);
        printf("\n");
    }

    /* Destory solver and preconditioner */
    HYPRE_ParCSRFlexGMRESDestroy(solver);
    HYPRE_BoomerAMGDestroy(precond);

    std::vector<double> xdata(6);
    xvec.extract_data(std::span{xdata});
//    ASSERT_DOUBLE_EQ( 0.75, xdata[0]);
//    ASSERT_DOUBLE_EQ( 0.25, xdata[1]);
//    ASSERT_DOUBLE_EQ(-0.25, xdata[2]);
//    ASSERT_DOUBLE_EQ( 0.45, xdata[3]);
//    ASSERT_DOUBLE_EQ(-0.20, xdata[4]);
//    ASSERT_DOUBLE_EQ( 0.75, xdata[5]);
}
