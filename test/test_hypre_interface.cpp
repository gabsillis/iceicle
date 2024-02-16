#include "HYPRE_parcsr_ls.h"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/hypre_interface.hpp"
#include "iceicle/petsc_interface.hpp"
#include "mdspan/mdspan.hpp"
#include "gtest/gtest.h"
#include <petscksp.h>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewer.h>

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

TEST(test_petsc_interface, test_matrix_utils){
    /// WARNING: FAILS WITH DEFAULT ILU PRECONDITIONER
    /// Probably because of zero diagonals that don't get resolved
    using namespace ICEICLE::PETSC;

    auto comm = MPI_COMM_WORLD;
    // MPI info
    int nproc;
    int proc_id;
    MPI_Comm_size(comm, &nproc);
    MPI_Comm_rank(comm, &proc_id);

    // create a 6x6 matrix on each processor
    Mat A;
    MatCreate(comm, &A);
    MatSetSizes(A, 6, 6, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetFromOptions(A);

    PetscInt global_index_start = 6 * proc_id;

    std::array vals1{
        0.0, 5.0, 1.0,
        3.0, 3.0, 4.0,
        2.0, 7.0, 6.0,
    };

    stdex::mdspan arr1{vals1.data(), stdex::extents{3, 3}};

    std::array vals2 {
        2.0, 1.0, 0.0, 0.0,
        7.0, 2.0, 0.0, 0.0,
        1.0, 4.0, 0.0, 0.0
    };
    stdex::mdspan arr2{vals2.data(), stdex::extents{3, 4}};

    add_to_petsc_mat(A, global_index_start + 0, global_index_start + 0, arr1);
    add_to_petsc_mat(A, global_index_start + 2, global_index_start + 1, arr2);
    add_to_petsc_mat(A, global_index_start + 3, global_index_start + 3, arr1);
// A =
//
//     0     5     1     0     0     0
//     3     3     4     0     0     0
//     2     9     7     0     0     0
//     0     7     2     0     5     1
//     0     1     4     3     3     4
//     0     0     0     2     7     6

    MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

    MatView(A, PETSC_VIEWER_STDOUT_WORLD);
    
    KSP ksp;
    PetscCallAbort(comm, KSPCreate(comm, &ksp));
    PetscCallAbort(comm, KSPSetOperators(ksp, A, A));

    Vec x, b;
    VecCreate(comm, &x);
    VecSetSizes(x, 6, PETSC_DETERMINE);
    VecSetFromOptions(x);
    PetscCallAbort(comm, VecDuplicate(x, &b));

    // Fill the b vector values 
    {
        VecSpan b_view{b};
        b_view[0] = 1.0;
        b_view[1] = 2.0;
        b_view[2] = 2.0;
        b_view[3] = 1.0;
        b_view[4] = 3.0;
        b_view[5] = 4.0;
    } // end scope to relinquish view
    VecSet(x, 0);

    // default to jacobi preconditioner 
    PC pc;
    PetscCallAbort(comm, KSPGetPC(ksp, &pc));
    PCSetType(pc, PCJACOBI);

    /// the user can still override
    PetscCallAbort(comm, KSPSetFromOptions(ksp));
    PetscCallAbort(comm, KSPSolve(ksp, b, x));

    {
        VecSpan xdata{x};
        ASSERT_NEAR( 0.75, xdata[0], 1e-8);
        ASSERT_NEAR( 0.25, xdata[1], 1e-8);
        ASSERT_NEAR(-0.25, xdata[2], 1e-8);
        ASSERT_NEAR( 0.45, xdata[3], 1e-8);
        ASSERT_NEAR(-0.20, xdata[4], 1e-8);
        ASSERT_NEAR( 0.75, xdata[5], 1e-8);
    }
    
    // cleanup
    KSPDestroy(&ksp);
    MatDestroy(&A);
    VecDestroy(&b);
    VecDestroy(&x);
}
