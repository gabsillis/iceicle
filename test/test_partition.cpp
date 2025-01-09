#include "iceicle/element/reference_element.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include "iceicle/disc/l2_error.hpp"
#include <cmath>
#include <gtest/gtest.h>
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/mesh/mesh_partition.hpp>
#include <iceicle/disc/projection.hpp>
#include <iceicle/linear_form_solver.hpp>
#include <mpi.h>
#include <string>
#include <type_traits>

using namespace NUMTOOL::TENSOR::FIXED_SIZE;
using namespace iceicle;

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
//    {
//        volatile int i = 0;
//        char hostname[256];
//        gethostname(hostname, sizeof(hostname));
//        printf("PID %d on %s ready for attach\n", getpid(), hostname);
//        fflush(stdout);
//        while (0 == i)
//            sleep(5);
//    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    MPI_Finalize();
    return result;
}

TEST(test_projection, test_l2) {

    // === get mpi information ===
    int nrank, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    // === create a serial mesh ===
    AbstractMesh<double, int, 2> mesh(
        Tensor<double, 2>{0.0, 0.0},
        Tensor<double, 2>{1.0, 1.0},
        Tensor<int, 2>{10, 7},
        1);


    // === define our exact solution that we will l2 project onto the space ===
    static constexpr int neq = 2;
    auto projfunc = [](const double *xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = std::sin(x) + std::cos(y);
        out[1] = std::cos(x) + std::cos(y);
    };
    // Linear form of our rhs
    Projection<double, int, 2, neq> projection{projfunc};

    int color = (myrank == 0) ? 0 : 1;
    MPI_Comm serial_comm;
    MPI_Comm_split(MPI_COMM_WORLD, color, myrank, &serial_comm);
    double serial_l2_error; // will bcast to this from rank 0
    if(myrank == 0){

        FESpace serial_fespace{&mesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
            FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, 
            tmp::compile_int<3>{}, true};

        // === set up our data storage and data view ===
        std::vector<double> u_serial_data(serial_fespace.ndof() * neq);
        fe_layout_right u_serial_layout{serial_fespace, tmp::to_size<neq>{}};
        fespan u_serial{u_serial_data, u_serial_layout};

        // === perform the projection ===
        {
            solvers::LinearFormSolver projection_solver{serial_fespace, projection};
            projection_solver.solve(u_serial);
        }

        std::function<void(double*, double*)> exact = [projfunc](double *x, double * out){ projfunc(x, out); };
        serial_l2_error = l2_error(exact, serial_fespace, u_serial, serial_comm);
    }
    MPI_Bcast(&serial_l2_error, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // ========================================
    // = Parallel version of same computation =
    // ========================================

    AbstractMesh pmesh{partition_mesh(mesh)};
    FESpace parallel_fespace{&pmesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
    FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, tmp::compile_int<3>{}};

    // === set up our data storage and data view ===

    // note this includes ghost dofs (unused)
    std::vector<double> u_parallel_data(parallel_fespace.ndof() * neq); 
    // in our view for projection we do not include ghost dofs 
    // (this is the default but we explicitly use std::false_type for clarity)
    fe_layout_right u_parallel_layout{parallel_fespace, tmp::to_size<neq>{}, std::false_type{}};
    fespan u_parallel{u_parallel_data, u_parallel_layout};

    // === perform the projection ===
    {
        solvers::LinearFormSolver projection_solver{parallel_fespace, projection};
        projection_solver.solve(u_parallel);
    }
    std::function<void(double*, double*)> exact = [projfunc](double *x, double * out){ projfunc(x, out); };
    double parallel_l2_error = l2_error(exact, parallel_fespace, u_parallel);

    mpi::execute_on_rank(0, [&]{ std::cout << "l2_error serial : " << serial_l2_error << " | parallel : " << parallel_l2_error << std::endl; });

    SCOPED_TRACE("MPI rank = " + std::to_string(mpi::mpi_world_rank()));
    ASSERT_NEAR(serial_l2_error, parallel_l2_error, 1e-10);
}
