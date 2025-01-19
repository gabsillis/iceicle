#include "iceicle/element/reference_element.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include "iceicle/disc/l2_error.hpp"
#include <cstdio>
#include <iceicle/disc/conservation_law.hpp>
#include <iceicle/form_residual.hpp>
#include <cmath>
#include <gtest/gtest.h>
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/mesh/mesh_partition.hpp>
#include <iceicle/disc/projection.hpp>
#include <iceicle/linear_form_solver.hpp>
#include <iceicle/disc/burgers.hpp>
#include <mpi.h>
#include <numbers>
#include <petscsys.h>
#include <string>
#include <type_traits>
#ifdef ICEICLE_USE_PETSC 
#include <iceicle/form_petsc_jacobian.hpp>
#endif
using namespace NUMTOOL::TENSOR::FIXED_SIZE;
using namespace iceicle;



int main(int argc, char **argv){

    mpi::init(&argc, &argv);
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
//    ::testing::TestEventListeners& listeners =
//    ::testing::UnitTest::GetInstance()->listeners();
//    if (mpi::mpi_world_rank() != 0) {
//        delete listeners.Release(listeners.default_result_printer());
//    }
    int result = RUN_ALL_TESTS();
    MPI_Allreduce(MPI_IN_PLACE, &result, 1, MPI_INT, MPI_MAX, mpi::comm_world);
    mpi::finalize();
    return result;
}

TEST(test_projection, test_l2) {
    mpi::mpi_sync();

    // === get mpi information ===
    int nrank, myrank;
    MPI_Comm_rank(mpi::comm_world, &myrank);
    MPI_Comm_size(mpi::comm_world, &nrank);

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
    MPI_Comm_split(mpi::comm_world, color, myrank, &serial_comm);
    double serial_l2_error; // will bcast to this from rank 0
    if(myrank == 0){

        FESpace serial_fespace{&mesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
            FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, 
            tmp::compile_int<3>{}, serial_comm};

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
    MPI_Bcast(&serial_l2_error, 1, MPI_DOUBLE, 0, mpi::comm_world);

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

TEST( test_fespan, test_sync ) {
    mpi::mpi_sync();
    // === get mpi information ===
    int nrank, myrank;
    MPI_Comm_rank(mpi::comm_world, &myrank);
    MPI_Comm_size(mpi::comm_world, &nrank);

    // === create a serial mesh ===
    AbstractMesh<double, int, 2> mesh(
        Tensor<double, 2>{0.0, 0.0},
        Tensor<double, 2>{1.0, 1.0},
        Tensor<int, 2>{3, 3},
        1, 
        Tensor<BOUNDARY_CONDITIONS, 4>{BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET,
            BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET},
        Tensor<int, 4>{0, 0, 0, 0}
    );

    static constexpr int neq = 2;
    static constexpr int Pn = 1;
    AbstractMesh pmesh{partition_mesh(mesh)};
    FESpace parallel_fespace{&pmesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
    FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, tmp::compile_int<Pn>{}};
    fe_layout_right u_layout{parallel_fespace, tmp::to_size<neq>{}, std::true_type{}};
    std::vector<double> u_data(u_layout.size());
    fespan u{u_data, u_layout};

    for(int igdof = 0; igdof < u.ndof(); ++igdof){
        int pidx = u.get_pdof(igdof);
        if(u.owning_rank(pidx) == myrank){
            for(int iv = 0; iv < neq; ++iv){
                u[igdof, iv] = 2 * pidx + iv;
            }
        }
    }

    u.sync_mpi();

    
    for(int igdof = 0; igdof < u.ndof(); ++igdof){
        int pidx = u.get_pdof(igdof);
        for(int iv = 0; iv < neq; ++iv){
            ASSERT_DOUBLE_EQ((u[igdof, iv]), (double) (2 * pidx + iv));
        }
    }
}

TEST(test_residual, test_heat_equation) {
    mpi::mpi_sync();
    // === get mpi information ===
    int nrank, myrank;
    MPI_Comm_rank(mpi::comm_world, &myrank);
    MPI_Comm_size(mpi::comm_world, &nrank);

    // === create a serial mesh ===
    AbstractMesh<double, int, 2> mesh(
        Tensor<double, 2>{0.0, 0.0},
        Tensor<double, 2>{1.0, 1.0},
        Tensor<int, 2>{7, 5},
        1, 
        Tensor<BOUNDARY_CONDITIONS, 4>{BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET,
            BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET},
        Tensor<int, 4>{0, 0, 0, 0}
    );

    // === set up the discretization ===
    static constexpr int neq = 1;
    BurgersCoefficients<double, 2> burgers_coeffs{
        .mu = 1e-3,
        .a = Tensor<double, 2>{0.0, 0.0},
        .b = Tensor<double, 2>{0.0, 0.0}
    };

    BurgersCoefficients<double, 2> disable_coeffs{
        .mu = 1e-3,
        .a = Tensor<double, 2>{0.0, 0.0},
        .b = Tensor<double, 2>{0.0, 0.0}
    };
    BurgersFlux physical_flux{burgers_coeffs};
    BurgersUpwind convective_flux{burgers_coeffs};
    BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
    ConservationLawDDG disc{std::move(physical_flux),
                          std::move(convective_flux),
                          std::move(diffusive_flux)};
    disc.field_names = std::vector<std::string>{"u"};
    disc.residual_names = std::vector<std::string>{"residual"};

    auto bc = [](const double* xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = 1 + 0.1 * std::sin(std::numbers::pi * y);
    };
    disc.dirichlet_callbacks.push_back(bc);

    auto ic = [](const double* xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = std::sin(x);
    };


    int color = (myrank == 0) ? 0 : 1;
    MPI_Comm serial_comm;
    MPI_Comm_split(mpi::comm_world, color, myrank, &serial_comm);

    // initialization linear form
    Projection<double, int, 2, neq> projection{ic};

    double res_vector_norm;
    if(myrank == 0){

        FESpace serial_fespace{&mesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
            FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, 
            tmp::compile_int<3>{}, serial_comm};

        // === set up our data storage and data view ===
        fe_layout_right u_layout{serial_fespace, tmp::to_size<neq>{}, std::true_type{}};
        fe_layout_right res_layout{serial_fespace, tmp::to_size<neq>{}, std::false_type{}};
        std::vector<double> u_data(u_layout.size());
        std::vector<double> res_data(res_layout.size());
        fespan u{u_data, u_layout};
        fespan res{res_data, res_layout};

        // === perform the projection ===
        {
            solvers::LinearFormSolver projection_solver{serial_fespace, projection};
            projection_solver.solve(u);
        }

        solvers::form_residual(serial_fespace, disc, u, res, serial_comm);

        res_vector_norm = res.vector_norm(serial_comm);
    }
    MPI_Bcast(&res_vector_norm, 1, MPI_DOUBLE, 0, mpi::comm_world);

    // ========================================
    // = Parallel version of same computation =
    // ========================================

    AbstractMesh pmesh{partition_mesh(mesh, EL_PARTITION_ALGORITHM::NAIVE)};
    FESpace parallel_fespace{&pmesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
    FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, tmp::compile_int<3>{}};

    // === set up our data storage and data view ===
    fe_layout_right u_layout{parallel_fespace, tmp::to_size<neq>{}, std::true_type{}};
    fe_layout_right res_layout{parallel_fespace, tmp::to_size<neq>{}, std::false_type{}};
    std::vector<double> u_data(u_layout.size());
    std::vector<double> res_data(res_layout.size());
    fespan u{u_data, u_layout};
    fespan res{res_data, res_layout};


    // === perform the projection ===
    {
        solvers::LinearFormSolver projection_solver{parallel_fespace, projection};
        projection_solver.solve(u);
    }

    solvers::form_residual(parallel_fespace, disc, u, res, mpi::comm_world);

    mpi::mpi_sync();
    SCOPED_TRACE("MPI rank = " + std::to_string(mpi::mpi_world_rank()));
    ASSERT_NEAR(res_vector_norm, res.vector_norm(), 1e-10);
}

#ifdef ICEICLE_USE_PETSC 
using T = build_config::T;
using IDX = build_config::IDX;
class domain_test_disc {
    public:
    static constexpr int ndim = 2;
    static constexpr int nv_comp = 2;
    static const int dnv_comp = 2;

    auto domain_integral(
        const FiniteElement<T, IDX, ndim> &el,
        elspan auto unkel,
        elspan auto res
    ) const -> void {
        static constexpr int neq = decltype(unkel)::static_extent();

// use the centroid distance from origin as a semi-unique identifier of elements
        auto centroid = el.centroid();
        T dist = std::sqrt(std::pow(centroid[0], 2) + std::pow(centroid[1], 2));
        // want
        // d res[i, j] / d u[k, l]= dist * (i * neq + j) * (k * neq + l);
        for(int i = 0; i < el.nbasis(); ++i){
            for(int j = 0; j < neq; ++j){
                for(int k = 0; k < el.nbasis(); ++k){
                    for(int l = 0; l < neq; ++l){
                        res[i, j] += dist * unkel[k, l] * (i * neq + j) * (k * neq + l);
                    }
                }
            }
        }
    }
    
    template<class IDX>
    auto domain_integral_jacobian(
        const FiniteElement<T, IDX, ndim>& el,
        elspan auto unkel,
        linalg::out_matrix auto dfdu
    ) {
        static constexpr int neq = decltype(unkel)::static_extent();
        auto centroid = el.centroid();
        T dist = std::sqrt(std::pow(centroid[0], 2) + std::pow(centroid[1], 2));
        auto el_layout = unkel.get_layout();
        for(int i = 0; i < el.nbasis(); ++i){
            for(int j = 0; j < neq; ++j){
                for(int k = 0; k < el.nbasis(); ++k){
                    for(int l = 0; l < neq; ++l){
                        int ijac = el_layout[k, l];
                        int jjac = el_layout[i, j];
                        dfdu[ijac, jjac] = dist * (i * neq + j) * (k * neq + l);
                    }
                }
            }
        }
    }

    template<class IDX, class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
    void trace_integral(
        const TraceSpace<T, IDX, ndim> &trace,
        NodeArray<T, ndim> &coord,
        dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelL,
        dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelR,
        dofspan<T, ResLayoutPolicy> resL,
        dofspan<T, ResLayoutPolicy> resR
    ) const requires ( 
        elspan<decltype(unkelL)> && 
        elspan<decltype(unkelR)> && 
        elspan<decltype(resL)> && 
        elspan<decltype(resL)>
    ) {}

    template<class IDX, class ULayoutPolicy, class UAccessorPolicy, class ResLayoutPolicy>
    void boundaryIntegral(
        const TraceSpace<T, IDX, ndim> &trace,
        NodeArray<T, ndim> &coord,
        dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelL,
        dofspan<T, ULayoutPolicy, UAccessorPolicy> unkelR,
        dofspan<T, ResLayoutPolicy> resL
    ) const requires(
        elspan<decltype(unkelL)> &&
        elspan<decltype(unkelR)> &&
        elspan<decltype(resL)> 
    ) {}

    template<class IDX>
    void interface_conservation(
        const TraceSpace<T, IDX, ndim>& trace,
        NodeArray<T, ndim>& coord,
        elspan auto unkelL,
        elspan auto unkelR,
        facspan auto res
    ) const {}
};

TEST(test_petsc_jacobian, test_domain_integral){

    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    static constexpr int ndim = 2;
    static constexpr int pn_order = 1;
    int nelemx = 11;
    int nelemy = 7;

    // set up mesh and fespace
    AbstractMesh mesh{partition_mesh(AbstractMesh<T, IDX, ndim>{
        Tensor<T, ndim>{{0.0, 0.0}},
        Tensor<T, ndim>{{1.0, 1.0}},
        Tensor<IDX, ndim>{{nelemx, nelemy}},
        1,
        Tensor<BOUNDARY_CONDITIONS, 4>{
            BOUNDARY_CONDITIONS::DIRICHLET,
            BOUNDARY_CONDITIONS::NEUMANN,
            BOUNDARY_CONDITIONS::DIRICHLET,
            BOUNDARY_CONDITIONS::NEUMANN,
        },
        Tensor<int, 4>{0, 0, 1, 0}
    })};

    FESpace<T, IDX, ndim> fespace{&mesh, FESPACE_ENUMS::LAGRANGE, FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, pn_order>{}};

    domain_test_disc disc{};

    static constexpr int neq = domain_test_disc::nv_comp;
    fe_layout_right u_layout{fespace, std::integral_constant<std::size_t, neq>{},
        std::true_type{}};
    fe_layout_right res_layout = exclude_ghost(u_layout);

    std::vector<T> u_storage(u_layout.size());
   
    fespan u{u_storage.data(), u_layout};
   
    // initialize solution to 0;
    std::iota(u_storage.begin(), u_storage.end(), 0.0);

    /// ===========================
    /// = Set up the data vectors =
    /// ===========================
    PetscInt local_res_size = res_layout.size();
    PetscInt local_u_size = u_layout.size();

    std::vector<T> res_storage(local_res_size);

    fespan res{res_storage.data(), res_layout};

    /// ===========================
    /// = Set up the Petsc matrix =
    /// ===========================

    // NOTE: we use owned sizes when forming petsc matrix
    // only
    Mat jac;
    MatCreate(PETSC_COMM_WORLD, &jac);
    MatSetSizes(jac, res.owned_size(PETSC_COMM_WORLD),
            u.owned_size(PETSC_COMM_WORLD),
            PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetFromOptions(jac);

    PetscInt rowstart, rowend;
    MatGetOwnershipRange(jac, &rowstart, &rowend);
    // get the jacobian and residual from petsc interface
    solvers::form_petsc_jacobian_fd(fespace, disc, u, res, jac);
    PetscCallVoid(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

    for(IDX ielem = 0; ielem < fespace.elements.size(); ++ielem){
        const FiniteElement<T, IDX, ndim>& el = fespace.elements[ielem];
        // use the centroid distance from origin as a semi-unique identifier of elements
        auto centroid = el.centroid();
        T dist = std::sqrt(std::pow(centroid[0], 2) + std::pow(centroid[1], 2));
        for(int i = 0; i < el.nbasis(); ++i ){
            for(int j = 0; j < neq; ++j){
                for(int k = 0; k < el.nbasis(); ++k){
                    for(int l = 0; l < neq; ++l){
                        T jac_val_expected = dist * (i * neq + j) * (k * neq + l);
                        IDX ijac = res.get_pindex(ielem, i, j);
                        IDX jjac = u.get_pindex(ielem, k, l);
                        // subtract out expected jacobian contribution
                        MatSetValue(jac, ijac, jjac, -jac_val_expected, ADD_VALUES);
                    }
                }
            }
        }

    }

    PetscCallVoid(MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY));
    PetscCallVoid(MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY));

    for(IDX ijac = rowstart; ijac < rowend; ++ijac) {
        for(IDX jjac = 0; jjac < u.size_parallel(); ++jjac) {
            PetscScalar matval;
            PetscCallVoid(MatGetValue(jac, ijac, jjac, &matval));
            SCOPED_TRACE("irow = " + std::to_string(ijac));
            SCOPED_TRACE("jcol = " + std::to_string(jjac));
            ASSERT_NEAR(0.0, matval, 1e-8);
        }
    }
}
#endif
