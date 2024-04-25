#include "gtest/gtest.h"
#include <fenv.h>
#include "iceicle/build_config.hpp"
#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/disc/projection.hpp"
#include <iceicle/element_linear_solve.hpp>
#include <petscmat.h>
#include "iceicle/form_petsc_jacobian.hpp"

using namespace iceicle;
TEST(test_petsc_solvers, test_form_jacobian){

    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // Get the floating point and index types from 
    // cmake configuration
    using T = build_config::T;
    using IDX = build_config::IDX;

    // 2 dimensional simulation
    static constexpr int ndim = 2;

    // =========================
    // = create a uniform mesh =
    // =========================

    IDX nx=2, ny=2;
    const IDX nelem_arr[ndim] = {nx, ny};
    // bottom left corner
    T xmin[ndim] = {-1.0, -1.0};
    // top right corner
    T xmax[ndim] = {1.0, 1.0};
    // boundary conditions
    Tensor<BOUNDARY_CONDITIONS, 2 * ndim> bctypes = {{
        BOUNDARY_CONDITIONS::DIRICHLET, // left side 
        BOUNDARY_CONDITIONS::NEUMANN,   // bottom side 
        BOUNDARY_CONDITIONS::DIRICHLET, // right side 
        BOUNDARY_CONDITIONS::NEUMANN    // top side
    }};
    int bcflags[2 * ndim] = {
        0, // left side 
        0, // bottom side 
        1, // right side
        0  // top side
    };
    int geometry_order = 1;

    AbstractMesh<T, IDX, ndim> mesh{xmin, xmax, 
        nelem_arr, geometry_order, bctypes, bcflags};

    // ===================================
    // = create the finite element space =
    // ===================================

    static constexpr int basis_order = 1;

    FESpace<T, IDX, ndim> fespace{
        &mesh, 
        FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE, 
        FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, 
        std::integral_constant<int, basis_order>{}
    };

    // =============================
    // = set up the discretization =
    // =============================

    HeatEquation<T, IDX, ndim> heat_equation{}; 
    // Dirichlet BC: Left side = 0
    heat_equation.dirichlet_values.push_back(1.0); 
    // Dirichlet BC: Rigght side = 1
    heat_equation.dirichlet_values.push_back(2.0); 
    // Neumann BC: Top and bottom have 0 normal gradient 
    heat_equation.neumann_values.push_back(0.0);

    // ============================
    // = set up a solution vector =
    // ============================
    constexpr int neq = decltype(heat_equation)::nv_comp;
    std::vector<T> u_data(fespace.dg_offsets.calculate_size_requirement(neq));
    dg_layout<T, neq> u_layout{fespace.dg_offsets};
    fespan u{u_data.data(), u_layout};

    // ===========================
    // = initialize the solution =
    // ===========================
    auto ic = [](const double *xarr, double *out) -> void{
        double x = xarr[0];
        double y = xarr[1];

        //out[0] = x;
        out[0] = std::sin(x) * std::cos(y);
    };
    
    // Manufactured solution BC 
    heat_equation.dirichlet_callbacks.resize(2);
    heat_equation.dirichlet_callbacks[1] = ic;


    Projection<T, IDX, ndim, neq> projection{ic};
    // TODO: extract into LinearFormSolver
    std::vector<T> u_local_data(fespace.dg_offsets.max_el_size_reqirement(neq));
    std::vector<T> res_local_data(fespace.dg_offsets.max_el_size_reqirement(neq));
    std::for_each(fespace.elements.begin(), fespace.elements.end(), 
        [&](const FiniteElement<T, IDX, ndim> &el){
            // form the element local views
            // TODO: maybe instead of scatter from local view 
            // we can directly create the view on the subset of u 
            // for CG this might require a different compact Layout 
            elspan u_local{u_local_data.data(), u.create_element_layout(el.elidx)};
            u_local = 0;

            elspan res_local{res_local_data.data(), u.create_element_layout(el.elidx)};
            res_local = 0;

            // project
            projection.domainIntegral(el, fespace.meshptr->nodes, res_local);

            // solve 
            solvers::ElementLinearSolver<T, IDX, ndim, neq> solver{el, fespace.meshptr->nodes};
            solver.solve(u_local, res_local);

            // scatter to global array 
            // (note we use 0 as multiplier for current values in global array)
            scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);
        }
    );

    // =================
    // = Test Jacobian =
    // =================
    MPI_Comm comm = MPI_COMM_WORLD;
    Mat jac;
    MatCreate(comm, &jac);

    std::vector<T> res_data(fespace.dg_offsets.calculate_size_requirement(neq));
    dg_layout<T, neq> res_layout{fespace.dg_offsets};
    fespan res{res_data.data(), res_layout};

    solvers::form_petsc_jacobian_fd(fespace, heat_equation, u, res, jac);
    

    MatDestroy(&jac);
}
