#include "iceicle/build_config.hpp"
#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/petsc_newton.hpp"
#include "iceicle/form_petsc_jacobian.hpp"
#include <gtest/gtest.h>

TEST(test_petsc_jacobian, test_mdg_bl){

    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace ICEICLE::UTIL;
    using namespace ICEICLE::SOLVERS;
    using namespace FE;
    static constexpr int ndim = 2;
    static constexpr int pn_order = 1;
    static constexpr int neq = 1;
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;
    int nelemx = 3;
    int nelemy = 3;

    // set up mesh and fespace
    MESH::AbstractMesh<T, IDX, ndim> mesh{
        Tensor<T, ndim>{{0.0, 0.0}},
        Tensor<T, ndim>{{1.0, 1.0}},
        Tensor<IDX, ndim>{{nelemx, nelemy}},
        1,
        Tensor<ELEMENT::BOUNDARY_CONDITIONS, 4>{
            ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET,
            ELEMENT::BOUNDARY_CONDITIONS::NEUMANN,
            ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET,
            ELEMENT::BOUNDARY_CONDITIONS::NEUMANN,
        },
        Tensor<int, 4>{0, 0, 1, 0}
    };

    FE::FESpace<T, IDX, ndim> fespace{&mesh, FE::FESPACE_ENUMS::LAGRANGE, FE::FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, pn_order>{}};

    // set up discretization
    DISC::HeatEquation<T, IDX, ndim> disc{};
    disc.mu = 0.01;
    disc.a = 1.0;
    disc.dirichlet_values.push_back(0.0);
    disc.dirichlet_values.push_back(1.0);
    disc.neumann_values.push_back(0.0);

    // define layout
    fe_layout_right u_layout{fespace.dg_map, std::integral_constant<std::size_t, neq>{}};

    std::vector<T> u_storage(u_layout.size());
   
    fespan u{u_storage.data(), u_layout};
   
    // initialize solution to 0;
    std::fill(u_storage.begin(), u_storage.end(), 0.0);

    // solve once on a static mesh
    ConvergenceCriteria<T, IDX> conv_criteria{
        .tau_abs = std::numeric_limits<T>::epsilon(),
        .tau_rel = 1e-9,
        .kmax = 2
    };
    PetscNewton solver{fespace, disc, conv_criteria};
    solver.solve(u);


}
