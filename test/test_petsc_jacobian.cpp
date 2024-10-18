#include "iceicle/build_config.hpp"
#include "iceicle/disc/conservation_law.hpp"
#include "iceicle/disc/burgers.hpp"
#include "iceicle/fe_function/component_span.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/form_dense_jacobian.hpp"
#include "iceicle/petsc_newton.hpp"
#include "iceicle/form_petsc_jacobian.hpp"
#include "mdspan/mdspan.hpp"
#include <gtest/gtest.h>
#include <petscmat.h>
#include <petscsys.h>
#include <ranges>

using namespace iceicle;
using namespace iceicle::util;
using namespace iceicle::solvers;
TEST(test_petsc_jacobian, test_mdg_bl){

    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    static constexpr int ndim = 2;
    static constexpr int pn_order = 1;
    static constexpr int neq = 1;
    using T = build_config::T;
    using IDX = build_config::IDX;
    int nelemx = 3;
    int nelemy = 3;

    // set up mesh and fespace
    AbstractMesh<T, IDX, ndim> mesh{
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
    };

    FESpace<T, IDX, ndim> fespace{&mesh, FESPACE_ENUMS::LAGRANGE, FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, pn_order>{}};

    // set up discretization

    BurgersCoefficients<T, ndim> burgers_coeffs{};
    burgers_coeffs.mu = 0.01;
    burgers_coeffs.a = 1.0;
    BurgersFlux physical_flux{burgers_coeffs};
    BurgersUpwind convective_flux{burgers_coeffs};
    BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
    ConservationLawDDG disc{std::move(physical_flux),
                          std::move(convective_flux),
                          std::move(diffusive_flux)};
    disc.field_names = std::vector<std::string>{"u"};
    disc.dirichlet_callbacks.push_back( 
        [](const T *x, T *out){
            out[0] = 0.0;
    });
    disc.dirichlet_callbacks.push_back( 
        [](const T *x, T *out){
            out[0] = 1.0;
    });
    disc.neumann_callbacks.push_back( 
        [](const T *x, T *out){
            out[0] = 1.0;
    });

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


    nodeset_dof_map<IDX> nodeset = select_all_nodes(fespace);

    // create a dof map selecting all traces 
    auto all_traces = std::views::iota( (std::size_t) 0 , fespace.traces.size());
    geo_dof_map geo_map{all_traces, fespace};
    mesh_parameterizations::hyper_rectangle(
            std::array{nelemx, nelemy}, std::array{0.0, 0.0},
            std::array{1.0, 1.0}, geo_map );


    /// ===========================
    /// = Set up the data vectors =
    /// ===========================
    ic_residual_layout<T, IDX, ndim, 1> mdg_layout{geo_map};
    geo_data_layout<T, IDX, ndim> geo_layout{geo_map};

    PetscInt local_res_size = u_layout.size() + mdg_layout.size();
    PetscInt local_u_size = u_layout.size() + geo_layout.size();

    std::vector<T> res_storage(local_res_size);
    std::vector<T> coord_data(geo_layout.size());

    fespan res{res_storage.data(), u_layout};
    dofspan mdg_res{res_storage.data() + res.size(), mdg_layout};
    component_span coord{coord_data, geo_layout};
    extract_geospan(*(fespace.meshptr), coord);

    /// ===========================
    /// = Set up the Petsc matrix =
    /// ===========================

    Mat jac;
    MatCreate(PETSC_COMM_WORLD, &jac);
    MatSetSizes(jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetFromOptions(jac);

    // get the jacobian and residual from petsc interface
    form_petsc_jacobian_fd(fespace, disc, u, res, jac);
    form_petsc_mdg_jacobian_fd(fespace, disc, u, coord, mdg_res, jac);
    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);

    // get the jacobian and residual through the dense matrix interface
    std::vector<T> u_dense(local_u_size);
    std::vector<T> res_dense(local_res_size);
    std::copy_n(u.data(), u.size(), u_dense.begin());
    component_span coord_dense{u_dense.data() + u_layout.size(), geo_layout};
    extract_geospan(*(fespace.meshptr), coord_dense);

    std::vector<T> jac_dense_storage(local_u_size * local_res_size);
    std::mdspan jac_dense{jac_dense_storage.data(), std::extents{local_res_size, local_u_size}};
    form_dense_jacobian_fd(fespace, disc, geo_map, std::span{u_dense},
            std::span{res_dense}, jac_dense);

    for(int i = 0; i < local_res_size; ++i){
        SCOPED_TRACE("ires = " + std::to_string(i));
        ASSERT_DOUBLE_EQ(res_storage[i], res_dense[i]);
    }

    for(int i = 0; i < local_res_size; ++i){
        SCOPED_TRACE("irow = " + std::to_string(i));
        for(int j = 0; j < local_u_size; ++j){
            SCOPED_TRACE("jcol = " + std::to_string(j));
            T petsc_mat_val;
            MatGetValue(jac, i, j, &petsc_mat_val);
            
            std::cout << fmt::format("{:>16f}", (petsc_mat_val - jac_dense[i, j])) << " ";
            ASSERT_NEAR(petsc_mat_val, (jac_dense[i, j]), 1e-5);
        }
        std::cout << std::endl;
    }

}
