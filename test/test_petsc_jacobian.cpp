#include "iceicle/build_config.hpp"
#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/form_dense_jacobian.hpp"
#include "iceicle/petsc_newton.hpp"
#include "iceicle/form_petsc_jacobian.hpp"
#include "mdspan/mdspan.hpp"
#include <gtest/gtest.h>
#include <petscmat.h>
#include <petscsys.h>

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


    FE::nodeset_dof_map<IDX> nodeset = FE::select_all_nodes(fespace);
    Mat jac;
    MatCreate(PETSC_COMM_WORLD, &jac);
    PetscInt local_res_size = fespace.dg_map.calculate_size_requirement(1) + nodeset.selected_nodes.size() * ndim;
    PetscInt local_u_size = fespace.dg_map.calculate_size_requirement(1) + nodeset.selected_nodes.size() * ndim;
    MatSetSizes(jac, local_res_size, local_u_size, PETSC_DETERMINE, PETSC_DETERMINE);
    MatSetFromOptions(jac);

    std::vector<T> res_storage(local_res_size);
    FE::fespan res{res_storage.data(), u_layout};
    FE::node_selection_layout<IDX, ndim> mdg_layout{nodeset};
    FE::dofspan mdg_res{res_storage.data() + res.size(), mdg_layout};

    // get the jacobian and residual from petsc interface
    form_petsc_jacobian_fd(fespace, disc, u, res, jac);
    form_petsc_mdg_jacobian_fd(fespace, disc, u, mdg_res, jac);
    MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);

    // get the jacobian and residual through the dense matrix interface
    std::vector<T> u_dense(local_u_size);
    std::vector<T> res_dense(local_res_size);
    std::copy_n(u.data(), u.size(), u_dense.begin());
    for(int inode = 0; inode < nodeset.selected_nodes.size(); ++inode){
        IDX gnode = nodeset.selected_nodes[inode];
        for(int idim = 0; idim < ndim; ++idim){
            u_dense[u.size() + ndim * inode + idim] = fespace.meshptr->nodes[gnode][idim];
        }
    }

    std::vector<T> jac_dense_storage(local_u_size * local_res_size);
    std::mdspan jac_dense{jac_dense_storage.data(), std::extents{local_res_size, local_u_size}};
    form_dense_jacobian_fd(fespace, disc, nodeset, std::span{u_dense}, std::span{res_dense}, 
            jac_dense, std::integral_constant<int, ndim>{});

    for(int i = 0; i < local_res_size; ++i){
        SCOPED_TRACE("MDG indices start at: " + std::to_string(fespace.dg_map.calculate_size_requirement(1)));
        SCOPED_TRACE("ires = " + std::to_string(i));
        ASSERT_DOUBLE_EQ(res_storage[i], res_dense[i]);
    }

    for(int i = 0; i < local_res_size; ++i){
        SCOPED_TRACE("MDG indices start at: " + std::to_string(fespace.dg_map.calculate_size_requirement(1)));
        SCOPED_TRACE("irow = " + std::to_string(i));
        for(int j = 0; j < local_u_size; ++j){
            SCOPED_TRACE("jcol = " + std::to_string(j));
            T petsc_mat_val;
            MatGetValue(jac, i, j, &petsc_mat_val);
            // ASSERT_NEAR(petsc_mat_val, (jac_dense[i, j]), 1e-3);
            
            std::cout << std::format("{:>16f}", (petsc_mat_val - jac_dense[i, j])) << " ";
        }
        std::cout << std::endl;
    }

}
