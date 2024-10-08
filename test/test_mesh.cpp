/// @brief tests for mesh 

#include <gtest/gtest.h>
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/element_linear_solve.hpp"
#include "iceicle/fe_utils.hpp"

using namespace iceicle;

TEST(test_mesh, test_mixed_uniform_faces){

    using namespace MATH::GEOMETRY;
    std::vector<int> nelem{4, 10};
    std::vector<double> xmin{0.0, 0.0};
    std::vector<double> xmax{1.0, 1.0};
    std::vector<double> quad_ratio{0.5, 0.5};
    std::vector<BOUNDARY_CONDITIONS> bcs{
        BOUNDARY_CONDITIONS::DIRICHLET,
        BOUNDARY_CONDITIONS::DIRICHLET,
        BOUNDARY_CONDITIONS::DIRICHLET,
        BOUNDARY_CONDITIONS::DIRICHLET
    };

    std::vector<int> bcflags{0, 0, 0, 0};

    auto mesh_opt =
        mixed_uniform_mesh<double, int>(nelem, xmin, xmax, quad_ratio, bcs, bcflags);
    AbstractMesh<double, int, 2> mesh = mesh_opt.value();

    std::random_device rdev{};
    std::default_random_engine engine{rdev()};
    std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

    // test consistency of face transformations
    for(int iface = mesh.interiorFaceStart; iface < mesh.interiorFaceEnd; ++iface){
        Point<double, 1> s{domain_dist(engine)};

        const Face<double, int, 2> &face = *(mesh.faces[iface]);
        ElementTransformation<double, int, 2>* transL = mesh.el_transformations[face.elemL];
        ElementTransformation<double, int, 2>* transR = mesh.el_transformations[face.elemR];

        Point<double, 2> x_face, xiL, xiR, xL, xR;
        face.transform(s, mesh.coord, x_face);
        face.transform_xiL(s, xiL);
        face.transform_xiR(s, xiR);
        xL = transL->transform(mesh.get_el_coord(face.elemL), xiL);
        xR = transR->transform(mesh.get_el_coord(face.elemR), xiR);

        ASSERT_NEAR(x_face[0], xL[0], 1e-12);
        ASSERT_NEAR(x_face[1], xL[1], 1e-12);
        ASSERT_NEAR(x_face[0], xR[0], 1e-12);
        ASSERT_NEAR(x_face[1], xR[1], 1e-12);
    }

    static constexpr int pn_basis = 1;
    static constexpr int ndim = 2;

    // test a projection
    FESpace fespace{&mesh, FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
        FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, tmp::compile_int<pn_basis>{}};

    // define a pn_basis order polynomial function to project onto the space 
    auto projfunc = [](const double *xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = std::pow(x, pn_basis) + std::pow(y, pn_basis);
    };

    auto dprojfunc = [](const double *xarr) {
        double x = xarr[0];
        double y = xarr[1];
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim> deriv = { 
            pn_basis * std::pow(x, pn_basis - 1),
            pn_basis * std::pow(y, pn_basis - 1)
        };
        return deriv;
    };

    auto hessfunc = [](const double *xarr){
        double x = xarr[0];
        double y = xarr[1];
        int n = pn_basis;
        if (pn_basis < 2){
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim, ndim> hess = {{
                {0, 0},
                {0, 0}
            }};
            return hess;
        } else {
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim, ndim> hess = {{
                {n * (n-1) *std::pow(x, n-2), 0.0},
                {0.0, n * (n-1) *std::pow(y, n-2)}
            }};
            return hess;
        }
    };

    // create the projection discretization
    Projection<double, int, ndim, 1> projection{projfunc};

    for(const FiniteElement<double, int, ndim>& el : fespace.elements){
        compact_layout_right<int, 1> el_layout{el};
        std::vector<double> udata(el_layout.size()), resdata(el_layout.size());

        dofspan u{udata, el_layout};
        dofspan res{udata, el_layout};


        // projection residual
        projection.domain_integral(el, res);

        // solve 
        solvers::ElementLinearSolver<double, int, ndim, 1> solver{el};
        solver.solve(u, res);

        // test random locations
        for(int k = 0; k < 10; ++k){

            MATH::GEOMETRY::Point<double, ndim> ref_pt = random_domain_point(el.trans);
            MATH::GEOMETRY::Point<double, ndim> phys_pt = el.transform(ref_pt);

            double act_val;
            projfunc(phys_pt, &act_val);

            double projected_val = 0;
            std::vector<double> b_i(el.nbasis());
            el.eval_basis(ref_pt, b_i.data());
            u.contract_dofs(b_i.data(), &projected_val);

            ASSERT_NEAR(projected_val, act_val, 1e-8);


            // test the derivatives
            std::vector<double> grad_basis_data(el.nbasis() * ndim);
            auto grad_basis = el.eval_phys_grad_basis(ref_pt, grad_basis_data.data());
            static_assert(grad_basis.rank() == 2);
            static_assert(grad_basis.extent(1) == ndim);

            // get the derivatives for each equation by contraction
            std::vector<double> grad_eq_data(ndim, 0);
            auto grad_eq = u.contract_mdspan(grad_basis, grad_eq_data.data());

            auto dproj = dprojfunc(phys_pt);
            ASSERT_NEAR(dproj[0], (grad_eq[0, 0]), 1e-10);
            ASSERT_NEAR(dproj[1], (grad_eq[0, 1]), 1e-10);

            // test hessian
            std::vector<double> hess_basis_data(el.nbasis() * ndim * ndim);
            auto hess_basis = el.eval_phys_hess_basis(ref_pt, hess_basis_data.data());

            // get the hessian for each equation by contraction 
            std::vector<double> hess_eq_data(ndim * ndim, 0);
            auto hess_eq = u.contract_mdspan(hess_basis, hess_eq_data.data());
            auto hess_proj = hessfunc(phys_pt);
            ASSERT_NEAR(hess_proj[0][0], (hess_eq[0, 0, 0]), 1e-8);
            ASSERT_NEAR(hess_proj[0][1], (hess_eq[0, 0, 1]), 1e-8);
            ASSERT_NEAR(hess_proj[1][0], (hess_eq[0, 1, 0]), 1e-8);
            ASSERT_NEAR(hess_proj[1][1], (hess_eq[0, 1, 1]), 1e-8);
        }
    }
}
