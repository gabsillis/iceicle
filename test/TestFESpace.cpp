#include "iceicle/disc/projection.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/el_layout.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/solvers/element_linear_solve.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/fe_utils.hpp>

#include <gtest/gtest.h>
#include <pstl/glue_execution_defs.h>

TEST(test_fespace, test_element_construction){
    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 2;
    static constexpr int pn_basis = 3;

    // create a uniform mesh
    MESH::AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {2, 2}, pn_basis);

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, FE::FESPACE_ENUMS::LAGRANGE,
        FE::FESPACE_ENUMS::GAUSS_LEGENDRE, 
        ICEICLE::TMP::compile_int<pn_basis>()
    };

    ASSERT_EQ(fespace.elements.size(), 4);

    ASSERT_EQ(fespace.dg_offsets.calculate_size_requirement(2), 4 * 2 * std::pow(pn_basis + 1, pn_geo));
}

TEST(test_fespace, test_dg_projection){

    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 1;
    static constexpr int pn_basis = 4;
    static constexpr int neq = 1;

    // create a uniform mesh
    MESH::AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {50, 10}, pn_geo);

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, FE::FESPACE_ENUMS::LAGRANGE,
        FE::FESPACE_ENUMS::GAUSS_LEGENDRE, 
        ICEICLE::TMP::compile_int<pn_basis>()
    };


    // define a pn_basis order polynomial function to project onto the space
    auto projfunc = [](const double *xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = std::pow(x, pn_basis) + std::pow(y, pn_basis) + 0.5 + std::pow(x, pn_basis / 2);
    };

    // define a quadratic function to project onto the space
//    auto projfunc = [](const double *xarr, double *out){
//        double x = xarr[0];
//        double y = xarr[1];
//        out[0] = x*x + y*y;
//    };


    // create the projection discretization
    DISC::Projection<double, int, ndim, neq> projection{projfunc};

    T *u = new T[fespace.ndof_dg() * neq](); // 0 initialized
    FE::fespan<T, FE::dg_layout<T, 1>> u_span(u, fespace.dg_offsets);

    // solve the projection 
    std::for_each(fespace.elements.begin(), fespace.elements.end(),
        [&](const ELEMENT::FiniteElement<T, IDX, ndim> &el){
            FE::compact_layout<double, 1> el_layout{el};
            T *u_local = new T[el_layout.size()](); // 0 initialized 
            FE::elspan<T, FE::compact_layout<double, 1>> u_local_span(u_local, el_layout);

            T *res_local = new T[el_layout.size()](); // 0 initialized 
            FE::elspan<T, FE::compact_layout<double, 1>> res_local_span(res_local, el_layout);
            
            // projection residual
            projection.domainIntegral(el, fespace.meshptr->nodes, res_local_span);

            // solve 
            SOLVERS::ElementLinearSolver<T, IDX, ndim, neq> solver{el, fespace.meshptr->nodes};
            solver.solve(u_local_span, res_local_span);

            // test a bunch of random locations
            for(int k = 0; k < 50; ++k){
                MATH::GEOMETRY::Point<T, ndim> ref_pt = FE::random_domain_point(el.geo_el);
                MATH::GEOMETRY::Point<T, ndim> phys_pt;
                el.transform(fespace.meshptr->nodes, ref_pt, phys_pt);
                
                // get the actual value of the function at the given point in the physical domain
                T act_val;
                projfunc(phys_pt, &act_val);

                T projected_val = 0;
                T *basis_vals = new double[el.nbasis()];
                el.evalBasis(ref_pt, basis_vals);
                u_local_span.contract_dofs(basis_vals, &projected_val);

                ASSERT_NEAR(projected_val, act_val, 1e-8);
                delete[] basis_vals;
            }

            delete[] u_local;
            delete[] res_local;
        }
    );

    delete[] u;



}
