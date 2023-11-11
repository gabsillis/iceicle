#include "Numtool/MathUtils.hpp"
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/basis/lagrange.hpp"
#include <iceicle/solvers/element_linear_solve.hpp>
#include <iceicle/disc/projection.hpp>
#include <iceicle/element/finite_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <iceicle/quadrature/HypercubeGaussLegendre.hpp>
#include <Numtool/vector.hpp>
#include "gtest/gtest.h"
#include <fenv.h>
#include <random>

void testfunc_hypercube(const double x[4], double val[1]){
    val[0] = SQUARED(x[0] * x[2]) + SQUARED(x[1] * x[3]) + x[1] + x[3];
}

void testfunc_2d_square(const double x[2], double val[1]){
    val[0] = SQUARED(x[0] * x[1]) + x[0] + x[1];
}

TEST(test_quadratic_quad, project_polynomial){

    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    using namespace ELEMENT;
    using namespace ELEMENT::TRANSFORMATIONS;
    using namespace BASIS;

    std::random_device rdev{};
    std::default_random_engine engine{rdev()};

    std::uniform_real_distribution<double> peturb_dist{-0.2, 0.2};

    static constexpr int ndim = 2;
    static constexpr int geo_order = 2;
    HypercubeElement<double, int, ndim, geo_order> geo_el{};

    using Point = MATH::GEOMETRY::Point<double, ndim>;
    FE::NodalFEFunction<double, ndim> node_coords{};
    node_coords.resize(geo_el.n_nodes());

    // randomly peturb reference domain nodes 
    for(int i = 0; i < geo_el.n_nodes(); ++i){
        Point ptcopy = geo_el.transformation.reference_nodes()[i];
        for(int idim = 0; idim < ndim; ++idim){
            ptcopy[idim] += peturb_dist(engine);
        }

        // copy the peturbed node into the coordinates array
        node_coords[i] = ptcopy;

        // set the node index for the element
        geo_el.setNode(i, i);
    }

    Point center = {0.0, 0.0};
    Point center_phys{};

    DISC::Projection<double, int, ndim, 1> proj(testfunc_2d_square);

    auto testproject = [&]<int Pn>(){
        static constexpr int neq = 1;
        HypercubeLagrangeBasis<double, int, ndim, Pn> basis{};
        QUADRATURE::HypercubeGaussLegendre<double, int, ndim, Pn+1> quadrule{};
        FEEvaluation<double, int, ndim> evals(basis, quadrule);
        FiniteElement<double, int, ndim> fe{&geo_el, basis, quadrule, evals, 0};


        MATH::Vector<double, int> udata(basis.nbasis());
        MATH::Vector<double, int> resdata(basis.nbasis());
        
        FE::ElementData<double, neq> u(basis.nbasis(), udata.data());
        FE::ElementData<double, neq> res(basis.nbasis(), resdata.data());
        res = 0; // make sure to zero out residual before projection

        // get the domain integral of the projection
        proj.domainIntegral(fe, node_coords, res);
       
        // solve for u
        SOLVERS::ElementLinearSolver<double, int, ndim, neq> solver(fe, node_coords);
        solver.solve(u, res);

        // get error at centroid
        MATH::Vector<double, int> Bi(basis.nbasis());
        fe.evalBasis(center, Bi.data());
        double f_approx = udata.dot(Bi);
        double f_act;
        geo_el.transform(node_coords, center, center_phys);
        testfunc_2d_square(center_phys, &f_act);
        double err_centroid = f_act - f_approx;

        // get L2 error
        double sqerrL2 = 0;
        QUADRATURE::HypercubeGaussLegendre<double, int, ndim, 4> quadrule2{};
        for(int igauss = 0; igauss < quadrule2.npoints(); ++igauss){
            auto qp = quadrule2.getPoint(igauss);
            Point testpt_ref = qp.abscisse;
            Point testpt_act;

            fe.evalBasis(testpt_ref, Bi.data());
            double f_approx = udata.dot(Bi);

            double f_act;
            geo_el.transform(node_coords, testpt_ref, testpt_act);
            testfunc_2d_square(testpt_act, &f_act);

            double J[ndim][ndim];
            geo_el.Jacobian(node_coords, qp.abscisse, J);
            double detJ = MATH::MATRIX_T::determinant<ndim, double>(*J);

            double esq = SQUARED(f_approx - f_act);
            sqerrL2 += esq * detJ * qp.weight;
        }
        double errL2 = std::sqrt(sqerrL2);

        const int p_fwidth = 12;
        std::cout << "Pn: " << std::setw(3) << Pn 
            << " | f_approx: " << std::setw(p_fwidth) << f_approx
            << " | f_act: " << std::setw(p_fwidth) << f_act
            << " | error centroid: " << std::setw(p_fwidth) << err_centroid
            << " | error L2: " << std::setw(p_fwidth) << errL2
            << " | log err L2: " << std::setw(p_fwidth) << std::log(errL2) << std::endl;
    };

    NUMTOOL::TMP::constexpr_for_range<1, 7>(testproject);

    fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}

TEST(test_hypercube_linear, project_polynomial){
    
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    using namespace ELEMENT;
    using namespace ELEMENT::TRANSFORMATIONS;
    using namespace BASIS;

    std::random_device rdev{};
    std::default_random_engine engine{rdev()};

    std::uniform_real_distribution<double> peturb_dist{-0.2, 0.2};

    static constexpr int ndim = 4;
    HypercubeElement<double, int, ndim, 1> geo_el{};

    using Point = MATH::GEOMETRY::Point<double, ndim>;
    FE::NodalFEFunction<double, ndim> node_coords{};
    node_coords.resize(geo_el.n_nodes());

    // randomly peturb reference domain nodes 
    for(int i = 0; i < geo_el.n_nodes(); ++i){
        Point ptcopy = geo_el.transformation.reference_nodes()[i];
        for(int idim = 0; idim < ndim; ++idim){
            ptcopy[idim] += peturb_dist(engine);
        }

        // copy the peturbed node into the coordinates array
        node_coords[i] = ptcopy;

        // set the node index for the element
        geo_el.setNode(i, i);
    }

    Point center = {0.0, 0.0, 0.0, 0.0};
    Point center_phys{};

    DISC::Projection<double, int, 4, 1> proj(testfunc_hypercube);

    auto testproject = [&]<int Pn>(){
        static constexpr int neq = 1;
        HypercubeLagrangeBasis<double, int, ndim, Pn> basis{};
        QUADRATURE::HypercubeGaussLegendre<double, int, ndim, Pn+1> quadrule{};
        FEEvaluation<double, int, ndim> evals(basis, quadrule);
        FiniteElement<double, int, ndim> fe{&geo_el, basis, quadrule, evals, 0};


        MATH::Vector<double, int> udata(basis.nbasis());
        MATH::Vector<double, int> resdata(basis.nbasis());
        
        FE::ElementData<double, neq> u(basis.nbasis(), udata.data());
        FE::ElementData<double, neq> res(basis.nbasis(), resdata.data());
        res = 0; // make sure to zero out residual before projection

        // get the domain integral of the projection
        proj.domainIntegral(fe, node_coords, res);
       
        // solve for u
        SOLVERS::ElementLinearSolver<double, int, 4, neq> solver(fe, node_coords);
        solver.solve(u, res);

        // get error at centroid
        MATH::Vector<double, int> Bi(basis.nbasis());
        fe.evalBasis(center, Bi.data());
        double f_approx = udata.dot(Bi);
        double f_act;
        geo_el.transform(node_coords, center, center_phys);
        testfunc_hypercube(center_phys, &f_act);
        double err_centroid = f_act - f_approx;

        // get L2 error
        double sqerrL2 = 0;
        QUADRATURE::HypercubeGaussLegendre<double, int, 4, Pn+2> quadrule2{};
        for(int igauss = 0; igauss < quadrule2.npoints(); ++igauss){
            auto qp = quadrule2.getPoint(igauss);
            Point testpt_ref = qp.abscisse;
            Point testpt_act;

            fe.evalBasis(testpt_ref, Bi.data());
            double f_approx = udata.dot(Bi);

            double f_act;
            geo_el.transform(node_coords, testpt_ref, testpt_act);
            testfunc_hypercube(testpt_act, &f_act);

            double J[4][4];
            geo_el.Jacobian(node_coords, qp.abscisse, J);
            double detJ = MATH::MATRIX_T::determinant<4, double>(*J);

            double esq = SQUARED(f_approx - f_act);
            sqerrL2 += esq * detJ * qp.weight;
        }
        double errL2 = std::sqrt(sqerrL2);

        const int p_fwidth = 12;
        std::cout << "Pn: " << std::setw(3) << Pn 
            << " | f_approx: " << std::setw(p_fwidth) << f_approx
            << " | f_act: " << std::setw(p_fwidth) << f_act
            << " | error centroid: " << std::setw(p_fwidth) << err_centroid
            << " | error L2: " << std::setw(p_fwidth) << errL2
            << " | log err L2: " << std::setw(p_fwidth) << std::log(errL2) << std::endl;
    };

    NUMTOOL::TMP::constexpr_for_range<1, 7>(testproject);

    fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}
