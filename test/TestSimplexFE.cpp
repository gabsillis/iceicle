#include <iceicle/element_linear_solve.hpp>
#include <iceicle/disc/projection.hpp> 
#include <iceicle/element/finite_element.hpp> 
#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <iceicle/geometry/simplex_element.hpp> 
#include <iceicle/basis/lagrange.hpp>
#include <iceicle/quadrature/SimplexQuadrature.hpp>
#include <Numtool/vector.hpp>
#include "gtest/gtest.h" 
#include <iomanip>
#include <cmath>
#include <fenv.h>

using namespace iceicle;

void testfunc(const double x[4], double val[1]){
    //val[0] = std::sin(x[0] + x[1] + x[2] + x[3]);
    val[0] = SQUARED(x[0] * x[2]) + SQUARED(x[1] * x[3]) + x[1] + x[3];
}

// test of interoperability of fe components
//
// simplex integration can be represented by
// integrate 0 to 1
// integrate 0 to 1-x
// integrate 0 to 1-x-y
// integrate 0 to 1-(x+y+z)^2
// f(x)
// dw
// dz
// dy
// dx
TEST(test_simplex_4d_linear, project_nl_func){

    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    using namespace transformations;

//    SimplexGeoElement<double, int, 4, 2> simplex1{};
//
//    using Point = MATH::GEOMETRY::Point<double, 4>;
//    std::vector< Point > node_coords{};
//    
//    for(int i = 0; i < simplex1.transformation.nnodes(); ++i){
//        Point ptcopy = simplex1.transformation.reference_nodes()[i];
//        node_coords.push_back(ptcopy);
//        simplex1.setNode(i, i);
//    }
//
//    double bent_value = 0.5*(std::sqrt(5.0) - 1);
//
//    node_coords[8]  = Point{bent_value, 0, 0, bent_value};
//    node_coords[9]  = Point{0, bent_value, 0, bent_value};
//    node_coords[10] = Point{0, 0, bent_value, bent_value};

    SimplexGeoElement<double, int, 4, 1> simplex1{};

    using Point = MATH::GEOMETRY::Point<double, 4>;
    NodeArray<double, 4> node_coords{};
    node_coords.resize(simplex1.transformation.nnodes());
    
    for(int i = 0; i < simplex1.transformation.nnodes(); ++i){
        Point ptcopy = simplex1.transformation.reference_nodes()[i];
        node_coords[i] = ptcopy;
        simplex1.setNode(i, i);
    }

    Projection<double, int, 4, 1> proj(testfunc);

    Point testpt_ref = {0.2, 0.2, 0.2, 0.2}; // centroid (5 barycentric coords)
    Point testpt_act{};
    simplex1.transform(node_coords, testpt_ref, testpt_act);
    
    // test projection for many orders of basis function
    auto testproject = [&]<int Pn>() {
        static constexpr int neq = 1;
        SimplexLagrangeBasis<double, int, 4, Pn> basis{};
        GrundmannMollerSimplexQuadrature<double, int, 4, Pn+1> quadrule{};
        FEEvaluation<double, int, 4> evals(basis, quadrule);
        FiniteElement<double, int, 4> fe(&simplex1, basis, quadrule, evals, 0);

        MATH::Vector<double, int> udata(basis.nbasis());
        MATH::Vector<double, int> resdata(basis.nbasis());
        
        ElementData<double, neq> u(basis.nbasis(), udata.data());
        ElementData<double, neq> res(basis.nbasis(), resdata.data());
        res = 0; // make sure to zero out residual before projection

        // get the domain integral of the projection
        proj.domainIntegral(fe, node_coords, res);
       
        // solve for u
        solvers::ElementLinearSolver<double, int, 4, neq> solver(fe, node_coords);
        solver.solve(u, res);

        // get error at centroid
        MATH::Vector<double, int> Bi(basis.nbasis());
        fe.evalBasis(testpt_ref, Bi.data());
        double f_approx = udata.dot(Bi);
        double f_act;
        testfunc(testpt_act, &f_act);
        double err_centroid = f_act - f_approx;

        // get L2 error
        double sqerrL2 = 0;
        GrundmannMollerSimplexQuadrature<double, int, 4, Pn+2> quadrule2{};
        for(int igauss = 0; igauss < quadrule2.npoints(); ++igauss){
            auto qp = quadrule2.getPoint(igauss);
            testpt_ref = qp.abscisse;

            fe.evalBasis(testpt_ref, Bi.data());
            double f_approx = udata.dot(Bi);

            double f_act;
            simplex1.transform(node_coords, testpt_ref, testpt_act);
            testfunc(testpt_act, &f_act);

            double J[4][4];
            simplex1.Jacobian(node_coords, qp.abscisse, J);
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
    testproject.template operator()<1>();
    testproject.template operator()<2>();
    testproject.template operator()<3>();
    testproject.template operator()<4>();
    testproject.template operator()<5>();
    testproject.template operator()<6>();
    testproject.template operator()<7>();
    testproject.template operator()<8>();
    testproject.template operator()<9>();

    fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}


TEST(test_simplex_4d_quadratic, project_nl_func){

    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

    SimplexGeoElement<double, int, 4, 2> simplex1{};

    using Point = MATH::GEOMETRY::Point<double, 4>;
    NodeArray<double, 4> node_coords{simplex1.transformation.nnodes()};
    
    for(int i = 0; i < simplex1.transformation.nnodes(); ++i){
        node_coords[i] = simplex1.transformation.reference_nodes()[i];
        simplex1.setNode(i, i);
    }

    double bent_value = 0.5*(std::sqrt(5.0) - 1);

    node_coords[8]  = Point{bent_value, 0, 0, bent_value};
    node_coords[9]  = Point{0, bent_value, 0, bent_value};
    node_coords[10] = Point{0, 0, bent_value, bent_value};

    Projection<double, int, 4, 1> proj(testfunc);

    Point testpt_ref = {0.2, 0.2, 0.2, 0.2}; // centroid (5 barycentric coords)
    Point testpt_act{};
    simplex1.transform(node_coords, testpt_ref, testpt_act);
    
    // test projection for many orders of basis function
    auto testproject = [&]<int Pn>() {
        static constexpr int neq = 1;
        SimplexLagrangeBasis<double, int, 4, Pn> basis{};
        GrundmannMollerSimplexQuadrature<double, int, 4, 2*Pn> quadrule{};
        FEEvaluation<double, int, 4> evals(basis, quadrule);
        FiniteElement<double, int, 4> fe(&simplex1, basis, quadrule, evals, 0);

        MATH::Vector<double, int> udata(basis.nbasis());
        MATH::Vector<double, int> resdata(basis.nbasis());
        
        ElementData<double, neq> u(basis.nbasis(), udata.data());
        ElementData<double, neq> res(basis.nbasis(), resdata.data());
        res = 0; // make sure to zero out residual before projection

        // get the domain integral of the projection
        proj.domainIntegral(fe, node_coords, res);
       
        // solve for u
        solvers::ElementLinearSolver<double, int, 4, neq> solver(fe, node_coords);
        solver.solve(u, res);

        // get error at centroid
        MATH::Vector<double, int> Bi(basis.nbasis());
        fe.evalBasis(testpt_ref, Bi.data());
        double f_approx = udata.dot(Bi);
        double f_act;
        testfunc(testpt_act, &f_act);
        double err_centroid = f_act - f_approx;

        // get L2 error
        double sqerrL2 = 0;
        GrundmannMollerSimplexQuadrature<double, int, 4, Pn+2> quadrule2{};
        for(int igauss = 0; igauss < quadrule2.npoints(); ++igauss){
            auto qp = quadrule2.getPoint(igauss);
            testpt_ref = qp.abscisse;

            fe.evalBasis(testpt_ref, Bi.data());
            double f_approx = udata.dot(Bi);

            double f_act;
            simplex1.transform(node_coords, testpt_ref, testpt_act);
            testfunc(testpt_act, &f_act);

            double J[4][4];
            simplex1.Jacobian(node_coords, qp.abscisse, J);
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
    testproject.template operator()<1>();
    testproject.template operator()<2>();
    testproject.template operator()<3>();
    testproject.template operator()<4>();
    testproject.template operator()<5>();
    testproject.template operator()<6>();
    testproject.template operator()<7>();
    testproject.template operator()<8>();
    testproject.template operator()<9>();

    fedisableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
}
