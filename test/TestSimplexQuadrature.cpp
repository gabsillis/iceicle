#include "Numtool/matrixT.hpp"
#include "gtest/gtest.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <iceicle/quadrature/SimplexQuadrature.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

using namespace ELEMENT::TRANSFORMATIONS;
using namespace QUADRATURE;

TEST(test_simplex_quadrature, test_grundman_moller_tet){
   
    GrundmannMollerSimplexQuadrature<double, int, 3, 2> simplexquadrule{};
    ASSERT_EQ(simplexquadrule.npoints(), 15);
    std::stringstream sstream{};
    for(int igauss = 0; igauss < simplexquadrule.npoints(); ++igauss){
        const QuadraturePoint<double, 3> &quadpt = simplexquadrule.getPoint(igauss);
        sstream << std::setw(8) << igauss << " "
                  << std::setw(8) << quadpt.weight << " "
                  << std::setw(8) << quadpt.abscisse[0] << " "
                  << std::setw(8) << quadpt.abscisse[1] << " "
                  << std::setw(8) << quadpt.abscisse[2] << " "
                  << std::endl;
    }
        ASSERT_EQ(
"       0 0.0507937    0.125    0.125    0.125 \n"
"       1 0.0507937    0.375    0.125    0.125 \n"
"       2 0.0507937    0.625    0.125    0.125 \n"
"       3 0.0507937    0.125    0.375    0.125 \n"
"       4 0.0507937    0.375    0.375    0.125 \n"
"       5 0.0507937    0.125    0.625    0.125 \n"
"       6 0.0507937    0.125    0.125    0.375 \n"
"       7 0.0507937    0.375    0.125    0.375 \n"
"       8 0.0507937    0.125    0.375    0.375 \n"
"       9 0.0507937    0.125    0.125    0.625 \n"
"      10 -0.0964286 0.166667 0.166667 0.166667 \n"
"      11 -0.0964286      0.5 0.166667 0.166667 \n"
"      12 -0.0964286 0.166667      0.5 0.166667 \n"
"      13 -0.0964286 0.166667 0.166667      0.5 \n"
"      14 0.0444444     0.25     0.25     0.25 \n", sstream.str()
        );
}

TEST(test_simplex_quadrature, test_grundmann_moller_triangle){
    using Point2D = MATH::GEOMETRY::Point<double, 2>;

    auto testfcn = [](const double *x_vec){
        double x = x_vec[0];
        double y = x_vec[1];
        // x^4 + y^4
        return x * std::pow(y, 2);
    };

    auto testfcn2 = [](const double *x_vec){
        double x = x_vec[0];
        double y = x_vec[1];
        // x^4 + y^4
        return std::pow(x, 5) + std::pow(y, 5);
    };
    // should be integrated exactly by an integration rule of order 2
   
    std::vector<Point2D> nodes = {
        { 2.0, 0.0 },
        { 2.0, 2.0},
        { 0.0, 0.0 }
    };
    // Exact integral of x^4+y^4 on this triangle = 8.6
    std::vector<int> node_idxs = {0, 1, 2};
    
    // Linear Transformation
    SimplexElementTransformation<double, int, 2, 1> trans_lin{};
    GrundmannMollerSimplexQuadrature<double, int, 2, 2> simplexquadrule{};

    QuadratureRule<double, int, 2> &quadrule = simplexquadrule;
    ASSERT_EQ(quadrule.npoints(), 10);

    double J[2][2]; // storage for jacobian matrix
    double integral = 0;
    for(int igauss = 0; igauss < quadrule.npoints(); ++igauss){
        const QuadraturePoint<double, 2> &quadpt = quadrule[igauss];
        /*
        std::cout << std::setw(8) << igauss << " "
                  << std::setw(8) << quadpt.weight << " "
                  << std::setw(8) << quadpt.abscisse[0] << " "
                  << std::setw(8) << quadpt.abscisse[1] << " "
                  << std::endl;
        */
        const Point2D &refpt = quadpt.abscisse;
        Point2D xpt;
        trans_lin.transform(nodes, node_idxs.data(), refpt, xpt);
//        std::cout << "xpt: (" << xpt[0] << ", " << xpt[1] << ")" << std::endl;

        trans_lin.Jacobian(nodes, node_idxs.data(), refpt, J);
        double detJ = MATH::MATRIX_T::determinant<2>(*J);
        integral += testfcn(xpt) * quadpt.weight * detJ;
 //       std::cout << "f: " << testfcn(xpt) << " | total: " << integral << std::endl;
    }

    ASSERT_NEAR(integral, 32.0 / 15.0, 1e-13);

    std::vector<Point2D> nodes2 = {
        { 1.0, 0.0 },
        { 1.0, 3.0},
        { 0.0, 0.0 }
    };
    integral = 0;
    for(int igauss = 0; igauss < quadrule.npoints(); ++igauss){
        
        const Point2D &refpt = quadrule[igauss].abscisse;
        Point2D xpt;
        trans_lin.transform(nodes2, node_idxs.data(), refpt, xpt);
        trans_lin.Jacobian(nodes2, node_idxs.data(), refpt, J);
        double detJ = MATH::MATRIX_T::determinant<2>(*J);

        integral += testfcn2(xpt) * quadrule[igauss].weight * detJ;
    }
    ASSERT_NEAR(integral, 17.7857142867, 1e-6);
}
