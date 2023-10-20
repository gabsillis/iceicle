/**
 * @brief Testing for the simplex implementation of lagrange basis functions
 */

#include <iceicle/basis/lagrange.hpp>
#include "gtest/gtest.h"
#include <iceicle/transformations/SimplexElementTransformation.hpp>

using namespace BASIS;
TEST(test_simplex_lagrange, test_nbasis){
    SimplexLagrangeBasis<double, int, 4, 3> basis1{};
    ASSERT_EQ(basis1.nbasis(), 35);

    SimplexLagrangeBasis<double, int, 2, 2> basis2{};
    ASSERT_EQ(basis2.nbasis(), 6);
}

TEST(test_simplex_lagrange, test_eval){
    SimplexLagrangeBasis<double, int, 2, 2> basis{};
    // Test Lagrange property    
    double out[6];

    double xi0[2] = {1, 0};
    basis.evalBasis(xi0, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 0){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    double xi1[2] = {0, 1};
    basis.evalBasis(xi1, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 1){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    double xi2[2] = {0, 0};
    basis.evalBasis(xi2, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 2){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    double xi3[2] = {0.5, 0.5};
    basis.evalBasis(xi3, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 3){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    double xi4[2] = {0, 0.5};
    basis.evalBasis(xi4, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 4){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    double xi5[2] = {0.5, 0};
    basis.evalBasis(xi5, out);
    for(int ibasis = 0; ibasis < 6; ++ibasis){
        if(ibasis == 5){
            ASSERT_DOUBLE_EQ(out[ibasis], 1.0);
        } else {
            ASSERT_DOUBLE_EQ(out[ibasis], 0.0);
        }
    }

    auto B0 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        return x * (2*x - 1);
    };
    auto B1 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        return y * (2*y - 1);
    };
    auto B2 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        double lambda3 = 1.0 - x - y;
        return lambda3 * (2 * lambda3 - 1);
    };
    auto B3 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        return 4*x*y;
    };
    auto B4 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        double lambda3 = 1 - x - y;
        return 4*lambda3*y;
    };
    auto B5 = [](const double *xi){
        double x = xi[0];
        double y = xi[1];
        double lambda3 = 1 - x - y;
        return 4*lambda3*x;
    };

    ELEMENT::TRANSFORMATIONS::SimplexElementTransformation<double, int, 2, 10> nodegen{};

    for(int inode = 0; inode < nodegen.nnodes(); ++inode){
        const double *pt = nodegen.reference_nodes()[inode];
        basis.evalBasis(pt, out);
        ASSERT_NEAR(B0(pt), out[0], 1e-15);
        ASSERT_NEAR(B1(pt), out[1], 1e-15);
        ASSERT_NEAR(B2(pt), out[2], 1e-15);
        ASSERT_NEAR(B3(pt), out[3], 1e-15);
        ASSERT_NEAR(B4(pt), out[4], 1e-15);
        ASSERT_NEAR(B5(pt), out[5], 1e-15);
    }
}
