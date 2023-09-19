#include "gtest/gtest.h" 
#include <Numtool/finite_difference.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>
#include <Numtool/autodiff.hpp>
#include <fenv.h>

using namespace ELEMENT::TRANSFORMATIONS;
using Point3D = MATH::GEOMETRY::Point<double, 3>;

template<typename T>
MATH::GEOMETRY::Point<T, 3> point_transform(MATH::GEOMETRY::Point<T, 3> original){
    double dt = 0.01;
    MATH::GEOMETRY::Point<T, 3> pt_moved = original;
    for(int k = 0; k < 100; ++k){
        double x = pt_moved[0];
        double y = pt_moved[1];
        pt_moved[0] +=  x * dt;
        pt_moved[1] += -y * dt;
    }
    return pt_moved;
}

namespace ELEMENT::TRANSFORMATIONS {
    void _test_simplex_transform(){
        
        SimplexElementTransformation<double, int, 3, 4> trans{};

        for(int i = 0; i <= 10; ++i){
            double xi = 0.1 * i;
            for(int m = 0; m < 4; ++m){
                using namespace MATH;
                auto shpfcn = [&]<class T>(T z){
                    SimplexElementTransformation<T, int, 3, 4> trans2{};
                    return trans2.shapefcn_1d(m, z); 
                };
                auto dual_xi = make_dual(xi);
                auto dualf = shpfcn(dual_xi);
                const std::function<double(double)> &funcref = shpfcn;
                EXPECT_NEAR(trans.dshapefcn_1d(m, xi), dualf.gradient, 10 * std::numeric_limits<double>::epsilon());
            }
        }
    }
}

TEST(test_simplex_transform, test_1d_shp_der){
    _test_simplex_transform();
}

TEST(test_simplex_transform, test_jacobian){
    feenableexcept(FE_ALL_EXCEPT & ~ FE_INEXACT);

    // 3D 4rd order simplex
    // where the reference nodes are transformed by a taylor-green vortex

    SimplexElementTransformation<double, int, 3, 5> trans{};

    const Point3D *reference_nodes = trans.reference_nodes();
    
    // make a curved triangley boi with vortex
    std::vector<Point3D> points{};
    std::vector<int> node_numbers{};
    for(int inode = 0; inode < trans.nnodes(); ++inode){
        Point3D pt = trans.reference_nodes()[inode];
        pt = point_transform(pt);
        points.push_back(pt);
        node_numbers.push_back(inode);
    }

    // use the reference domain of a higher order simplex as a dense set of test points
    SimplexElementTransformation<double, int, 3, 10> dense_trans;

        

    for(int inode = 0; inode < dense_trans.nnodes(); ++inode){
        const Point3D test_pt = dense_trans.reference_nodes()[inode];

        feenableexcept(FE_ALL_EXCEPT & ~ FE_INEXACT);
        double J[3][3];
        trans.Jacobian(points, node_numbers.data(), test_pt, J);
        fedisableexcept(FE_ALL_EXCEPT & ~ FE_INEXACT);

        for(int idim = 0; idim < 3; ++idim){
            for(int jdim = 0; jdim < 3; ++jdim){
                
                // the 1d function to take the derivaive of
                auto func1d = [&test_pt, idim, jdim](double xi_j) -> double {
                    Point3D argpt = test_pt;
                    argpt[jdim] = xi_j;
                    Point3D xpt = point_transform(argpt);
                    return xpt[idim];
                };

                const std::function<double(double)> &funcref = func1d;
                double deriv = MATH::FD::forward_diff_order1(funcref, test_pt[jdim], 1e-9);
                EXPECT_NEAR(J[idim][jdim], deriv, 1e-5);

            }
        }
    }
}
