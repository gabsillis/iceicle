#include "Numtool/polydefs/LagrangePoly.hpp"
#include <iceicle/transformations/HypercubeElementTransformation.hpp>
#include "gtest/gtest.h"

using namespace ELEMENT::TRANSFORMATIONS;

TEST(test_hypercube_transform, test_ijk_poin){
  HypercubeElementTransformation<double, int, 1, 4> trans1{};
  ASSERT_EQ(
      "[ 0 ]\n"
      "[ 1 ]\n"
      "[ 2 ]\n"
      "[ 3 ]\n"
      "[ 4 ]\n",
      trans1.print_ijk_poin()
      );

  HypercubeElementTransformation<double, int, 3, 3> trans2{};
  ASSERT_EQ(
      "[ 0 0 0 ]\n"
      "[ 0 0 1 ]\n"
      "[ 0 0 2 ]\n"
      "[ 0 0 3 ]\n"
      "[ 0 1 0 ]\n"
      "[ 0 1 1 ]\n"
      "[ 0 1 2 ]\n"
      "[ 0 1 3 ]\n"
      "[ 0 2 0 ]\n"
      "[ 0 2 1 ]\n"
      "[ 0 2 2 ]\n"
      "[ 0 2 3 ]\n"
      "[ 0 3 0 ]\n"
      "[ 0 3 1 ]\n"
      "[ 0 3 2 ]\n"
      "[ 0 3 3 ]\n"
      "[ 1 0 0 ]\n"
      "[ 1 0 1 ]\n"
      "[ 1 0 2 ]\n"
      "[ 1 0 3 ]\n"
      "[ 1 1 0 ]\n"
      "[ 1 1 1 ]\n"
      "[ 1 1 2 ]\n"
      "[ 1 1 3 ]\n"
      "[ 1 2 0 ]\n"
      "[ 1 2 1 ]\n"
      "[ 1 2 2 ]\n"
      "[ 1 2 3 ]\n"
      "[ 1 3 0 ]\n"
      "[ 1 3 1 ]\n"
      "[ 1 3 2 ]\n"
      "[ 1 3 3 ]\n"
      "[ 2 0 0 ]\n"
      "[ 2 0 1 ]\n"
      "[ 2 0 2 ]\n"
      "[ 2 0 3 ]\n"
      "[ 2 1 0 ]\n"
      "[ 2 1 1 ]\n"
      "[ 2 1 2 ]\n"
      "[ 2 1 3 ]\n"
      "[ 2 2 0 ]\n"
      "[ 2 2 1 ]\n"
      "[ 2 2 2 ]\n"
      "[ 2 2 3 ]\n"
      "[ 2 3 0 ]\n"
      "[ 2 3 1 ]\n"
      "[ 2 3 2 ]\n"
      "[ 2 3 3 ]\n"
      "[ 3 0 0 ]\n"
      "[ 3 0 1 ]\n"
      "[ 3 0 2 ]\n"
      "[ 3 0 3 ]\n"
      "[ 3 1 0 ]\n"
      "[ 3 1 1 ]\n"
      "[ 3 1 2 ]\n"
      "[ 3 1 3 ]\n"
      "[ 3 2 0 ]\n"
      "[ 3 2 1 ]\n"
      "[ 3 2 2 ]\n"
      "[ 3 2 3 ]\n"
      "[ 3 3 0 ]\n"
      "[ 3 3 1 ]\n"
      "[ 3 3 2 ]\n"
      "[ 3 3 3 ]\n"
      , trans2.print_ijk_poin()
      );
}

TEST(test_hypercube_transform, test_fill_shp){
  static constexpr int Pn = 8;
  HypercubeElementTransformation<double, int, 4, Pn> trans1{};
  constexpr int nnode1 = trans1.n_nodes();
  std::array<double, nnode1> shp;
  MATH::GEOMETRY::Point<double, 4> xi = {0.3, 0.2, 0.1, 0.4};
  trans1.fill_shp(xi, shp);
  for(int inode = 0; inode < nnode1; ++inode){
    ASSERT_NEAR(
        (POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][0], xi[0]) 
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][1], xi[1])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][2], xi[2])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][3], xi[3])),
      shp[inode], 1e-11
    );
  }
}
