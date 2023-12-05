#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/matrix/dense_matrix.hpp"
#include "Numtool/polydefs/LagrangePoly.hpp"
#include <Numtool/tmp_flow_control.hpp>
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include <iceicle/transformations/HypercubeElementTransformation.hpp>
#include "gtest/gtest.h"
#include <random>
#include <limits>

using namespace ELEMENT::TRANSFORMATIONS;

TEST(test_hypercube_orient_transform, test_transform){

  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> peturb_dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{0.0, 1.0};

  NUMTOOL::TMP::constexpr_for_range<2, 5>([&]<int ndim>{
      NUMTOOL::TMP::constexpr_for_range<1, 4>([&]<int Pn>{
      using TracePoint = MATH::GEOMETRY::Point<double, ndim - 1>;
      using ElPoint = MATH::GEOMETRY::Point<double, ndim>;

      HypercubeElementTransformation<double, int, ndim, Pn> domain_trans{};
      HypercubeTraceTransformation<double, int, ndim, Pn> trace_trans{};
      HypercubeTraceOrientTransformation<double, int, ndim> orient_trans{};

      FE::NodalFEFunction<double, ndim> coord{domain_trans.n_nodes()};

      // peturb the coordinates 
      for(int inode = 0; inode < domain_trans.n_nodes(); ++inode){
        for(int idim = 0; idim < ndim; ++idim){
          coord[inode][idim] = domain_trans.reference_nodes()[inode][idim] + peturb_dist(engine);
        }
      }

      // global node index array
      int gnodes[domain_trans.n_nodes()];
      for(int i = 0; i < domain_trans.n_nodes(); ++i) gnodes[i] = i;

      for(int iface = 0; iface < 2 * ndim; ++iface){

      }
    
    });
  });
}

TEST(test_hypercube_trace_transform, test_jacobian){

  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> peturb_dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

  { // unpeturbed cube normals
    
    using TracePoint = MATH::GEOMETRY::Point<double, 2>;
    using ElPoint = MATH::GEOMETRY::Point<double, 3>;
    // Generate a cube
    static constexpr int ndim = 3;
    HypercubeElementTransformation<double, int, ndim, 1> trans_cube{};

    FE::NodalFEFunction<double, 3> coord{trans_cube.n_nodes()};

    // unpeturbed coordinates 
    for(int inode = 0; inode < trans_cube.n_nodes(); ++inode){
      for(int idim = 0; idim < ndim; ++idim){
        coord[inode][idim] = trans_cube.reference_nodes()[inode][idim];
      }
    }

    // global node index array
    int gnodes[trans_cube.n_nodes()];
    for(int i = 0; i < trans_cube.n_nodes(); ++i) gnodes[i] = i;

    // ensure the normals are correct for each face number 
    HypercubeTraceTransformation<double, int, 3, 1> trans_face{};

    for(int facenr = 0; facenr < 6; ++facenr){
      TracePoint s = {domain_dist(engine), domain_dist(engine)};
      ElPoint xi;
      trans_face.transform(gnodes, facenr, s, xi);

      // Get the element jacobian 
      auto Jel = trans_cube.Jacobian(coord, gnodes, xi);
      auto Jtrace = trans_face.Jacobian(gnodes, facenr, s, Jel);

      auto normal = NUMTOOL::TENSOR::FIXED_SIZE::calc_ortho(Jtrace);

      int trace_coord = facenr % ndim;
      if(facenr / ndim == 0){
        // -1 side: normal should be -1 in direction trace_coord
        // NOTE: this isn't 4 because the area of the reference face is 4
        // if you want surface area from normal you need to include quadrature rule
        // or reference domain area information
        for(int idim = 0; idim < ndim; ++idim){
          if(idim == trace_coord)
            { ASSERT_DOUBLE_EQ(-1.0, normal[idim]); }
          else 
            { ASSERT_NEAR(0.0, normal[idim], 1e-15); }
        }
      } else {
        // 1 side: normal should be 1 at trace_coord
        for(int idim = 0; idim < ndim; ++idim){
          if(idim == trace_coord)
            { ASSERT_DOUBLE_EQ(1.0, normal[idim]); }
          else 
            { ASSERT_NEAR(0.0, normal[idim], 1e-15); }
        }
      }
    }
  }

  { // petrubed cube normals

    static constexpr int ndim = 3;
    static constexpr int Pn = 1;
    using TracePoint = MATH::GEOMETRY::Point<double, ndim - 1>;
    using ElPoint = MATH::GEOMETRY::Point<double, ndim>;
    // Generate a cube
    HypercubeElementTransformation<double, int, ndim, Pn> trans_cube{};

    FE::NodalFEFunction<double, ndim> coord{trans_cube.n_nodes()};

    // peturbed coordinates 
    for(int inode = 0; inode < trans_cube.n_nodes(); ++inode){
      for(int idim = 0; idim < ndim; ++idim){
        coord[inode][idim] = trans_cube.reference_nodes()[inode][idim] + peturb_dist(engine);
      }
    }

    // global node index array
    int gnodes[trans_cube.n_nodes()];
    for(int i = 0; i < trans_cube.n_nodes(); ++i) gnodes[i] = i;

    // ensure the normals are correct for each face number 
    HypercubeTraceTransformation<double, int, ndim, Pn> trans_face{};

    for(int facenr = 0; facenr < 2 * ndim; ++facenr){
      TracePoint s = {domain_dist(engine), domain_dist(engine)};
      ElPoint xi;
      trans_face.transform(gnodes, facenr, s, xi);

      // Get the element jacobian 
      auto Jel = trans_cube.Jacobian(coord, gnodes, xi);
      // make sure everything in the element jacobian is not nan 
      for(int i = 0; i < ndim; ++i){
        for(int j = 0; j < ndim; ++j){
          ASSERT_EQ(Jel[i][j], Jel[i][j]);
        }
      }

      // get the trace jacobian
      auto Jtrace = trans_face.Jacobian(gnodes, facenr, s, Jel);

      // get the trace jacobian from global nodes 
      int face_nodes[trans_cube.n_nodes_face(facenr)];
      trans_cube.get_face_nodes(facenr, gnodes, face_nodes);

      auto Jtrace2 = trans_face.Jacobian(coord, face_nodes, facenr, s);

      // finite difference
      static double epsilon = 1e-8;

      ElPoint unpeturb_transform{};
      trans_cube.transform(coord, gnodes, xi, unpeturb_transform);

      for(int is = 0; is < ndim - 1; ++is){
        double tmp = s[is];
        s[is] += epsilon;
        ElPoint xi_2;
        trans_face.transform(gnodes, facenr, s, xi_2);

        ElPoint peturb_transform{};
        trans_cube.transform(coord, gnodes, xi_2, peturb_transform);

        double scaled_tol = 1e-6 * (std::pow(10, 0.4 * (Pn - 1)));
        for(int ix = 0; ix < ndim; ++ix){
          double Jfdterm = (peturb_transform[ix] - unpeturb_transform[ix]) / epsilon;
          ASSERT_NEAR(Jfdterm, Jtrace[ix][is], scaled_tol);
          //ASSERT_NEAR(Jfdterm, Jtrace2[ix][is], scaled_tol);
        }

        s[is] = tmp;
      }
    }
  }
}

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
  trans1.fill_shp(xi, shp.data());
  for(int inode = 0; inode < nnode1; ++inode){
    ASSERT_NEAR(
        (POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][0], xi[0]) 
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][1], xi[1])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][2], xi[2])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.ijk_poin[inode][3], xi[3])),
      shp[inode], 1e-13
    );
  }
}

TEST( test_hypercube_transform, test_ref_coordinates ){
  HypercubeElementTransformation<double, int, 3, 2> trans1{};

  const MATH::GEOMETRY::Point<double, 3> *xi1 = trans1.reference_nodes();
  ASSERT_DOUBLE_EQ(-1.0, xi1[0][0]);
  ASSERT_DOUBLE_EQ(-1.0, xi1[0][1]);
  ASSERT_DOUBLE_EQ(-1.0, xi1[0][2]);

  ASSERT_DOUBLE_EQ(-1.0, xi1[1][0]);
  ASSERT_DOUBLE_EQ(-1.0, xi1[1][1]);
  ASSERT_DOUBLE_EQ( 0.0, xi1[1][2]);

  ASSERT_DOUBLE_EQ(-1.0, xi1[2][0]);
  ASSERT_DOUBLE_EQ(-1.0, xi1[2][1]);
  ASSERT_DOUBLE_EQ( 1.0, xi1[2][2]);

  ASSERT_DOUBLE_EQ(-1.0, xi1[3][0]);
  ASSERT_DOUBLE_EQ( 0.0, xi1[3][1]);
  ASSERT_DOUBLE_EQ(-1.0, xi1[3][2]);

  ASSERT_DOUBLE_EQ(-1.0, xi1[4][0]);
  ASSERT_DOUBLE_EQ( 0.0, xi1[4][1]);
  ASSERT_DOUBLE_EQ( 0.0, xi1[4][2]);

  ASSERT_DOUBLE_EQ(-1.0, xi1[5][0]);
  ASSERT_DOUBLE_EQ( 0.0, xi1[5][1]);
  ASSERT_DOUBLE_EQ( 1.0, xi1[5][2]);



  HypercubeElementTransformation<double, int, 3, 4> trans2{};
  constexpr int nnode2 = trans2.n_nodes();
  std::array<double, nnode2> shp;

  for(int inode = 0; inode < nnode2; ++inode){
    trans2.fill_shp(trans2.reference_nodes()[inode], shp.data());
    for(int jnode = 0; jnode < nnode2; ++jnode){
      if(inode == jnode) ASSERT_DOUBLE_EQ(1.0, shp[jnode]);
      else ASSERT_DOUBLE_EQ(0.0, shp[jnode]);
    }

  }
}

TEST( test_hypercube_transform, test_transform ){
  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

  auto rand_doub = [&]() -> double { return dist(engine); };

  HypercubeElementTransformation<double, int, 2, 1> trans_lin2d{};
  auto lagrange0 = [](double s){ return (1.0 - s) / 2.0; };
  auto lagrange1 = [](double s){ return (1.0 + s) / 2.0; };

  { // linear 2d transformation test
    int node_indices[trans_lin2d.n_nodes()];
    FE::NodalFEFunction<double, 2> node_coords(trans_lin2d.n_nodes());

    // randomly peturb
    for(int inode = 0; inode < trans_lin2d.n_nodes(); ++inode){
      node_indices[inode] = inode; // set the node index
      for(int idim = 0; idim < 2; ++idim){
        node_coords[inode][idim] = trans_lin2d.reference_nodes()[inode][idim] + rand_doub();
      }
    }
    for(int k = 0; k < 1000; ++k){
      MATH::GEOMETRY::Point<double, 2> xi = {domain_dist(engine), domain_dist(engine)};
      MATH::GEOMETRY::Point<double, 2> x_act = {
        lagrange0(xi[0]) * lagrange0(xi[1]) * node_coords[0][0]
        + lagrange0(xi[0]) * lagrange1(xi[1]) * node_coords[1][0]
        + lagrange1(xi[0]) * lagrange0(xi[1]) * node_coords[2][0]
        + lagrange1(xi[0]) * lagrange1(xi[1]) * node_coords[3][0],

        lagrange0(xi[0]) * lagrange0(xi[1]) * node_coords[0][1]
        + lagrange0(xi[0]) * lagrange1(xi[1]) * node_coords[1][1]
        + lagrange1(xi[0]) * lagrange0(xi[1]) * node_coords[2][1]
        + lagrange1(xi[0]) * lagrange1(xi[1]) * node_coords[3][1]
      };

      MATH::GEOMETRY::Point<double, 2> x_trans;
      trans_lin2d.transform(node_coords, node_indices, xi, x_trans);
      ASSERT_NEAR(x_trans[0], x_act[0], 1e-15);
      ASSERT_NEAR(x_trans[1], x_act[1], 1e-15);
    }
  }

  HypercubeElementTransformation<double, int, 2, 3> trans1{};

  int node_indices[trans1.n_nodes()];
  FE::NodalFEFunction<double, 2> node_coords(trans1.n_nodes());

  // randomly peturb
  for(int inode = 0; inode < trans1.n_nodes(); ++inode){
    node_indices[inode] = inode; // set the node index
    for(int idim = 0; idim < 2; ++idim){
      node_coords[inode][idim] = trans1.reference_nodes()[inode][idim] + rand_doub();
    }
  }

  // Require: kronecker property
  for(int inode = 0; inode < trans1.n_nodes(); ++inode){
    const MATH::GEOMETRY::Point<double, 2> &xi = trans1.reference_nodes()[inode];
    MATH::GEOMETRY::Point<double, 2> x;
    trans1.transform(node_coords, node_indices, xi, x);
    for(int idim = 0; idim < 2; ++idim){
      ASSERT_NEAR(node_coords[inode][idim], x[idim], 1e-14); // tiny bit of roundoff error
    }
  }
}

TEST( test_hypercube_transform, test_fill_deriv ){
  static constexpr int ndim = 2;
  static constexpr int Pn = 1;
  HypercubeElementTransformation<double, int, ndim, Pn> trans{};

  auto lagrange0 = [](double s){ return (1.0 - s) / 2.0; };
  auto lagrange1 = [](double s){ return (1.0 + s) / 2.0; };

  auto dlagrange0 = [](double s){ return -0.5; };
  auto dlagrange1 = [](double s){ return 0.5; };

  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, 4, 2> dBidxj;
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, 4, 2> dBidxj_2;
  MATH::GEOMETRY::Point<double, 2> xi = {0.3, -0.3};
  trans.fill_deriv(xi, dBidxj);
  trans.fill_deriv(xi, dBidxj_2);

  for(int inode = 0; inode < 4; ++inode) for(int idim = 0; idim < 2; ++idim) ASSERT_DOUBLE_EQ(dBidxj[inode][idim], dBidxj_2[inode][idim]);

  ASSERT_DOUBLE_EQ( dlagrange0(xi[0]) *  lagrange0(xi[1]), dBidxj[0][0]);
  ASSERT_DOUBLE_EQ(  lagrange0(xi[0]) * dlagrange0(xi[1]), dBidxj[0][1]);

  ASSERT_DOUBLE_EQ( dlagrange0(xi[0]) *  lagrange1(xi[1]), dBidxj[1][0]);
  ASSERT_DOUBLE_EQ(  lagrange0(xi[0]) * dlagrange1(xi[1]), dBidxj[1][1]);

  ASSERT_DOUBLE_EQ( dlagrange1(xi[0]) *  lagrange0(xi[1]), dBidxj[2][0]);
  ASSERT_DOUBLE_EQ(  lagrange1(xi[0]) * dlagrange0(xi[1]), dBidxj[2][1]);

  ASSERT_DOUBLE_EQ( dlagrange1(xi[0]) *  lagrange1(xi[1]), dBidxj[3][0]);
  ASSERT_DOUBLE_EQ(  lagrange1(xi[0]) * dlagrange1(xi[1]), dBidxj[3][1]);
}


TEST( test_hypercube_transform, test_jacobian ){
  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

  static double epsilon = 1e-8;//std::sqrt(std::numeric_limits<double>::epsilon());

  auto rand_doub = [&]() -> double { return dist(engine); };

  NUMTOOL::TMP::constexpr_for_range<1, 5>([&]<int ndim>(){
    NUMTOOL::TMP::constexpr_for_range<1, 9>([&]<int Pn>(){
      std::cout << "ndim: " << ndim << " | Pn: " << Pn << std::endl;
      HypercubeElementTransformation<double, int, ndim, Pn> trans1{};

      int node_indices[trans1.n_nodes()];
      FE::NodalFEFunction<double, ndim> node_coords(trans1.n_nodes());

      for(int k = 0; k < 50; ++k){ // repeat test 50 times
        // randomly peturb
        for(int inode = 0; inode < trans1.n_nodes(); ++inode){
          node_indices[inode] = inode; // set the node index
          for(int idim = 0; idim < ndim; ++idim){
            node_coords[inode][idim] = trans1.reference_nodes()[inode][idim] + rand_doub();
          }
        }

    //    std::cout << "pt1: [ " << node_coords[0][0] << ", " << node_coords[0][1] << " ]" << std::endl;
    //    std::cout << "pt2: [ " << node_coords[1][0] << ", " << node_coords[1][1] << " ]" << std::endl;
    //    std::cout << "pt3: [ " << node_coords[2][0] << ", " << node_coords[2][1] << " ]" << std::endl;
    //    std::cout << "pt4: [ " << node_coords[3][0] << ", " << node_coords[3][1] << " ]" << std::endl;
        // test 10 random points in the domain 
        for(int k2 = 0; k2 < 10; ++k2){
          MATH::GEOMETRY::Point<double, ndim> testpt;
          for(int idim = 0; idim < ndim; ++idim) testpt[idim] = domain_dist(engine);

          double Jfd[ndim][ndim];
          auto Jtrans = trans1.Jacobian(node_coords, node_indices, testpt);

          MATH::GEOMETRY::Point<double, ndim> unpeturb_transform;
          trans1.transform(node_coords, node_indices, testpt, unpeturb_transform);

          for(int ixi = 0; ixi < ndim; ++ixi){
            double temp = testpt[ixi];
            testpt[ixi] += epsilon;
            MATH::GEOMETRY::Point<double, ndim> peturb_transform;
            trans1.transform(node_coords, node_indices, testpt, peturb_transform);
            for(int ix = 0; ix < ndim; ++ix){
              Jfd[ix][ixi] = (peturb_transform[ix] - unpeturb_transform[ix]) / epsilon;

            }
            testpt[ixi] = temp;
          }

    //      std::cout << Jtrans[0][0] << " " << Jtrans[0][1] << std::endl;
    //      std::cout << Jtrans[1][0] << " " << Jtrans[1][1] << std::endl;
    //      std::cout << std::endl;
    //
    //      std::cout << Jfd[0][0] << " " << Jfd[0][1] << std::endl;
    //      std::cout << Jfd[1][0] << " " << Jfd[1][1] << std::endl;
    //      std::cout << std::endl;
          double scaled_tol = 1e-6 * (std::pow(10, 0.4 * (Pn - 1)));
          for(int ix = 0; ix < ndim; ++ix) for(int ixi = 0; ixi < ndim; ++ixi) ASSERT_NEAR(Jtrans[ix][ixi], Jfd[ix][ixi], scaled_tol);
        }
      }
    });
  });
}


TEST(test_hypercube_transform, test_get_element_vert){

  HypercubeElementTransformation<double, int, 3, 2> trans{};

  int gnodes[trans.n_nodes()];
  for(int i = 0; i < trans.n_nodes(); ++i) gnodes[i] = i;

  int gvert[trans.n_vert()];

  trans.get_element_vert(gnodes, gvert);

  ASSERT_EQ(gvert[0], 0);
  ASSERT_EQ(gvert[1], 2);
  ASSERT_EQ(gvert[2], 6);
  ASSERT_EQ(gvert[3], 8);
  ASSERT_EQ(gvert[4], 18);
  ASSERT_EQ(gvert[5], 20);
  ASSERT_EQ(gvert[6], 24);
  ASSERT_EQ(gvert[7], 26);
}

TEST(test_hypercube_transform, test_get_face_vert){

  HypercubeElementTransformation<double, int, 3, 2> trans{};

  int gnodes[trans.n_nodes()];
  for(int i = 0; i < trans.n_nodes(); ++i) gnodes[i] = i;

  int facevert[trans.n_facevert(0)];

  trans.get_face_vert(0, gnodes, facevert);
  ASSERT_EQ(facevert[0], 0);
  ASSERT_EQ(facevert[1], 2);
  ASSERT_EQ(facevert[2], 6);
  ASSERT_EQ(facevert[3], 8);

  trans.get_face_vert(1, gnodes, facevert);
  ASSERT_EQ(facevert[0], 0);
  ASSERT_EQ(facevert[1], 2);
  ASSERT_EQ(facevert[2], 18);
  ASSERT_EQ(facevert[3], 20);

  trans.get_face_vert(2, gnodes, facevert);
  ASSERT_EQ(facevert[0], 0);
  ASSERT_EQ(facevert[1], 6);
  ASSERT_EQ(facevert[2], 18);
  ASSERT_EQ(facevert[3], 24);

  trans.get_face_vert(3, gnodes, facevert);
  ASSERT_EQ(facevert[0], 18);
  ASSERT_EQ(facevert[1], 20);
  ASSERT_EQ(facevert[2], 24);
  ASSERT_EQ(facevert[3], 26);

  trans.get_face_vert(4, gnodes, facevert);
  ASSERT_EQ(facevert[0], 6);
  ASSERT_EQ(facevert[1], 8);
  ASSERT_EQ(facevert[2], 24);
  ASSERT_EQ(facevert[3], 26);

  trans.get_face_vert(5, gnodes, facevert);
  ASSERT_EQ(facevert[0], 2);
  ASSERT_EQ(facevert[1], 8);
  ASSERT_EQ(facevert[2], 20);
  ASSERT_EQ(facevert[3], 26);
}


TEST(test_hypercube_transform, test_get_face_number){

  HypercubeElementTransformation<double, int, 3, 2> trans{};

  int gnodes[trans.n_nodes()];
  for(int i = 0; i < trans.n_nodes(); ++i) gnodes[i] = i;

  int facevert0[trans.n_facevert(0)] = {0, 2, 6, 8};
  int facevert1[trans.n_facevert(1)] = {0, 2, 18, 20};
  int facevert2[trans.n_facevert(2)] = {0, 6, 18, 24};
  int facevert3[trans.n_facevert(3)] = {18, 20, 24, 26};
  int facevert4[trans.n_facevert(4)] = {6, 26, 24, 8};
  int facevert5[trans.n_facevert(5)] = {26, 2, 8, 20};

  ASSERT_EQ(0, trans.get_face_nr(gnodes, facevert0));
  ASSERT_EQ(1, trans.get_face_nr(gnodes, facevert1));
  ASSERT_EQ(2, trans.get_face_nr(gnodes, facevert2));
  ASSERT_EQ(3, trans.get_face_nr(gnodes, facevert3));
  ASSERT_EQ(4, trans.get_face_nr(gnodes, facevert4));
  ASSERT_EQ(5, trans.get_face_nr(gnodes, facevert5));
}
