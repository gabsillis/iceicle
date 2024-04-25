#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/polydefs/LagrangePoly.hpp"
#include <Numtool/tmp_flow_control.hpp>
#include <Numtool/matrixT.hpp>
#include <iceicle/transformations/HypercubeTransformations.hpp>
#include "gtest/gtest.h"
#include <random>

using namespace iceicle;
using namespace transformations;

TEST(test_hypercube_transform, test_get_face_nodes){
  { // Linear 2D element
    HypercubeElementTransformation<double, int, 2, 1> trans2dp1{};
    int nodes_el[4] = {0, 3, 1, 4};
    int nodes_fac[2];
    int vert_fac[2];

    trans2dp1.get_face_nodes(0, nodes_el, nodes_fac);
    ASSERT_EQ(nodes_fac[0], 3);
    ASSERT_EQ(nodes_fac[1], 0);

    trans2dp1.get_face_nodes(1, nodes_el, nodes_fac);
    ASSERT_EQ(nodes_fac[0], 0);
    ASSERT_EQ(nodes_fac[1], 1);

    trans2dp1.get_face_nodes(2, nodes_el, nodes_fac);
    ASSERT_EQ(nodes_fac[0], 1);
    ASSERT_EQ(nodes_fac[1], 4);

    trans2dp1.get_face_nodes(3, nodes_el, nodes_fac);
    ASSERT_EQ(nodes_fac[0], 4);
    ASSERT_EQ(nodes_fac[1], 3);

    trans2dp1.get_face_vert(0, nodes_el, vert_fac);
    ASSERT_EQ(vert_fac[0], 3);
    ASSERT_EQ(vert_fac[1], 0);

    trans2dp1.get_face_vert(1, nodes_el, vert_fac);
    ASSERT_EQ(vert_fac[0], 0);
    ASSERT_EQ(vert_fac[1], 1);

    trans2dp1.get_face_vert(2, nodes_el, vert_fac);
    ASSERT_EQ(vert_fac[0], 1);
    ASSERT_EQ(vert_fac[1], 4);

    trans2dp1.get_face_vert(3, nodes_el, vert_fac);
    ASSERT_EQ(vert_fac[0], 4);
    ASSERT_EQ(vert_fac[1], 3);
  }
}

TEST(test_hypercube_orient_transform, test_transform){

  using namespace NUMTOOL::TENSOR::FIXED_SIZE;
  using namespace MATH::MATRIX_T;

  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> peturb_dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{0.0, 1.0};
  std::uniform_real_distribution<double> size_dist{0.8, 1.2};

  static constexpr int ndim = 3;
  NUMTOOL::TMP::constexpr_for_range<1, 2>([&]<int Pn>{
    using TracePoint = MATH::GEOMETRY::Point<double, ndim - 1>;
    using ElPoint = MATH::GEOMETRY::Point<double, ndim>;

    HypercubeElementTransformation<double, int, ndim, Pn> domain_trans{};
    HypercubeTraceTransformation<double, int, ndim, Pn> trace_trans{};
    HypercubeTraceOrientTransformation<double, int, ndim> orient_trans{};

    // Generate a random x-direction vector
    // Tensor<double, ndim> xdir = {size_dist(engine), size_dist(engine), size_dist(engine)};
    Tensor<double, ndim> xdir = {-0.8, 0.8, 0.8};

    // Generate the tangent vectors 
    Tensor<double, ndim, ndim - 1> orthomat1 = {{ 
      {xdir[0], size_dist(engine)},
      {xdir[1], size_dist(engine)},
      {xdir[2], size_dist(engine)}
    }};
    Tensor<double, ndim> ydir = calc_ortho(orthomat1);

    Tensor<double, ndim, ndim - 1> orthomat2 = {{
      {xdir[0], ydir[0]},
      {xdir[1], ydir[1]},
      {xdir[2], ydir[2]}
    }};
    Tensor<double, ndim> zdir = calc_ortho(orthomat2);

    for(int attached_face = 0; attached_face < 6; ++attached_face){
      // Get the attached face and information 
      int face_coord = attached_face % ndim;
      bool is_negative_xi = (attached_face / ndim == 0);

      // global node total
      std::size_t n_nodes_total = 2 * domain_trans.n_nodes();
      // we doubly generate the nodes at the face to make it easy
        //      - domain_trans.n_nodes_face(attached_face);

      // Generate the first element nodes
      NodeArray<double, ndim> coord{n_nodes_total};
      
      for(int inode = 0; inode < domain_trans.n_nodes(); ++inode){
        coord[inode][0] = dotprod<double, ndim>(xdir.data(), domain_trans.reference_nodes()[inode]); 
        coord[inode][1] = dotprod<double, ndim>(ydir.data(), domain_trans.reference_nodes()[inode]); 
        coord[inode][2] = dotprod<double, ndim>(zdir.data(), domain_trans.reference_nodes()[inode]); 
      }

      // global node indices on the first element
      int nodes_el1[domain_trans.n_nodes()];
      for(int i = 0; i < domain_trans.n_nodes(); ++i) nodes_el1[i] = i;

      // Generate the second element nodes 
      // Generate the extra nodes for the face 
      // (we'll replace this later when we fuse the elements)

      // generation directions
      Tensor<double, ndim> xdir2 = xdir;
      Tensor<double, ndim> ydir2 = ydir;
      Tensor<double, ndim> zdir2 = zdir;
      if(is_negative_xi){ // flip the face direction if facing negative
        switch(face_coord){
          case 0:
            for(int i = 0; i < ndim; ++i) xdir2[i] = -xdir2[i];
            break;
          case 1:
            for(int i = 0; i < ndim; ++i) ydir2[i] = -ydir2[i];
            break;
          case 2:
            for(int i = 0; i < ndim; ++i) zdir2[i] = -zdir2[i];
            break;
        }
      }

      double dx = 2.0 / (Pn);
      int inode = domain_trans.n_nodes();
      for(int i = 0; i < Pn + 1; ++i){
        for(int j = 0; j < Pn + 1; ++j){
          for(int k = 0; k < Pn + 1; ++k){
            double xi[3] = {dx * i, dx * j, dx * k};
            coord[inode][0] = dotprod<double, ndim>(xdir.data(), xi);
            coord[inode][1] = dotprod<double, ndim>(ydir.data(), xi);
            coord[inode][2] = dotprod<double, ndim>(zdir.data(), xi);
            // offset at the attached face
            coord[inode][face_coord] += (is_negative_xi) ? -1.0 : 1.0;
            inode++;
          }
        }
      }

      // global node indices on the first element
      int nodes_el2[domain_trans.n_nodes()];
      for(int i = 0; i < domain_trans.n_nodes(); ++i) nodes_el2[i] = domain_trans.n_nodes() + i;

      // replace the face nodes to fuse the elements
      std::vector<int> i_set;
      std::vector<int> j_set;
      std::vector<int> k_set;
      if(face_coord == 0) i_set.push_back((is_negative_xi) ? 0 : Pn);
      else for(int i = 0; i < Pn + 1; ++i) i_set.push_back(i);

      if(face_coord == 1) j_set.push_back((is_negative_xi) ? 0 : Pn);
      else for(int i = 0; i < Pn + 1; ++i) j_set.push_back(i);
    
      if(face_coord == 2) k_set.push_back((is_negative_xi) ? 0 : Pn);
      else for(int i = 0; i < Pn + 1; ++i) k_set.push_back(i);

      for(int i : i_set){
        for(int j : j_set){
          for(int k : k_set){
            int ijk_l[3] = {i, j, k};
            int ijk_r[3] = {i, j, k};
            // other side for the right element (element 2)
            ijk_r[face_coord] = Pn - ijk_r[face_coord];
            nodes_el2[domain_trans.tensor_prod.convert_ijk(ijk_r)] =
              nodes_el1[domain_trans.tensor_prod.convert_ijk(ijk_l)];
          }
        }
      }

      // make sure the correct faces are attached 
      int face_vert_l[4];
      domain_trans.get_face_vert(attached_face, nodes_el1, face_vert_l);
      int attached_face_r = (attached_face < ndim) 
        ? attached_face + ndim : attached_face - ndim;
      ASSERT_EQ(domain_trans.get_face_nr(nodes_el2, face_vert_l), attached_face_r);

      // add some kind of random rotations
      domain_trans.rotate_x(nodes_el2);
      domain_trans.rotate_z(nodes_el2);
      domain_trans.rotate_z(nodes_el2);
      domain_trans.rotate_y(nodes_el2);

      for(int iorient = 0; iorient < 4; ++iorient){
        { // Ensure the transformation gives the same physical point L and R
          // get the face numbers
          int face_nr_l = attached_face;
          int face_vert_l[4];
          domain_trans.get_face_vert(face_nr_l, nodes_el1, face_vert_l);
          int face_nr_r = domain_trans.get_face_nr(nodes_el2, face_vert_l);

          // get the orientation on the right 
          int face_vert_r[4];
          domain_trans.get_face_vert(face_nr_r, nodes_el2, face_vert_r);
          int orientation_r = orient_trans.getOrientation(face_vert_l, face_vert_r);

          // Generate a random point in the trace reference domain 
          TracePoint s_l = {domain_dist(engine), domain_dist(engine)};

          // Get the physical point for the left element
          ElPoint xi_l, x_l;
          trace_trans.transform(nodes_el1, face_nr_l, s_l, xi_l);
          domain_trans.transform(coord, nodes_el1, xi_l, x_l);
          std::cout << "xL: [ " << x_l[0] << " " << x_l[1] 
            << " " << x_l[2] << " ]" << std::endl;

          // Get the physical point for the right element
          TracePoint s_r;
          ElPoint xi_r, x_r;
          orient_trans.transform(orientation_r, s_l, s_r);
          trace_trans.transform(nodes_el2, face_nr_r, s_r, xi_r);
          domain_trans.transform(coord, nodes_el2, xi_r, x_r);
          std::cout << "xR: [ " << x_r[0] << " " << x_r[1] 
            << " " << x_r[2] << " ]" << std::endl;

          // Assert that they give the same point 
          for(int idim = 0; idim < ndim; ++idim){
            ASSERT_NEAR(x_l[idim], x_r[idim], 1e-14);
          }

          // finally, rotate el_r about the face coordinate to get the next orientation  
          switch(face_coord){
            case 0:
              domain_trans.rotate_x(nodes_el2);
              break;
            case 1:
              domain_trans.rotate_y(nodes_el2);
              break;
            case 2:
              domain_trans.rotate_z(nodes_el2);
              break;
          }
        }
      }
    }
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

    NodeArray<double, 3> coord{trans_cube.n_nodes()};

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

    NodeArray<double, ndim> coord{trans_cube.n_nodes()};

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
      trans1.tensor_prod.print_ijk_poin()
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
      , trans2.tensor_prod.print_ijk_poin()
      );
}

TEST(test_hypercube_transform, test_fill_shp){
  static constexpr int Pn = 8;
  HypercubeElementTransformation<double, int, 4, Pn> trans1{};
  constexpr int nnode1 = trans1.n_nodes();
  std::array<double, nnode1> shp;
  MATH::GEOMETRY::Point<double, 4> xi = {0.3, 0.2, 0.1, 0.4};
  trans1.tensor_prod.fill_shp(trans1.interpolation_1d, xi, shp.data());
  for(int inode = 0; inode < nnode1; ++inode){
    ASSERT_NEAR(
        (POLYNOMIAL::lagrange1d<double, Pn>(trans1.tensor_prod.ijk_poin[inode][0], xi[0]) 
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.tensor_prod.ijk_poin[inode][1], xi[1])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.tensor_prod.ijk_poin[inode][2], xi[2])  
        * POLYNOMIAL::lagrange1d<double, Pn>(trans1.tensor_prod.ijk_poin[inode][3], xi[3])),
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
    trans2.tensor_prod.fill_shp(trans2.interpolation_1d, trans2.reference_nodes()[inode], shp.data());
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
    NodeArray<double, 2> node_coords(trans_lin2d.n_nodes());

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
  NodeArray<double, 2> node_coords(trans1.n_nodes());

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
  trans.tensor_prod.fill_deriv(trans.interpolation_1d, xi, dBidxj);
  trans.tensor_prod.fill_deriv(trans.interpolation_1d, xi, dBidxj_2);

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

TEST( test_hypercube_transform, test_fill_hess){

  auto lagrange0 = [](double s){ return s*(s - 1) / 2.0;};
  auto lagrange1 = [](double s){ return 1 - s * s; };
  auto lagrange2 = [](double s){ return s * (1 + s) / 2.0;};

  auto dlagrange0 = [](double s){ return s - 0.5; };
  auto dlagrange1 = [](double s){ return -2.0 * s; };
  auto dlagrange2 = [](double s){ return s + 0.5; };

  auto d2lagrange0 = [](double s){ return 1.0; };
  auto d2lagrange1 = [](double s){ return -2.0; };
  auto d2lagrange2 = [](double s){ return 1.0; };

  static constexpr int ndim = 2;
  static constexpr int Pn = 2;
  HypercubeElementTransformation<double, int, ndim, Pn> trans{};
  MATH::GEOMETRY::Point<double, 2> xi = {0.3, -0.4};

  std::vector<double> hess_data(ndim * ndim * trans.n_nodes());
  auto hess = trans.tensor_prod.fill_hess(trans.interpolation_1d, xi, hess_data.data());

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange0(xi[1]), (hess[0, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]), (hess[0, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]), (hess[0, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange0(xi[1]), (hess[0, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange1(xi[1]), (hess[1, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]), (hess[1, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]), (hess[1, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange1(xi[1]), (hess[1, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange2(xi[1]), (hess[2, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange2(xi[1]), (hess[2, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange2(xi[1]), (hess[2, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange2(xi[1]), (hess[2, 1, 1]));



  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange0(xi[1]), (hess[3, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]), (hess[3, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]), (hess[3, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange0(xi[1]), (hess[3, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange1(xi[1]), (hess[4, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]), (hess[4, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]), (hess[4, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange1(xi[1]), (hess[4, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange2(xi[1]), (hess[5, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange2(xi[1]), (hess[5, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange2(xi[1]), (hess[5, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange2(xi[1]), (hess[5, 1, 1]));



  ASSERT_DOUBLE_EQ( d2lagrange2(xi[0]) *   lagrange0(xi[1]), (hess[6, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange0(xi[1]), (hess[6, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange0(xi[1]), (hess[6, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange2(xi[0]) * d2lagrange0(xi[1]), (hess[6, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange2(xi[0]) *   lagrange1(xi[1]), (hess[7, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange1(xi[1]), (hess[7, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange1(xi[1]), (hess[7, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange2(xi[0]) * d2lagrange1(xi[1]), (hess[7, 1, 1]));

  ASSERT_DOUBLE_EQ( d2lagrange2(xi[0]) *   lagrange2(xi[1]), (hess[8, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange2(xi[1]), (hess[8, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange2(xi[0]) *  dlagrange2(xi[1]), (hess[8, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange2(xi[0]) * d2lagrange2(xi[1]), (hess[8, 1, 1]));
}

TEST( test_hypercube_transform, test_fill_hess_3d){

  static constexpr int ndim = 3;
  static constexpr int Pn = 1;

  auto lagrange0 = [](double s){ return (1.0 - s) / 2.0; };
  auto lagrange1 = [](double s){ return (1.0 + s) / 2.0; };

  auto dlagrange0 = [](double s){ return -0.5; };
  auto dlagrange1 = [](double s){ return 0.5; };

  auto d2lagrange0 = [](double s){ return 0; };
  auto d2lagrange1 = [](double s){ return 0; };

  HypercubeElementTransformation<double, int, ndim, Pn> trans{};
  MATH::GEOMETRY::Point<double, ndim> xi = {0.3, -0.4, 0.1};

  std::vector<double> hess_data(ndim * ndim * trans.n_nodes());
  auto hess = trans.tensor_prod.fill_hess(trans.interpolation_1d, xi, hess_data.data());

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange0(xi[1]) *   lagrange0(xi[2]), (hess[0, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]) *   lagrange0(xi[2]), (hess[0, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[0, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]) *   lagrange0(xi[2]), (hess[0, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange0(xi[1]) *   lagrange0(xi[2]), (hess[0, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[0, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[0, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[0, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *   lagrange0(xi[1]) * d2lagrange0(xi[2]), (hess[0, 2, 2]));

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange0(xi[1]) *   lagrange1(xi[2]), (hess[1, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]) *   lagrange1(xi[2]), (hess[1, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[1, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange0(xi[1]) *   lagrange1(xi[2]), (hess[1, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange0(xi[1]) *   lagrange1(xi[2]), (hess[1, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[1, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[1, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[1, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *   lagrange0(xi[1]) * d2lagrange1(xi[2]), (hess[1, 2, 2]));


  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange1(xi[1]) *   lagrange0(xi[2]), (hess[2, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]) *   lagrange0(xi[2]), (hess[2, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[2, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]) *   lagrange0(xi[2]), (hess[2, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange1(xi[1]) *   lagrange0(xi[2]), (hess[2, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[2, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[2, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[2, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *   lagrange1(xi[1]) * d2lagrange0(xi[2]), (hess[2, 2, 2]));

  ASSERT_DOUBLE_EQ( d2lagrange0(xi[0]) *   lagrange1(xi[1]) *   lagrange1(xi[2]), (hess[3, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]) *   lagrange1(xi[2]), (hess[3, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[3, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *  dlagrange1(xi[1]) *   lagrange1(xi[2]), (hess[3, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) * d2lagrange1(xi[1]) *   lagrange1(xi[2]), (hess[3, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[3, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange0(xi[0]) *   lagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[3, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *  dlagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[3, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange0(xi[0]) *   lagrange1(xi[1]) * d2lagrange1(xi[2]), (hess[3, 2, 2]));




  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange0(xi[1]) *   lagrange0(xi[2]), (hess[4, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]) *   lagrange0(xi[2]), (hess[4, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[4, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]) *   lagrange0(xi[2]), (hess[4, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange0(xi[1]) *   lagrange0(xi[2]), (hess[4, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[4, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[4, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange0(xi[1]) *  dlagrange0(xi[2]), (hess[4, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *   lagrange0(xi[1]) * d2lagrange0(xi[2]), (hess[4, 2, 2]));

  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange0(xi[1]) *   lagrange1(xi[2]), (hess[5, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]) *   lagrange1(xi[2]), (hess[5, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[5, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange0(xi[1]) *   lagrange1(xi[2]), (hess[5, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange0(xi[1]) *   lagrange1(xi[2]), (hess[5, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[5, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[5, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange0(xi[1]) *  dlagrange1(xi[2]), (hess[5, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *   lagrange0(xi[1]) * d2lagrange1(xi[2]), (hess[5, 2, 2]));


  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange1(xi[1]) *   lagrange0(xi[2]), (hess[6, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]) *   lagrange0(xi[2]), (hess[6, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[6, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]) *   lagrange0(xi[2]), (hess[6, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange1(xi[1]) *   lagrange0(xi[2]), (hess[6, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[6, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[6, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange1(xi[1]) *  dlagrange0(xi[2]), (hess[6, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *   lagrange1(xi[1]) * d2lagrange0(xi[2]), (hess[6, 2, 2]));

  ASSERT_DOUBLE_EQ( d2lagrange1(xi[0]) *   lagrange1(xi[1]) *   lagrange1(xi[2]), (hess[7, 0, 0]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]) *   lagrange1(xi[2]), (hess[7, 0, 1]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[7, 0, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *  dlagrange1(xi[1]) *   lagrange1(xi[2]), (hess[7, 1, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) * d2lagrange1(xi[1]) *   lagrange1(xi[2]), (hess[7, 1, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[7, 1, 2]));
  ASSERT_DOUBLE_EQ(  dlagrange1(xi[0]) *   lagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[7, 2, 0]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *  dlagrange1(xi[1]) *  dlagrange1(xi[2]), (hess[7, 2, 1]));
  ASSERT_DOUBLE_EQ(   lagrange1(xi[0]) *   lagrange1(xi[1]) * d2lagrange1(xi[2]), (hess[7, 2, 2]));
}


TEST( test_hypercube_transform, test_jacobian ){

  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

  static double epsilon = 1e-8;//std::sqrt(std::numeric_limits<double>::epsilon());

  /// Test when degenerated to a triangle
  {
    static constexpr int ndim = 2;
    static constexpr int Pn = 1;
    NodeArray<double, ndim> coord{
      {0, 0},
      {0.0, 1.0},
      {1.0, 0.0},
      {0.5, 0.5}
    };

    HypercubeElementTransformation<double, int, ndim, Pn> trans{};

    for(int k = 0; k < 10; ++k){
      // random point in domain 
        MATH::GEOMETRY::Point<double, ndim> testpt;
        for(int idim = 0; idim < ndim; ++idim) testpt[idim] = domain_dist(engine);
        int node_indices[4] = {0, 1, 2, 3};

        double xi = testpt[0];
        double eta = testpt[1];
        auto J = trans.Jacobian(coord, node_indices, testpt);
        ASSERT_NEAR(J[0][0],  (3 - eta) / 8.0, 1e-10);
        ASSERT_NEAR(J[0][1], -(1 + xi ) / 8.0, 1e-10);
        ASSERT_NEAR(J[1][0], -(1 + eta) / 8.0, 1e-10);
        ASSERT_NEAR(J[1][1],  (3 - xi ) / 8.0, 1e-10);
    }

  }
  auto rand_doub = [&]() -> double { return dist(engine); };

  NUMTOOL::TMP::constexpr_for_range<1, 5>([&]<int ndim>(){
    NUMTOOL::TMP::constexpr_for_range<1, 5>([&]<int Pn>(){
      std::cout << "ndim: " << ndim << " | Pn: " << Pn << std::endl;
      HypercubeElementTransformation<double, int, ndim, Pn> trans1{};

      int node_indices[trans1.n_nodes()];
      NodeArray<double, ndim> node_coords(trans1.n_nodes());

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

TEST( test_hypercube_transform, test_hessian ) {
  using namespace MATH::GEOMETRY;
  std::random_device rdev{};
  std::default_random_engine engine{rdev()};
  std::uniform_real_distribution<double> dist{-0.2, 0.2};
  std::uniform_real_distribution<double> domain_dist{-1.0, 1.0};

  static constexpr double epsilon = 1e-6;
  static constexpr int ntest = 50;
  static constexpr int n_domain_pts = 10;

  NUMTOOL::TMP::constexpr_for_range<1, 5>([&]<int ndim>(){
    NUMTOOL::TMP::constexpr_for_range<1, 4>([&]<int Pn>(){
      std::cout << "ndim: " << ndim << " | Pn: " << Pn << std::endl;
      HypercubeElementTransformation<double, int, ndim, Pn> trans1{};

      int node_indices[trans1.n_nodes()];
      NodeArray<double, ndim> node_coords(trans1.n_nodes());

      for(int k = 0; k < ntest; ++k){ // repeat test 50 times
        // randomly peturb
        for(int inode = 0; inode < trans1.n_nodes(); ++inode){
          node_indices[inode] = inode; // set the node index
          for(int idim = 0; idim < ndim; ++idim){
            node_coords[inode][idim] = trans1.reference_nodes()[inode][idim] + dist(engine);
          }
        }

        // test a bunch of random points in the domain 
        for(int k2 = 0; k2 < n_domain_pts; ++k2){
          Point<double, ndim> xi;
          for(int idim = 0; idim < ndim; ++idim) xi[idim] = domain_dist(engine);

          double hessfd[ndim][ndim][ndim];
          auto hess = trans1.Hessian(node_coords, node_indices, xi);

//          Point<double, ndim> unpeturb_x;

          for(int ixi = 0; ixi < ndim; ++ixi){
            for(int jxi = 0; jxi < ndim; ++jxi){

              if(ixi == jxi){ // diagonal
                // set up all the perturbations
                Point<double, ndim> xi_pi = xi;
                xi_pi[ixi] += epsilon;

                Point<double, ndim> xi_pj = xi;
                xi_pj[jxi] += epsilon;
                
                Point<double, ndim> xi_mi = xi;
                xi_mi[ixi] -= epsilon;

                Point<double, ndim> xi_mj = xi;
                xi_mj[jxi] -= epsilon;



                Point<double, ndim> xi_p2i = xi;
                xi_p2i[ixi] += 2 * epsilon;

                Point<double, ndim> xi_p2j = xi;
                xi_p2j[jxi] += 2 * epsilon;
                
                Point<double, ndim> xi_m2i = xi;
                xi_m2i[ixi] -= 2 * epsilon;

                Point<double, ndim> xi_m2j = xi;
                xi_m2j[jxi] -= 2 * epsilon;

                // calculate the transformations
                Point<double, ndim> x, x_pi, x_pj, x_mi, x_mj, x_p2i, x_p2j, x_m2i, x_m2j;
                trans1.transform(node_coords, node_indices, xi, x);
                trans1.transform(node_coords, node_indices, xi_pi, x_pi);
                trans1.transform(node_coords, node_indices, xi_pj, x_pj);
                trans1.transform(node_coords, node_indices, xi_mi, x_mi);
                trans1.transform(node_coords, node_indices, xi_mj, x_mj);

                trans1.transform(node_coords, node_indices, xi_p2i, x_p2i);
                trans1.transform(node_coords, node_indices, xi_p2j, x_p2j);
                trans1.transform(node_coords, node_indices, xi_m2i, x_m2i);
                trans1.transform(node_coords, node_indices, xi_m2j, x_m2j);
                for(int ix = 0; ix < ndim; ++ix){
                  hessfd[ix][ixi][jxi] = (
                      -x_p2i[ix]
                      + 16 * x_pi[ix] 
                      - 30 * x[ix]
                      + 16 * x_mi[ix]
                      - x_m2i[ix]
                  ) / (12 * epsilon * epsilon);
                }
              } else {
                // set up all the perturbations 
                Point<double, ndim> xi_pi_pj = xi;
                xi_pi_pj[ixi] += epsilon;
                xi_pi_pj[jxi] += epsilon;

                Point<double, ndim> xi_pi_mj = xi;
                xi_pi_mj[ixi] += epsilon;
                xi_pi_mj[jxi] -= epsilon;

                Point<double, ndim> xi_mi_pj = xi;
                xi_mi_pj[ixi] -= epsilon;
                xi_mi_pj[jxi] += epsilon;

                Point<double, ndim> xi_mi_mj = xi;
                xi_mi_mj[ixi] -= epsilon;
                xi_mi_mj[jxi] -= epsilon;

                // calcualate the transformations 
                Point<double, ndim> x_pi_pj, x_pi_mj, x_mi_pj, x_mi_mj;
                trans1.transform(node_coords, node_indices, xi_pi_pj, x_pi_pj);
                trans1.transform(node_coords, node_indices, xi_pi_mj, x_pi_mj);
                trans1.transform(node_coords, node_indices, xi_mi_pj, x_mi_pj);
                trans1.transform(node_coords, node_indices, xi_mi_mj, x_mi_mj);

                for(int ix = 0; ix < ndim; ++ix){
                  hessfd[ix][ixi][jxi] = (
                    x_pi_pj[ix]
                    - x_pi_mj[ix]
                    - x_mi_pj[ix]
                    + x_mi_mj[ix]
                  ) / (4 * epsilon * epsilon);
                }
              }
            }
          }

          // print the hessians 
//          auto print_hess = []<class hess_type>(const hess_type &hess){
//            std::cout << std::setprecision(5);
//            for(int ix = 0; ix < ndim; ++ix){
//              for(int ixi = 0; ixi < ndim; ++ixi){
//                std::cout << "| ";
//                for(int jxi = 0; jxi < ndim; ++jxi){
//                  std::cout << std::setw(7) << hess[ix][ixi][jxi] << " ";
//                }
//                std::cout << "|" << std::endl;
//              }
//              std::cout << std::endl;
//            }
//          };
//          std::cout << "Calculated Hessian" << std::endl;
//          print_hess(hess);
//          std::cout << std::endl;
//          std::cout << "FD Hessian" << std::endl;
//          print_hess(hessfd);

          double scaled_tol = epsilon * 5000 * (std::pow(10, 0.4 * (Pn - 1)));
          for(int ix = 0; ix < ndim; ++ix) for(int ixi = 0; ixi < ndim; ++ixi) for(int jxi = 0; jxi < ndim; ++jxi)
            { ASSERT_NEAR(hess[ix][ixi][jxi], hessfd[ix][ixi][jxi], scaled_tol); }
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
  ASSERT_EQ(facevert[0], 6);
  ASSERT_EQ(facevert[1], 8);
  ASSERT_EQ(facevert[2], 0);
  ASSERT_EQ(facevert[3], 2);

  trans.get_face_vert(1, gnodes, facevert);
  ASSERT_EQ(facevert[0], 0);
  ASSERT_EQ(facevert[1], 2);
  ASSERT_EQ(facevert[2], 18);
  ASSERT_EQ(facevert[3], 20);

  trans.get_face_vert(2, gnodes, facevert);
  ASSERT_EQ(facevert[0], 18);
  ASSERT_EQ(facevert[1], 24);
  ASSERT_EQ(facevert[2], 0);
  ASSERT_EQ(facevert[3], 6);

  trans.get_face_vert(3, gnodes, facevert);
  ASSERT_EQ(facevert[0], 18);
  ASSERT_EQ(facevert[1], 20);
  ASSERT_EQ(facevert[2], 24);
  ASSERT_EQ(facevert[3], 26);

  trans.get_face_vert(4, gnodes, facevert);
  ASSERT_EQ(facevert[0], 24);
  ASSERT_EQ(facevert[1], 26);
  ASSERT_EQ(facevert[2], 6);
  ASSERT_EQ(facevert[3], 8);

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
