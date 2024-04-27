#include "iceicle/geometry/face.hpp"
#include <iceicle/fe_definitions.hpp>
#include "iceicle/geometry/hypercube_element.hpp"
#include <iceicle/geometry/hypercube_face.hpp>
#include "gtest/gtest.h"

using namespace iceicle;

TEST(test_hypercube_face, test_transform){
    static constexpr int ndim = 2;
    using T = double;
    using IDX = int;
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    std::size_t n_nodes = 6;
    NodeArray<T, ndim> coord{6};
    coord[0][0] = -1;
    coord[0][1] = 0;

    coord[1][0] = 0;
    coord[1][1] = 0;

    coord[2][0] = 1;
    coord[2][1] = 0;

    coord[3][0] = -1;
    coord[3][1] = 1;

    coord[4][0] = 0;
    coord[4][1] = 1;

    coord[5][0] = 1;
    coord[5][1] = 1;

    HypercubeElement<T, IDX, ndim, 1> elL{}, elR{};
    elL.setNode(0, 0);
    elL.setNode(1, 3);
    elL.setNode(2, 1);
    elL.setNode(3, 4);

    elR.setNode(0, 1);
    elR.setNode(1, 4);
    elR.setNode(2, 2);
    elR.setNode(3, 5);

    auto transL = decltype(elL)::transformation;
    auto transR = decltype(elR)::transformation;

    using FaceType = HypercubeFace<T, IDX, ndim, 1>;

    int face_nr_l = ndim + 0; // positive side
    int face_nr_r = 0;
    Tensor<IDX, FaceType::trans.n_nodes> face_nodes;
    transL.get_face_nodes(face_nr_l, elL.nodes(), face_nodes.data());

    ASSERT_EQ(1, face_nodes[0]);
    ASSERT_EQ(4, face_nodes[1]);

    int face_vert_l[2];
    int face_vert_r[2];
    transL.get_face_vert(face_nr_l, elL.nodes(), face_vert_l);
    transR.get_face_vert(face_nr_r, elR.nodes(), face_vert_r);

    ASSERT_EQ(1, face_vert_l[0]);
    ASSERT_EQ(4, face_vert_l[1]);

    ASSERT_EQ(4, face_vert_r[0]);
    ASSERT_EQ(1, face_vert_r[1]);

    int orientationr = FaceType::orient_trans.getOrientation(face_vert_l, face_vert_r);

    ASSERT_EQ(face_nr_l, transL.get_face_nr(elL.nodes(), face_vert_l));
    ASSERT_EQ(face_nr_r, transR.get_face_nr(elR.nodes(), face_vert_r));

    FaceType face(0, 1, face_nodes, face_nr_l, face_nr_r, orientationr, BOUNDARY_CONDITIONS::PERIODIC, 0);

    using FacePoint = MATH::GEOMETRY::Point<T, ndim-1>;
    using DomainPoint = MATH::GEOMETRY::Point<T, ndim>;

    DomainPoint xiL, xiR;
    FacePoint s = {-0.8};
    face.transform_xiL(s, xiL);
    face.transform_xiR(s, xiR);

    ASSERT_EQ(xiL[0],  1.0);
    ASSERT_EQ(xiL[1], -0.8);

    ASSERT_EQ(xiR[0], -1.0);
    ASSERT_EQ(xiR[1], -0.8);

    DomainPoint xL, xR;
    elL.transform(coord, xiL, xL);
    elR.transform(coord, xiR, xR);

    ASSERT_DOUBLE_EQ(0.0, xL[0]);
    ASSERT_DOUBLE_EQ(0.1, xL[1]);

    ASSERT_DOUBLE_EQ(0.0, xR[0]);
    ASSERT_DOUBLE_EQ(0.1, xR[1]);
    
}
