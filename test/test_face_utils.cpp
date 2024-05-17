
#include "iceicle/anomaly_log.hpp"
#include <iceicle/geometry/face_utils.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <gtest/gtest.h>

TEST(test_face_utils, test_make_face){
    using namespace iceicle;
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    using T = build_config::T;
    using IDX = build_config::IDX;

    // ======
    // = 1D =
    // ======
    {
        static constexpr int ndim = 1;
        using Face_t = Face<T, IDX, ndim>;
        
        HypercubeElement<T, IDX, ndim, 1> el1{{0, 1}};
        HypercubeElement<T, IDX, ndim, 1> el2{{0, 2}};

        auto face1_opt = make_face(0, 1, &el1, &el2, BOUNDARY_CONDITIONS::INTERIOR, 0);
        ASSERT_TRUE((bool) face1_opt);
        Face_t* face1 = face1_opt.value();

        // basic face data
        ASSERT_EQ(face1->elemL, 0);
        ASSERT_EQ(face1->elemR, 1);
        ASSERT_EQ(face1->face_nr_l(), 0);
        ASSERT_EQ(face1->face_nr_r(), 0);
        ASSERT_EQ(face1->orientation_r(), 0);
        ASSERT_EQ(face1->geometry_order(), 1);

        // normal vector 
        NodeArray<T, ndim> nodes1{{0.0}, {1.0}, {-1.0}, {2.0}, {3.0}};
        {
            auto centroid_fac = face_centroid(*face1, nodes1);
            auto centroid1 = el1.centroid(nodes1);
            auto centroid2 = el2.centroid(nodes1);
            Tensor<T, ndim> internal_l{centroid1[0] - centroid_fac[0]};
            Tensor<T, ndim> internal_r{centroid2[0] - centroid_fac[0]};
            auto normal = calc_normal(*face1, nodes1, {});
            ASSERT_TRUE(dot(normal, internal_l) < 0.0);
            ASSERT_TRUE(dot(normal, internal_r) > 0.0);
        }

        HypercubeElement<T, IDX, ndim, 2> el3{{1, 3, 4}};
        auto face2_opt = make_face(0, 2, &el1, &el3, BOUNDARY_CONDITIONS::INTERIOR, 0);
        ASSERT_TRUE((bool) face2_opt);
        Face_t* face2 = face2_opt.value();

        // basic face data
        ASSERT_EQ(face2->elemL, 0);
        ASSERT_EQ(face2->elemR, 2);
        ASSERT_EQ(face2->face_nr_l(), 1);
        ASSERT_EQ(face2->face_nr_r(), 0);
        ASSERT_EQ(face2->orientation_r(), 0);
        ASSERT_EQ(face2->geometry_order(), 1);

        // normal vector 
        {
            auto centroid_fac = face_centroid(*face2, nodes1);
            auto centroidl = el1.centroid(nodes1);
            auto centroidr = el3.centroid(nodes1);
            Tensor<T, ndim> internal_l{centroidl[0] - centroid_fac[0]};
            Tensor<T, ndim> internal_r{centroidr[0] - centroid_fac[0]};
            auto normal = calc_normal(*face2, nodes1, {});
            ASSERT_TRUE(dot(normal, internal_l) < 0.0);
            ASSERT_TRUE(dot(normal, internal_r) > 0.0);
        }

        HypercubeElement<T, IDX, ndim, 1> el4{{3, 4}};
        auto face3_opt = make_face(0, 4, &el1, &el4, BOUNDARY_CONDITIONS::INTERIOR, 0);
        ASSERT_FALSE((bool) face3_opt);
        delete face1;
        delete face2;
    }

    // ======
    // = 2D =
    // ======
    {
        static constexpr int ndim = 2;
        using Face_t = Face<T, IDX, ndim>;

        HypercubeElement<T, IDX, ndim, 1> el0{{0, 1, 2, 3}};
        HypercubeElement<T, IDX, ndim, 1> el1{{3, 2, 4, 5}};

        auto face1_opt = make_face(0, 1, &el0, &el1);
        ASSERT_TRUE((bool) face1_opt);
        Face_t* face1 = face1_opt.value();

        // basic face data
        ASSERT_EQ(face1->elemL, 0);
        ASSERT_EQ(face1->elemR, 1);
        ASSERT_EQ(face1->face_nr_l(), 2);
        ASSERT_EQ(face1->face_nr_r(), 0);
        ASSERT_EQ(face1->geometry_order(), 1);

        NodeArray<T, ndim> nodes1{{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}, {2.0, 1.0}, {2.0, 0.0}};
        // normal vector 
        {
            auto centroid_fac = face_centroid(*face1, nodes1);
            auto centroid_l = el0.centroid(nodes1);
            auto centroid_r = el1.centroid(nodes1);
            Tensor<T, ndim> internal_l{centroid_l[0] - centroid_fac[0]};
            Tensor<T, ndim> internal_r{centroid_r[0] - centroid_fac[0]};
            auto normal = calc_normal(*face1, nodes1, {0.0});
            ASSERT_TRUE(dot(normal, internal_l) < 0.0);
            ASSERT_TRUE(dot(normal, internal_r) > 0.0);
        }

        // NOTE: 1 and 2 don't form a face in el0
        HypercubeElement<T, IDX, ndim, 1> el2{{1, 6, 8, 2}};
        auto face2_opt = make_face(0, 2, &el0, &el2);
        ASSERT_FALSE((bool) face2_opt);

        NodeArray<T, ndim> nodes2{ 
            {0.0, 0.0}, // 0
            {0.0, 1.0}, // 1
            {1.0, 0.0}, // 2
            {1.0, 0.5}, // 3
            {1.0, 1.0}, // 4
            {1.5, 0.0}, // 5
            {1.5, 0.5}, // 6
            {1.5, 0.1}, // 7
            {2.0, 0.0}, // 8
            {2.0, 0.5}, // 9
            {2.0, 1.0}, // 10
        };
        HypercubeElement<T, IDX, ndim, 1> el3{{0, 1, 2, 4}};
        HypercubeElement<T, IDX, ndim, 2> el4{{ 2, 3, 4, 5, 6, 7, 8, 9, 10}};
        auto face3_opt = make_face(3, 4, &el3, &el4);
        ASSERT_TRUE((bool) face3_opt);
        Face_t* face3 = face3_opt.value();
        ASSERT_EQ(face3->geometry_order(), 1);

        // normal vector 
        {
            auto centroid_fac = face_centroid(*face3, nodes2);
            auto centroid_l = el3.centroid(nodes2);
            auto centroid_r = el4.centroid(nodes2);
            Tensor<T, ndim> internal_l{centroid_l[0] - centroid_fac[0]};
            Tensor<T, ndim> internal_r{centroid_r[0] - centroid_fac[0]};
            auto normal = calc_normal(*face3, nodes2, {0.0});
            ASSERT_TRUE(dot(normal, internal_l) < 0.0);
            ASSERT_TRUE(dot(normal, internal_r) > 0.0);
        }
        
        delete face1;
        delete face3;
    }

    util::AnomalyLog::handle_anomalies();

}
