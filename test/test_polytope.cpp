#include "gtest/gtest.h"
#include "iceicle/transformations/polytope_transformations.hpp"

using namespace iceicle;
using namespace polytope;

//  === Element Topologies ===
static constexpr bitset<1> segment_t{"0"};
static constexpr tcode<1> segmentb_t{"1"};

static constexpr tcode<2> tri_a_t("00");
static constexpr tcode<2> tri_b_t("01");
static constexpr tcode<2> quad_a_t("10");
static constexpr tcode<2> quad_b_t("11");

static constexpr tcode<3> tet_a_t{"000"};
static constexpr tcode<3> tet_b_t{"001"};
static constexpr tcode<3> pyra_a_t{"010"};
static constexpr tcode<3> pyra_b_t{"011"};
static constexpr tcode<3> prism_a_t{"100"};
static constexpr tcode<3> prism_b_t{"101"};
static constexpr tcode<3> hexa_a_t{"110"};
static constexpr tcode<3> hexa_b_t{"111"};

TEST(test_polytope, test_vertices){

    // === 1D elements ===

    ASSERT_EQ(1, get_ndim(segment_t));
    ASSERT_EQ(2, n_vert(segment_t));
    ASSERT_EQ(2, n_vert(segmentb_t));

    {
        auto expected_vlist = std::array{
            bitset<1>{"0"},
            bitset<1>{"1"}
        };
        ASSERT_EQ(gen_vert<segment_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<segmentb_t>(), expected_vlist );
    }

    // ==== 2D elements ===
    { // triangles
        auto expected_vlist = std::array{
            vcode<2>{"00"},
            vcode<2>{"01"},
            vcode<2>{"10"}
        };
        ASSERT_EQ(gen_vert<tri_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<tri_b_t>(), expected_vlist );
    }

    { // quads
        auto expected_vlist = std::array{
            vcode<2>{"00"},
            vcode<2>{"01"},
            vcode<2>{"10"},
            vcode<2>{"11"}
        };
        ASSERT_EQ(gen_vert<quad_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<quad_b_t>(), expected_vlist );
    }

    // === 3D elements ===
    { // tetrahedron
        auto expected_vlist = std::array{
            vcode<3>{"000"},
            vcode<3>{"001"},
            vcode<3>{"010"},
            vcode<3>{"100"}
        };
        ASSERT_EQ(gen_vert<tet_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<tet_b_t>(), expected_vlist );
    }

    { // pyramid 
        auto expected_vlist = std::array{
            vcode<3>{"000"},
            vcode<3>{"001"},
            vcode<3>{"010"},
            vcode<3>{"011"},
            vcode<3>{"100"}
        };
        ASSERT_EQ(gen_vert<pyra_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<pyra_b_t>(), expected_vlist );
    }

    { // triangle prism
        auto expected_vlist = std::array{
            vcode<3>{"000"},
            vcode<3>{"001"},
            vcode<3>{"010"},
            vcode<3>{"100"},
            vcode<3>{"101"},
            vcode<3>{"110"}
        };
        ASSERT_EQ(gen_vert<prism_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<prism_b_t>(), expected_vlist );
    }

    { // hexahedron
        auto expected_vlist = std::array{
            vcode<3>{"000"},
            vcode<3>{"001"},
            vcode<3>{"010"},
            vcode<3>{"011"},
            vcode<3>{"100"},
            vcode<3>{"101"},
            vcode<3>{"110"},
            vcode<3>{"111"}
        };
        ASSERT_EQ(gen_vert<hexa_a_t>(), expected_vlist );
        ASSERT_EQ(gen_vert<hexa_b_t>(), expected_vlist );
    }
}

TEST(test_polytope, test_extrusion_parities) {
    {
        ecode<3> e{"011"};
        vcode<3> v{"100"};
        ASSERT_TRUE(extrusion_parity(e, v));
    }

    {
        ecode<3> e{"011"};
        vcode<3> v{"110"};
        ASSERT_FALSE(extrusion_parity(e, v));
    }

    // test the notion of ccw = outward normal
    { // consider: x = 0 face of unit cube

        // if we choose v = 000 and extrude y then z
        // right hand rule tells us the normal points into the domain
        ecode<3> e{"110"};
        vcode<3> v1{"000"};

        ASSERT_NE(
            extrusion_parity(~e, v1), // dual extrusion 
            hodge_extrusion_parity(e, v1)
        );

        // if we choose v = 010 or v = 100 and extrude y then z
        // right hand rule tells use the normal points out of the domain (ccw -> outward normal)
        vcode<3> v2{"010"};
        vcode<3> v3{"100"};
        ASSERT_EQ(
            extrusion_parity(~e, v2), // dual extrusion 
            hodge_extrusion_parity(e, v2)
        );
        ASSERT_EQ(
            extrusion_parity(~e, v3), // dual extrusion 
            hodge_extrusion_parity(e, v3)
        );

    }
}

TEST(test_polytope, test_node_count){
    ASSERT_EQ(get_n_node(segment_t, full_extrusion<1>, 2), 2);
    ASSERT_EQ(get_n_node(segment_t, full_extrusion<1>, 5), 5);
    ASSERT_EQ(get_n_node(segmentb_t, full_extrusion<1>, 2), 2);
    ASSERT_EQ(get_n_node(segmentb_t, full_extrusion<1>, 5), 5);


    ASSERT_EQ(get_n_node(tri_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(tri_a_t, full_extrusion<2>, 4), 10);
    ASSERT_EQ(get_n_node(quad_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(quad_a_t, full_extrusion<2>, 4), 16);
    ASSERT_EQ(get_n_node(tri_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(tri_b_t, full_extrusion<2>, 4), 10);
    ASSERT_EQ(get_n_node(quad_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(quad_b_t, full_extrusion<2>, 4), 16);


    ASSERT_EQ(get_n_node(tet_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(tet_a_t, full_extrusion<2>, 4), 10+6+3+1);
    ASSERT_EQ(get_n_node(pyra_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(pyra_a_t, full_extrusion<2>, 4), 16+9+4+1);
    ASSERT_EQ(get_n_node(prism_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(prism_a_t, full_extrusion<2>, 4), 40);
    ASSERT_EQ(get_n_node(hexa_a_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(hexa_a_t, full_extrusion<2>, 4), 64);
    ASSERT_EQ(get_n_node(tet_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(tet_b_t, full_extrusion<2>, 4), 10+6+3+1);
    ASSERT_EQ(get_n_node(pyra_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(pyra_b_t, full_extrusion<2>, 4), 16+9+4+1);
    ASSERT_EQ(get_n_node(prism_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(prism_b_t, full_extrusion<2>, 4), 40);
    ASSERT_EQ(get_n_node(hexa_b_t, full_extrusion<2>, 1), 1);
    ASSERT_EQ(get_n_node(hexa_b_t, full_extrusion<2>, 4), 64);
}

TEST(test_polytope, test_facet_nodes){


    {
        static constexpr tcode<3> t{"010"};
        static constexpr ecode<3> e{"101"};
        static constexpr vcode<3> v{"011"};
        constexpr auto nodes {facet_nodes<double, t, e, v, 3>()};
        ASSERT_DOUBLE_EQ(nodes[0][0], 1.0);
        ASSERT_DOUBLE_EQ(nodes[0][1], 1.0);
        ASSERT_DOUBLE_EQ(nodes[0][2], 0.0);

        ASSERT_DOUBLE_EQ(nodes[1][0], 0.5);
        ASSERT_DOUBLE_EQ(nodes[1][1], 1.0);
        ASSERT_DOUBLE_EQ(nodes[1][2], 0.0);

        ASSERT_DOUBLE_EQ(nodes[2][0], 0.0);
        ASSERT_DOUBLE_EQ(nodes[2][1], 1.0);
        ASSERT_DOUBLE_EQ(nodes[2][2], 0.0);

        ASSERT_DOUBLE_EQ(nodes[3][0], 0.5);
        ASSERT_DOUBLE_EQ(nodes[3][1], 0.5);
        ASSERT_DOUBLE_EQ(nodes[3][2], 0.5);

        ASSERT_DOUBLE_EQ(nodes[4][0], 0.0);
        ASSERT_DOUBLE_EQ(nodes[4][1], 0.5);
        ASSERT_DOUBLE_EQ(nodes[4][2], 0.5);

        ASSERT_DOUBLE_EQ(nodes[5][0], 0.0);
        ASSERT_DOUBLE_EQ(nodes[5][1], 0.0);
        ASSERT_DOUBLE_EQ(nodes[5][2], 1.0);
    }
}
