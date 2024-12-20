#include <gtest/gtest.h>
#include <iceicle/geometry/hypercube.hpp>

TEST(test_hypercube, test_nfacets){
    using namespace iceicle::shapes;

    // Point 
    ASSERT_EQ((hypercube<0>::n_facets(0)), 1);

    // Segment
    ASSERT_EQ((hypercube<1>::n_facets(0)), 2);
    ASSERT_EQ((hypercube<1>::n_facets(1)), 1);

    // square
    ASSERT_EQ((hypercube<2>::n_facets(0)), 4);
    ASSERT_EQ((hypercube<2>::n_facets(1)), 4);
    ASSERT_EQ((hypercube<2>::n_facets(2)), 1);

    // cube
    ASSERT_EQ((hypercube<3>::n_facets(0)), 8);
    ASSERT_EQ((hypercube<3>::n_facets(1)), 12);
    ASSERT_EQ((hypercube<3>::n_facets(2)), 6);
    ASSERT_EQ((hypercube<3>::n_facets(3)), 1);


}
