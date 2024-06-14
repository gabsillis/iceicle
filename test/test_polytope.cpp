#include "gtest/gtest.h"
#include "iceicle/transformations/polytope_transformations.hpp"

TEST(test_polytope, test_vertices){
    using namespace iceicle;
    using namespace polytope;

    // === 1D elements ===

    static constexpr bitset<1> segment_t{"0"};
    static constexpr tcode<1> segmentb_t{"1"};

    ASSERT_EQ(1, get_ndim(segment_t));
    ASSERT_EQ(2, n_vert(segment_t));
    ASSERT_EQ(2, n_vert(segmentb_t));

}
