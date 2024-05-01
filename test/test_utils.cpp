#include "gtest/gtest.h"
#include "iceicle/algo.hpp"

using namespace iceicle::util;

TEST(test_util, test_set_ops){
    std::vector<int> a1{1, 3,    5,    6, 7, 7};
    std::vector<int> b1{1, 3, 4, 5, 5, 6, 7,    9};
    ASSERT_TRUE(subset(a1, b1));
    ASSERT_FALSE(eqset(a1, b1));

    std::vector<int> a2{1, 2, 5};
    std::vector<int> b2{1, 2, 5};
    ASSERT_TRUE(subset(a2, b2));
    ASSERT_TRUE(eqset(a2, b2));

    std::vector<int> a3{ 2, 3 };
    std::vector<int> b3{1, 2, 3};
    ASSERT_TRUE(subset(a3, b3));
    ASSERT_FALSE(eqset(a3, b3));


    std::vector<double> a_double{1.0, 2.0, 3.0};
    std::vector<double> b_double{1.0, 2.0, 3.0};
    ASSERT_TRUE(subset(a_double, b_double));
    ASSERT_TRUE(eqset(a_double, b_double));

    std::vector<int> a4{1, 3};
    std::vector<int> b4{1, 2, 3, 4, 5};
    ASSERT_TRUE(subset(a4, b4));
    ASSERT_FALSE(eqset(a4, b4));

    std::vector<int> a5{1, 3};
    std::vector<int> b5{1, 2, 4, 5};
    ASSERT_FALSE(subset(a5, b5));
    ASSERT_FALSE(eqset(a5, b5));

    std::vector<int> a6{1, 2, 4, 5, 6};
    std::vector<int> b6{1, 2, 4, 5};
    ASSERT_FALSE(subset(a6, b6));
    ASSERT_FALSE(eqset(a6, b6));

}
