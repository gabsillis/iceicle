#include "gtest/gtest.h"
#include "iceicle/algo.hpp"
#include "iceicle/bitset.hpp"

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

TEST(test_util, test_bitset){

    using namespace iceicle;
   
    /// equality of zero size bitsets
    ASSERT_TRUE((bitset<0>() == bitset<0>(0b100)));

    // ==== Constructors ===

    // Test construction from ulong
    ASSERT_EQ( bitset<1>{0b1}._data, 0b1);
    ASSERT_EQ( bitset<5>{0b10110}._data, 0b10110);
    ASSERT_EQ( bitset<64>{0b10110}._data, 0b10110);
    ASSERT_EQ( bitset<3>{0b10110}._data, 0b110);

    // Test default construction
    ASSERT_EQ(bitset<64>{}._data, 0);

    // Test construction from string
    ASSERT_EQ(bitset<5>{"10110"}._data, 0b10110);
    ASSERT_EQ(bitset<3>{"10110"}._data, 0b110);

    // === Comparison ===
    ASSERT_EQ(bitset<5>{"10010"}, bitset<5>{0b10010});
    ASSERT_TRUE(bitset<5>{0b10000} > bitset<5>{0b01000});

    // === operator[] ===
    bitset<5> bits_a{0b10110};
    ASSERT_EQ(bits_a[0], 0);
    ASSERT_EQ(bits_a[1], 1);
    ASSERT_EQ(bits_a[2], 1);
    ASSERT_EQ(bits_a[3], 0);
    ASSERT_EQ(bits_a[4], 1);

    bits_a[1] = 0;
    ASSERT_EQ(bits_a[1], 0);

    bits_a[1] = bits_a[4];
    ASSERT_EQ(bits_a[1], 1);

    bits_a[1].flip();
    ASSERT_EQ(bits_a[1], 0);
    bits_a[1].flip();
    ASSERT_EQ(bits_a[1], 1);
    ASSERT_EQ(bits_a._data, 0b10110);

    // === test() ===
    ASSERT_THROW(bits_a.test(5), std::out_of_range);
    ASSERT_EQ(bits_a.test(1), true);

    // === all(), any(), none() ===
    ASSERT_FALSE(bitset<5>{"10011"}.all());
    ASSERT_TRUE (bitset<5>{"10011"}.any());
    ASSERT_FALSE(bitset<5>{"10011"}.none());

    ASSERT_TRUE (bitset<5>{"11111"}.all());
    ASSERT_TRUE (bitset<5>{"11111"}.any());
    ASSERT_FALSE(bitset<5>{"11111"}.none());

    ASSERT_FALSE(bitset<5>{"00000"}.all());
    ASSERT_FALSE(bitset<5>{"00000"}.any());
    ASSERT_TRUE (bitset<5>{"00000"}.none());

    // === count() ===
    ASSERT_EQ(bitset<6>{"000111"}.count(), 3);
    ASSERT_EQ(bitset<6>{"101011"}.count(), 4);
    ASSERT_EQ(bitset<6>{"111111"}.count(), 6);
    ASSERT_EQ(bitset<6>{}.set().count(), 6);

    // === &=, |=, and ^=
    
    bitset<6> bits_b{"010110"};
    bitset<6> bits_c{"011010"};

    bits_c &= bits_b;
    ASSERT_EQ(bits_c._data, 0b10110 & 0b011010);
    bits_c = bitset<6>{"011010"};

    bits_c |= bits_b;
    ASSERT_EQ(bits_c._data, 0b10110 | 0b011010);
    bits_c = bitset<6>{"011010"};

    bits_c ^= bits_b;
    ASSERT_EQ(bits_c._data, 0b10110 ^ 0b011010);
    bits_c = bitset<6>{"011010"};

    /// === conversions ===
    ASSERT_EQ(bitset<0>{}.to_string(), "");
    ASSERT_EQ(bitset<6>{"011010"}.to_string(), "011010");
    ASSERT_EQ(bitset<6>{"011010"}.to_ulong(), 0b011010);
    ASSERT_EQ(bitset<6>{"011010"}.to_ullong(), 0b011010);
}
