#include "iceicle/fe_utils.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include "iceicle/basis/lagrange.hpp"
#include "iceicle/quadrature/HypercubeGaussLegendre.hpp"
#include "iceicle/element/finite_element.hpp"
#include <gtest/gtest.h>
#include <iceicle/element/evaluation.hpp>
#include <iceicle/disc/projection.hpp>

using namespace iceicle;

TEST(test_evaluation, test_2d_quad) {

    // definitions
    static constexpr int ndim = 2;
    using Point = MATH::GEOMETRY::Point<double, ndim>;

    // nodes
    std::vector<int> inodes{1, 2, 3, 4};
    std::vector<Point> coord{
        { 0.0,  0.0},
        {-0.1,  1.2},
        { 1.1,  0.1},
        { 1.3,  0.9}
    };

    // transformation
    int geo_order = 1;
    auto trans_ptr = transformation_table<double, int, ndim>
        .get_transform(DOMAIN_TYPE::HYPERCUBE, geo_order);

    // basis
    static constexpr int order = 1;
    HypercubeLagrangeBasis<double, int, ndim, order> basis{}; 

    // quadrature
    HypercubeGaussLegendre<double, int, ndim, order + 1> quadrule{};

    // evaluation 
    auto evals = quadrature_point_evaluations(basis, quadrule);

    int elidx = 0;

    FiniteElement<double, int, ndim> el{trans_ptr, &basis, &quadrule, evals, inodes, coord, elidx};

    for(int k = 0; k < 1000; ++k){
        // make the basis evaluation at a random point
        auto domain_pt = random_domain_point(trans_ptr);
        BasisEvaluation eval{basis, domain_pt};

        double xi = domain_pt[0];
        double eta = domain_pt[1];
        SCOPED_TRACE("Reference Domain Point: (" + std::to_string(xi) + ", " + std::to_string(eta) + ")");

        ASSERT_NEAR(eval.bi_span[0], (xi - 1) * (eta - 1) / 4 , std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(eval.bi_span[1], -(xi - 1) * (eta + 1) / 4, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(eval.bi_span[2], -(xi + 1) * (eta - 1) / 4, std::numeric_limits<double>::epsilon());
        ASSERT_NEAR(eval.bi_span[3], (xi + 1) * (eta + 1) / 4 , std::numeric_limits<double>::epsilon());
    }
}
