#include "iceicle/basis/lagrange.hpp"
#include "iceicle/geometry/hypercube_element.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/quadrature/HypercubeGaussLegendre.hpp>
#include <iceicle/fe_function/fespan.hpp>
#include <iceicle/fe_function/dglayout.hpp>
#include <iceicle/fe_function/layout_right.hpp>
#include <iceicle/element/finite_element.hpp>

#include <gtest/gtest.h>

TEST(test_fespan, test_dglayout){

    using T = double;
    using IDX = int;
    static constexpr int ndim = 4;
    static constexpr int Pn = 3;
    using GeoElement = ELEMENT::HypercubeElement<T, IDX, ndim, Pn>;
    using BasisType = BASIS::HypercubeLagrangeBasis<T, IDX, ndim, Pn>;
    using QuadratureType = QUADRATURE::HypercubeGaussLegendre<T, IDX, ndim, Pn>;
    using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;
    using EvaluationType = ELEMENT::FEEvaluation<T, IDX, ndim>;

    std::vector<FiniteElement> elements;

    // create necesarry items to make finite elements 
    BasisType basis{};
    QuadratureType quadrule{};
    EvaluationType evals(basis, quadrule);

    // create geometric elements 
    GeoElement gel1{};
    GeoElement gel2{};
    int inode = 0;
    for(int i = 0; i < gel1.n_nodes(); ++i, ++inode){
        gel1.setNode(i, inode);
    }
    for(int i = 0; i < gel2.n_nodes(); ++i, ++inode){
        gel2.setNode(i, inode);
    }

    // create the finite elements 
    FiniteElement el1{&gel1, &basis, &quadrule, &evals, 0};
    FiniteElement el2{&gel2, &basis, &quadrule, &evals, 1};
    elements.push_back(el1);
    elements.push_back(el2);

    // get the offsets
    FE::dg_dof_map offsets{elements};

    std::vector<T> data(offsets.calculate_size_requirement(2));
    std::iota(data.begin(), data.end(), 0.0);
    FE::fe_layout_right<IDX, decltype(offsets), 2> layout(offsets);
    // alternate layout syntax
    using namespace ICEICLE::TMP;
    FE::fe_layout_right layout2{offsets, to_size<2>{}};

    FE::fespan fespan1(data.data(), layout);

    static constexpr int ndof_per_elem = ndim * (Pn + 1);
   
    int neq = 2;
    ASSERT_EQ(neq * 2 + 1.0, (fespan1[0, 2, 1]));

    std::size_t iel = 1, idof = 2, iv = 0;
    ASSERT_EQ(iel * std::pow(ndim, (Pn + 1)) * neq + idof * neq + iv, (fespan1[iel, idof, iv]));


    auto local_layout = fespan1.create_element_layout(1);
    std::vector<T> el_memory(local_layout.size());
    FE::dofspan elspan1{el_memory.data(), local_layout};
    FE::extract_elspan(1, fespan1, elspan1);

    iel = 1;
    idof = 8;
    iv = 1;
    ASSERT_DOUBLE_EQ(iel * std::pow(ndim, (Pn + 1)) * neq + idof * neq + iv, (elspan1[idof, iv]));

    FE::scatter_elspan(1, 1.0, elspan1, 1.0, fespan1);
    ASSERT_DOUBLE_EQ(2.0 * (iel * std::pow(ndim, (Pn + 1)) * neq + idof * neq + iv), (fespan1[iel, idof, iv]));
    iel = 1;
    idof = 1;
    iv = 1;
    ASSERT_DOUBLE_EQ(2.0 *(iel * std::pow(ndim, (Pn + 1)) * neq + idof * neq + iv), (fespan1[iel, idof, iv]));
    iel = 0;
    idof = 8;
    iv = 0;
    // make sure original element data remains unchanged
    ASSERT_DOUBLE_EQ(iel * std::pow(ndim, (Pn + 1)) * neq + idof * neq + iv, (fespan1[iel, idof, iv]));

    //TODO: add more tests
}
