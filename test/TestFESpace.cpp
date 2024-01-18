#include "iceicle/element/reference_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/fespace/fespace.hpp>

#include <gtest/gtest.h>

TEST(test_fespace, test_element_construction){
    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 2;
    static constexpr int pn_basis = 3;

    // create a uniform mesh
    MESH::AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {2, 2}, pn_basis);

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, FE::FESPACE_ENUMS::LAGRANGE,
        FE::FESPACE_ENUMS::GAUSS_LEGENDRE, 
        ICEICLE::TMP::compile_int<pn_basis>()
    };

    ASSERT_EQ(fespace.elements.size(), 4);

    ASSERT_EQ(fespace.dg_offsets.calculate_size_requirement(2), 4 * 2 * std::pow(pn_basis + 1, pn_geo));
}
