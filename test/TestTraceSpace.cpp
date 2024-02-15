#include "iceicle/basis/lagrange.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/quadrature/HypercubeGaussLegendre.hpp"
#include <iceicle/element/TraceSpace.hpp>


#include <gtest/gtest.h>

TEST(test_trace_space, test_basis_eval){
    // compile-time test parameters
    static constexpr int ndim = 2;
    static constexpr int PnL = 1;
    static constexpr int PnR = 2;

    // type aliases
    using T = double;
    using IDX = int;

    using BasisTypeL = BASIS::HypercubeLagrangeBasis<T, IDX, ndim, PnL>;
    using BasisTypeR = BASIS::HypercubeLagrangeBasis<T, IDX, ndim, PnR>;

    using QuadTypeL = QUADRATURE::HypercubeGaussLegendre<T, IDX, ndim, PnL>;
    using QuadTypeR = QUADRATURE::HypercubeGaussLegendre<T, IDX, ndim, PnR>;
    using TraceQuadrature = QUADRATURE::HypercubeGaussLegendre<T, IDX, ndim - 1, (PnL > PnR) ? PnL : PnR>;

    using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;
    using GeoFaceType = ELEMENT::Face<T, IDX, ndim>;

    using FiniteElement = ELEMENT::FiniteElement<T, IDX, ndim>;
    using TraceSpace = ELEMENT::TraceSpace<T, IDX, ndim>;

    // Use the Uniform Mesh API to generate a two element mesh with one 
    // internal face
    MESH::AbstractMesh<T, IDX, ndim> mesh{
        {-1.0, -1.0},
        {1.0, 1.0}, 
        {2, 1}
    };

    // create the left element 
    BasisTypeL basisL{};
    QuadTypeL quadratureL{};
    ELEMENT::FEEvaluation<T, IDX, ndim> evalsL{&basisL, &quadratureL};
    FiniteElement elL {
        mesh.elements[0],
        &basisL,
        &quadratureL,
        &evalsL,
        0
    };

    // create the right element 
    BasisTypeR basisR{};
    QuadTypeR quadratureR{};
    ELEMENT::FEEvaluation<T, IDX, ndim> evalsR{&basisR, &quadratureR};
    FiniteElement elR {
        mesh.elements[1],
        &basisR,
        &quadratureR,
        &evalsR,
        1
    };

    // create the Trace Space 
    TraceQuadrature trace_quadrule{};
    ELEMENT::TraceEvaluation<T, IDX, ndim> trace_eval{};
    TraceSpace trace {
        mesh.faces[mesh.interiorFaceStart],
        &elL,
        &elR,
        &trace_quadrule,
        &trace_eval,
        mesh.interiorFaceStart
    };

    // Test the number of basis functions 
    ASSERT_EQ(trace.nbasisL(), std::pow(PnL + 1, ndim));
    ASSERT_EQ(trace.nbasisR(), std::pow(PnR + 1, ndim));

    MATH::GEOMETRY::Point<T, ndim - 1> s = {0.0};

    std::vector<double> BiL(trace.nbasisL());
    std::vector<double> BiR(trace.nbasisR());

    trace.evalBasisL(s, BiL.data());
    trace.evalBasisR(s, BiR.data());

    ASSERT_DOUBLE_EQ(BiL[0], 0.0);
    ASSERT_DOUBLE_EQ(BiL[1], 0.0);
    ASSERT_DOUBLE_EQ(BiL[2], 0.5);
    ASSERT_DOUBLE_EQ(BiL[3], 0.5);
    
}
