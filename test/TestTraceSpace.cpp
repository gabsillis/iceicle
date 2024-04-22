#include "iceicle/basis/lagrange.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/quadrature/HypercubeGaussLegendre.hpp"
#include <iceicle/element/TraceSpace.hpp>


#include <gtest/gtest.h>
using namespace iceicle;
TEST(test_trace_space, test_basis_eval){
    // compile-time test parameters
    static constexpr int ndim = 2;
    static constexpr int PnL = 1;
    static constexpr int PnR = 2;

    // type aliases
    using T = double;
    using IDX = int;

    using BasisTypeL = HypercubeLagrangeBasis<T, IDX, ndim, PnL>;
    using BasisTypeR = HypercubeLagrangeBasis<T, IDX, ndim, PnR>;
    using TraceBasisType = HypercubeLagrangeBasis<T, IDX, ndim - 1, PnR>;

    using QuadTypeL = HypercubeGaussLegendre<T, IDX, ndim, PnL>;
    using QuadTypeR = HypercubeGaussLegendre<T, IDX, ndim, PnR>;
    using TraceQuadrature = HypercubeGaussLegendre<T, IDX, ndim - 1, (PnL > PnR) ? PnL : PnR>;

    using GeoElementType = GeometricElement<T, IDX, ndim>;
    using GeoFaceType = Face<T, IDX, ndim>;

    using FiniteElement = FiniteElement<T, IDX, ndim>;
    using TraceSpace = TraceSpace<T, IDX, ndim>;

    // Use the Uniform Mesh API to generate a two element mesh with one 
    // internal face
    AbstractMesh<T, IDX, ndim> mesh{
        {-1.0, -1.0},
        {1.0, 1.0}, 
        {2, 1}
    };

    // create the left element 
    BasisTypeL basisL{};
    QuadTypeL quadratureL{};
    FEEvaluation<T, IDX, ndim> evalsL{&basisL, &quadratureL};
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
    FEEvaluation<T, IDX, ndim> evalsR{&basisR, &quadratureR};
    FiniteElement elR {
        mesh.elements[1],
        &basisR,
        &quadratureR,
        &evalsR,
        1
    };

    // create the Trace Space 
    TraceBasisType trace_basis{};
    TraceQuadrature trace_quadrule{};
    TraceEvaluation<T, IDX, ndim> trace_eval{};
    TraceSpace trace {
        mesh.faces[mesh.interiorFaceStart],
        &elL,
        &elR,
        &trace_basis,
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
