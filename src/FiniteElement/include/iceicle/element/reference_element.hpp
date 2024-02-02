/**
 * @brief reference element stores things that are common between FiniteElements
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/basis/lagrange.hpp"
#include "iceicle/fe_enums.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include "iceicle/quadrature/SimplexQuadrature.hpp"
#include "iceicle/quadrature/HypercubeGaussLegendre.hpp"
#include "iceicle/quadrature/quadrules_1d.hpp"
#include <iceicle/element/finite_element.hpp>
#include <iceicle/element/TraceSpace.hpp>
#include <iceicle/tmp_utils.hpp>

#include <Numtool/tmp_flow_control.hpp>
#include <memory>

namespace FE {
    namespace FESPACE_ENUMS {

        enum FESPACE_BASIS_TYPE {
            /// Lagrange polynomials
            LAGRANGE = 0, 
            N_BASIS_TYPES,
        };

        enum FESPACE_QUADRATURE {
            /// Gauss Legendre Quadrature rules and tensor extensions thereof
            /// uses Grundmann Moller for Simplex type elements
            GAUSS_LEGENDRE,
            N_QUADRATURE_TYPES
        };
    }
}

namespace ELEMENT {

    template<typename T, typename IDX, int ndim>
    class ReferenceElement {
        using FEEvalType = ELEMENT::FEEvaluation<T, IDX, ndim>;
        using BasisType = BASIS::Basis<T, ndim>;
        using QuadratureType = QUADRATURE::QuadratureRule<T, IDX, ndim>;
        using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;

        public:
        std::unique_ptr<BasisType> basis;
        std::unique_ptr<QuadratureType> quadrule;
        FEEvalType eval;

        ReferenceElement() = default;

        template<int basis_order>
        ReferenceElement(
            const GeoElementType *geo_el,
            FE::FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
            FE::FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            ICEICLE::TMP::compile_int<basis_order> basis_order_arg
        ) {
            using namespace FE;
            using namespace FESPACE_ENUMS;
            switch(geo_el->domain_type()){
                case FE::HYPERCUBE: {
                    // construct the basis 
                    switch(basis_type){
                        case LAGRANGE:
                            basis = std::make_unique<BASIS::HypercubeLagrangeBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        default:
                            break;
                    }

                    // construct the quadrature rule
                    // TODO: change quadrature order based on high order geo elements
                    auto el_order_dispatch = [&]<int geo_order>{
                        switch(quadrature_type){
                            case FE::FESPACE_ENUMS::GAUSS_LEGENDRE:
                                quadrule = std::make_unique<QUADRATURE::HypercubeGaussLegendre<T, IDX, ndim, (geo_order+1)+(basis_order+1)>>();
                                break;
                            default:
                                break;
                        }
                        return 0;
                    };
                    NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, ELEMENT::MAX_DYNAMIC_ORDER>{},
                        geo_el->geometry_order(),
                        el_order_dispatch
                    );

                    // construct the evaluation
                    eval = FEEvalType(basis.get(), quadrule.get());

                    break;
                }
                case FE::SIMPLEX:
                    // construct the basis 
                    switch(basis_type){
                        case LAGRANGE:
                            basis = std::make_unique<BASIS::SimplexLagrangeBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        default:
                            break;
                    }

                    // construct the quadrature rule
                    switch(quadrature_type){
                        case FE::FESPACE_ENUMS::GAUSS_LEGENDRE:
                            quadrule = std::make_unique<QUADRATURE::GrundmannMollerSimplexQuadrature<T, IDX, ndim, basis_order+1>>();
                            break;
                        default:
                            break;
                    }

                    // construct the evaluation
                    eval = FEEvalType(basis.get(), quadrule.get());

                    break;
                default: 
                    break;
            }
        }
    };

    template<typename T, typename IDX, int ndim>
    class ReferenceTraceSpace {
        using Evals = ELEMENT::TraceEvaluation<T, IDX, ndim>;
        using Quadrature = QUADRATURE::QuadratureRule<T, IDX, ndim - 1>;
        using FaceType = ELEMENT::Face<T, IDX, ndim>;

        public:
        std::unique_ptr<Quadrature> quadrule;
        Evals eval;

        ReferenceTraceSpace() = default;

        template<int basis_order, int geo_order>
        ReferenceTraceSpace(
            const FaceType *fac,
            FE::FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            std::integral_constant<int, basis_order> b_order,
            std::integral_constant<int, geo_order> g_order
        ) : eval{} {
            using namespace FE;
            using namespace FE::FESPACE_ENUMS;

            switch(fac->domain_type()){
                case HYPERCUBE:

                    switch(quadrature_type){
                        case GAUSS_LEGENDRE:
                            quadrule = std::make_unique<
                                QUADRATURE::HypercubeGaussLegendre<
                                    T, IDX, ndim - 1,
                                    (geo_order+1)+(basis_order+1)
                                >
                            >();
                            break;
                        default:
                            break;
                    }
                    break;

                default:
                    break;
            }

        }
    };
}
