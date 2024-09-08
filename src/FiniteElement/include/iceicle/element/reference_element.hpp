/**
 * @brief reference element stores things that are common between FiniteElements
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/basis/lagrange.hpp"
#include "iceicle/basis/legendre.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include "iceicle/quadrature/HypercubeGaussLegendre.hpp"
#include "iceicle/quadrature/SimplexQuadrature.hpp"
#include <iceicle/element/finite_element.hpp>
#include <iceicle/element/TraceSpace.hpp>
#include <iceicle/tmp_utils.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <memory>

namespace iceicle {
    namespace FESPACE_ENUMS {

        enum FESPACE_BASIS_TYPE {
            /// Lagrange polynomials
            LAGRANGE = 0, 
            LEGENDRE = 1,
            N_BASIS_TYPES,
        };

        enum FESPACE_QUADRATURE {
            /// Gauss Legendre Quadrature rules and tensor extensions thereof
            /// uses Grundmann Moller for Simplex type elements
            GAUSS_LEGENDRE,
            N_QUADRATURE_TYPES
        };
    }

    template<typename T, typename IDX, int ndim>
    class ReferenceElement {
        using FEEvalType = FEEvaluation<T, IDX, ndim>;
        using BasisType = Basis<T, ndim>;
        using QuadratureType = QuadratureRule<T, IDX, ndim>;
        using GeoElementType = GeometricElement<T, IDX, ndim>;

        public:
        std::unique_ptr<BasisType> basis;
        std::unique_ptr<QuadratureType> quadrule;
        FEEvalType eval;

        ReferenceElement() = default;

        template<int basis_order>
        ReferenceElement(
            DOMAIN_TYPE domain_type,
            int geometry_order,
            FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
            FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            tmp::compile_int<basis_order> basis_order_arg
        ) {
            using namespace FESPACE_ENUMS;
            switch(domain_type){
                case DOMAIN_TYPE::HYPERCUBE: {
                    // construct the basis 
                    switch(basis_type){
                        case LAGRANGE:
                            basis = std::make_unique<HypercubeLagrangeBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        case LEGENDRE:
                            basis = std::make_unique<HypercubeLegendreBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        default:
                            break;
                    }

                    // construct the quadrature rule
                    auto el_order_dispatch = [&]<int geo_order>{
                        switch(quadrature_type){
                            // number of quadrature points in 1D
                            static constexpr int nqp = geo_order + basis_order;
                            case FESPACE_ENUMS::GAUSS_LEGENDRE:
                                quadrule = std::make_unique<HypercubeGaussLegendre<T, IDX, ndim, nqp>>();
                                break;
                            default:
                                break;
                        }
                        return 0;
                    };
                    NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                        geometry_order,
                        el_order_dispatch
                    );

                    // construct the evaluation
                    eval = FEEvalType(basis.get(), quadrule.get());

                    break;
                }
                case DOMAIN_TYPE::SIMPLEX: {
                    // construct the basis 
                    switch(basis_type){
                        case LAGRANGE:
                            basis = std::make_unique<SimplexLagrangeBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        case LEGENDRE:
                            basis = std::make_unique<SimplexLagrangeBasis<
                                T, IDX, ndim, basis_order>>();
                            break;
                        default:
                            break;
                    }

                    // construct the quadrature rule
                    auto el_order_dispatch = [&]<int geo_order>{
                        switch(quadrature_type){
                            // number of quadrature points in 1D
                            static constexpr int nqp = geo_order + basis_order;
                            case FESPACE_ENUMS::GAUSS_LEGENDRE:
                                quadrule = std::make_unique<GrundmannMollerSimplexQuadrature<T, IDX, ndim, nqp>>();
                                break;
                            default:
                                break;
                        }
                        return 0;
                    };
                    NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                        geometry_order,
                        el_order_dispatch
                    );

                    // construct the evaluation
                    eval = FEEvalType(basis.get(), quadrule.get());

                    break;
                }
                default: 
                    break;
            }
        }

        /// @brief construct an isoparametric CG element
        ReferenceElement(
            DOMAIN_TYPE domain_type,
            int geometry_order
        ) {
            switch (domain_type) {
                case DOMAIN_TYPE::HYPERCUBE:
                {
                    auto el_order_dispatch = [&]<int geo_order>{
                        static constexpr int nqp = geo_order + 1;
                        quadrule = std::make_unique<HypercubeGaussLegendre<T, IDX, ndim, nqp>>();
                        basis = std::make_unique<HypercubeLagrangeBasis<T, IDX, ndim, geo_order>>();
                        return 0;
                    };
                    NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                        geometry_order, 
                        el_order_dispatch
                    );

                    break;
                }
                case DOMAIN_TYPE::SIMPLEX:
                {
                    auto el_order_dispatch = [&]<int geo_order>{
                        static constexpr int nqp = geo_order + 1;
                        quadrule = std::make_unique<GrundmannMollerSimplexQuadrature<T, IDX, ndim, nqp>>();
                        basis = std::make_unique<SimplexLagrangeBasis<T, IDX, ndim, geo_order>>();
                        return 0;
                    };
                    NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                        geometry_order,
                        el_order_dispatch
                    );
                    break;
                }
                default:
                    break;
            }

            // construct the evaluation
            eval = FEEvalType(basis.get(), quadrule.get());
        }
    };

    template<typename T, typename IDX, int ndim>
    class ReferenceTraceSpace {
        using Evals = TraceEvaluation<T, IDX, ndim>;
        using TraceBasisType = Basis<T, ndim-1>;
        using Quadrature = QuadratureRule<T, IDX, ndim - 1>;
        using FaceType = Face<T, IDX, ndim>;

        public:
        std::unique_ptr<Quadrature> quadrule;
        std::unique_ptr<TraceBasisType> trace_basis;
        Evals eval;

        ReferenceTraceSpace() = default;

        template<int basis_order, int geo_order>
        ReferenceTraceSpace(
            const FaceType *fac,
            FESPACE_ENUMS::FESPACE_BASIS_TYPE btype,
            FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            std::integral_constant<int, basis_order> b_order,
            std::integral_constant<int, geo_order> g_order
        ) : eval{} {
            using namespace FESPACE_ENUMS;

            switch(fac->domain_type()){
                case DOMAIN_TYPE::HYPERCUBE:
                        
                    // NOTE: we want the trace basis to be in the space of the geometry 
                    // so we use lagrange polynomials with geometry order
                    trace_basis = std::make_unique<HypercubeLagrangeBasis<
                        T, IDX, ndim - 1, geo_order>>();
                    

                    switch(quadrature_type){
                        case GAUSS_LEGENDRE:
                            quadrule = std::make_unique<
                                HypercubeGaussLegendre<
                                    T, IDX, ndim - 1,
                                    (geo_order)+(basis_order)
                                >
                            >();
                            break;
                        default:
                            break;
                    }
                    break;

                case DOMAIN_TYPE::SIMPLEX:
                        
                    // NOTE: we want the trace basis to be in the space of the geometry 
                    // so we use lagrange polynomials with geometry order
                    trace_basis = std::make_unique<SimplexLagrangeBasis<
                        T, IDX, ndim - 1, geo_order>>();

                    switch(quadrature_type){
                        case GAUSS_LEGENDRE:
                            quadrule = std::make_unique<
                                GrundmannMollerSimplexQuadrature<
                                    T, IDX, ndim - 1,
                                    (geo_order)+(basis_order)
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
