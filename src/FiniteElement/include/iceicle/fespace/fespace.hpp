/**
 * @brief A finite element space represents a collection of finite elements 
 * and trace spaces (if applicable)
 * that provides a general interface to a finite element discretization of the domain
 * and simple generation utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "iceicle/basis/basis.hpp"
#include "iceicle/crs.hpp"
#include "iceicle/element/finite_element.hpp"
#include <iceicle/element/reference_element.hpp>
#include "iceicle/fe_enums.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/tmp_utils.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <map>
#include <type_traits>

namespace FE {
    /**
     * Key to define the surjective mapping from an element 
     * to the corresponding evaluation
     */
    struct FETypeKey {
        int basis_order;

        int geometry_order;

        FE::DOMAIN_TYPE domain_type;

        FESPACE_ENUMS::FESPACE_QUADRATURE qtype;

        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype;


        friend bool operator<(const FETypeKey &l, const FETypeKey &r){
            using namespace FESPACE_ENUMS;
            if(l.qtype != r.qtype){
                return (int) l.qtype < (int) r.qtype;
            } else if(l.btype != r.btype){
                return (int) l.btype < (int) r.btype;
            } else if(l.domain_type != r.domain_type) {
                return (int) l.domain_type < (int) r.domain_type;
            } else if (l.geometry_order != r.geometry_order){
                return l.geometry_order < r.geometry_order;
            } else if( l.basis_order != r.basis_order) {
                return l.basis_order < r.basis_order;
            } else {
                // they are equal so less than is false for both cases
                return false;
            }
        }
    };

    /**
     * key to define surjective mapping from trace space to 
     * corresponding evaluation 
     */
    struct TraceTypeKey {

        int basis_order;

        int geometry_order;

        FE::DOMAIN_TYPE domain_type;

        FESPACE_ENUMS::FESPACE_QUADRATURE qtype;

        friend bool operator<(const TraceTypeKey &l, const TraceTypeKey &r){
            using namespace FESPACE_ENUMS;
           
            if(l.qtype != r.qtype){
                return (int) l.qtype < (int) r.qtype;
            } else if(l.domain_type != r.domain_type) {
                return (int) l.domain_type < (int) r.domain_type;
            } else if (l.geometry_order != r.geometry_order){
                return l.geometry_order < r.geometry_order;
            } else if( l.basis_order != r.basis_order) {
                return l.basis_order < r.basis_order;
            } else {
                // they are equal so less than is false for both cases
                return false;
            }
        }
    };

    /**
     * @brief Collection of FiniteElements and TraceSpaces
     * to form a unified interface for a discretization of a domain 
     *
     * @tparam T the numeric type 
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim>
    class FESpace {
        public:

        using ElementType = ELEMENT::FiniteElement<T, IDX, ndim>;
        using TraceType = ELEMENT::TraceSpace<T, IDX, ndim>;
        using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;
        using GeoFaceType = ELEMENT::Face<T, IDX, ndim>;
        using MeshType = MESH::AbstractMesh<T, IDX, ndim>;
        using BasisType = BASIS::Basis<T, ndim>;
        using QuadratureType = QUADRATURE::QuadratureRule<T, IDX, ndim>;

        /// @brief pointer to the mesh used
        MeshType *meshptr;

        /// @brief Array of finite elements in the space
        std::vector<ElementType> elements;

        /// @brief Array of trace spaces in the space 
        std::vector<TraceType> traces;

        /// @brief the start index of the interior traces 
        std::size_t interior_trace_start;
        /// @brief the end index of the interior traces (exclusive) 
        std::size_t interior_trace_end;

        /// @brief the start index of the boundary traces 
        std::size_t bdy_trace_start;
        /// @brief the end index of the boundary traces (exclusive)
        std::size_t bdy_trace_end;

        /** @brief maps local dofs to global dofs for dg space */
        dg_dof_map<IDX> dg_map;

        /** @brief the mapping of faces connected to each node */
        ICEICLE::UTIL::crs<IDX> fac_surr_nodes;

        private:

        // ========================================
        // = Maps to Basis, Quadrature, and Evals =
        // ========================================

        using ReferenceElementType = ELEMENT::ReferenceElement<T, IDX, ndim>;
        using ReferenceTraceType = ELEMENT::ReferenceTraceSpace<T, IDX, ndim>;
        std::map<FETypeKey, ReferenceElementType> ref_el_map;
        std::map<TraceTypeKey, ReferenceTraceType> ref_trace_map;

        public:

        // default constructor
        FESpace() = default;

        // delete copy semantics
        FESpace(const FESpace &other) = delete;
        FESpace<T, IDX, ndim>& operator=(const FESpace &other) = delete;

        // keep move semantics
        FESpace(FESpace &&other) = default;
        FESpace<T, IDX, ndim>& operator=(FESpace &&other) = default;

        /**
         * @brief construct an FESpace with uniform 
         * quadrature rules, and basis functions over all elements 
         *
         * @tparam basis_order the polynomial order of 1D basis functions
         *
         * @param meshptr pointer to the mesh 
         * @param basis_type enumeration of what basis to use 
         * @param quadrature_type enumeration of what quadrature rule to use 
         * @param basis_order_arg for template argument deduction of the basis order
         */
        template<int basis_order>
        FESpace(
            MeshType *meshptr,
            FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
            FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            ICEICLE::TMP::compile_int<basis_order> basis_order_arg
        ) : meshptr(meshptr), elements{} {

            // Generate the Finite Elements
            elements.reserve(meshptr->elements.size());
            for(const GeoElementType *geo_el : meshptr->elements){
                // create the Element Domain type key
                FETypeKey fe_key = {
                    .basis_order = basis_order,
                    .geometry_order = geo_el->geometry_order(),
                    .domain_type = geo_el->domain_type(),
                    .qtype = quadrature_type,
                    .btype = basis_type
                };

                // check if an evaluation doesn't exist yet
                if(ref_el_map.find(fe_key) == ref_el_map.end()){
                    ref_el_map[fe_key] = ReferenceElementType(geo_el, basis_type, quadrature_type, basis_order_arg);
                }
                ReferenceElementType &ref_el = ref_el_map[fe_key];
               
                // create the finite element
                ElementType fe(
                    geo_el,
                    ref_el.basis.get(),
                    ref_el.quadrule.get(),
                    &(ref_el.eval),
                    elements.size() // this will be the index of the new element
                );

                // add to the elements list
                elements.push_back(fe);
            }

            // Generate the Trace Spaces
            traces.reserve(meshptr->faces.size());
            for(const GeoFaceType *fac : meshptr->faces){
                // NOTE: assuming element indexing is the same as the mesh still

                bool is_interior = fac->bctype == ELEMENT::INTERIOR;
                ElementType &elL = elements[fac->elemL];
                ElementType &elR = (is_interior) ? elements[fac->elemR] : elements[fac->elemL];

                int geo_order = std::max(elL.geo_el->geometry_order(), elR.geo_el->geometry_order());

                auto geo_order_dispatch = [&]<int geo_order>() -> int{
                    TraceTypeKey trace_key = {
                        .basis_order = std::max(elL.basis->getPolynomialOrder(), elR.basis->getPolynomialOrder()), 
                        .geometry_order = geo_order,
                        .domain_type = fac->domain_type(),
                        .qtype = quadrature_type
                    };

                    if(ref_trace_map.find(trace_key) == ref_trace_map.end()){
                        ref_trace_map[trace_key] = ReferenceTraceType(fac,
                            basis_type, quadrature_type, 
                            std::integral_constant<int, basis_order>{},
                            std::integral_constant<int, geo_order>{});
                    }
                    ReferenceTraceType &ref_trace = ref_trace_map[trace_key];
                    
                    if(is_interior){
                        TraceType trace{ fac, &elL, &elR, ref_trace.trace_basis.get(),
                            ref_trace.quadrule.get(), &(ref_trace.eval), (IDX) traces.size() };
                        traces.push_back(trace);
                    } else {
                        TraceType trace = TraceType::make_bdy_trace_space(fac, &elL, ref_trace.trace_basis.get(), 
                            ref_trace.quadrule.get(), &(ref_trace.eval), (IDX) traces.size());
                        traces.push_back(trace);
                    }

                    return 0;
                };

                NUMTOOL::TMP::invoke_at_index(
                    NUMTOOL::TMP::make_range_sequence<int, 1, ELEMENT::MAX_DYNAMIC_ORDER>{},
                    geo_order,
                    geo_order_dispatch                    
                );
            }
            // reuse the face indexing from the mesh
            interior_trace_start = meshptr->interiorFaceStart;
            interior_trace_end = meshptr->interiorFaceEnd;
            bdy_trace_start = meshptr->bdyFaceStart;
            bdy_trace_end = meshptr->bdyFaceEnd;

            // generate the dof offsets 
            dg_map = dg_dof_map{elements};

            // generate the face surrounding nodes connectivity matrix 
            std::vector<std::vector<IDX>> connectivity_ragged(meshptr->nodes.n_nodes());
            for(int itrace = 0; itrace < traces.size(); ++itrace){
                const TraceType& trace = traces[itrace];
                for(IDX inode : trace.face->nodes_span()){
                    connectivity_ragged[inode].push_back(itrace);
                }
            }
            fac_surr_nodes = ICEICLE::UTIL::crs{connectivity_ragged};
        } 

        /**
         * @brief get the number of dg degrees of freedom in the entire fespace 
         * multiply this by the nummber of components to get the size requirement for 
         * a dg fespan or use the built_in function in the dg_map member
         * @return the number of dg degrees of freedom
         */
        constexpr std::size_t ndof_dg() const noexcept
        {
            return dg_map.calculate_size_requirement(1);
        }

        /**
         * @brief get the span that is the subset of the trace space list 
         * that only includes interior traces 
         * @return span over the interior traces 
         */
        std::span<TraceType> get_interior_traces(){
            return std::span<TraceType>{traces.begin() + interior_trace_start,
                traces.begin() + interior_trace_end};
        }

        /**
         * @brief get the span that is the subset of the trace space list 
         * that only includes boundary traces 
         * @return span over the boundary traces 
         */
        std::span<TraceType> get_boundary_traces(){
            return std::span<TraceType>{traces.begin() + bdy_trace_start,
                traces.begin() + bdy_trace_end};
        }

    };
    
}
