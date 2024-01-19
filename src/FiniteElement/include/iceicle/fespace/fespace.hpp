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
#include "iceicle/element/finite_element.hpp"
#include <iceicle/element/reference_element.hpp>
#include "iceicle/fe_enums.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/layout_enums.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/tmp_utils.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <map>

namespace FE {
    /**
     * Key to define the surjective mapping from an element 
     * to the corresponding evaluation
     */
    struct FETypeKey {
        int basis_order;

        FE::DOMAIN_TYPE domain_type;

        FESPACE_ENUMS::FESPACE_QUADRATURE qtype;

        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype;

        friend bool operator<(const FETypeKey &l, const FETypeKey &r){
            using namespace FESPACE_ENUMS;
            int ileft = 
                (int) N_BASIS_TYPES * (int) N_QUADRATURE_TYPES * (int) N_DOMAIN_TYPES * l.basis_order
                + (int) N_BASIS_TYPES * (int) N_DOMAIN_TYPES * (int) l.qtype 
                + (int) N_BASIS_TYPES * (int) l.domain_type
                + l.btype;
            int iright = 
                (int) N_BASIS_TYPES * (int) N_QUADRATURE_TYPES * (int) N_DOMAIN_TYPES * r.basis_order
                + (int) N_BASIS_TYPES * (int) N_DOMAIN_TYPES * (int) r.qtype 
                + (int) N_BASIS_TYPES * (int) r.domain_type
                + r.btype;
            return ileft < iright;
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
        using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;
        using MeshType = MESH::AbstractMesh<T, IDX, ndim>;
        using BasisType = BASIS::Basis<T, ndim>;
        using QuadratureType = QUADRATURE::QuadratureRule<T, IDX, ndim>;

        /// @brief pointer to the mesh used
        MeshType *meshptr;

        /// @brief ArrayList of finite elements in the space
        std::vector<ElementType> elements;

        /** @brief index offsets for dg degrees of freedom */
        dg_dof_offsets dg_offsets;

        private:

        // ========================================
        // = Maps to Basis, Quadrature, and Evals =
        // ========================================

        using ReferenceElementType = ELEMENT::ReferenceElement<T, IDX, ndim>;
        std::map<FETypeKey, ReferenceElementType> ref_el_map;

        public:

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

            // generate the dof offsets 
            dg_offsets = dg_dof_offsets(elements);
        } 

        /**
         * @brief generate an fespan to provide an index space for all 
         * vector components of all degrees of freedom that exist in a dg representation 
         * of the space 
         *
         * @tparam ncomp the number of vector components 
         * @tparam order the order of dofs and vector components in the memory layout 
         *         LEFT means it comes first in C style array indexing and thus is the slower index 
         * @param data the data array to create a view for 
         *        WARNING: it is the users responsibility to ensure that the size of the array 
         *        is large enough to encapsulate the entire index space 
         */
        template<int ncomp, LAYOUT_VECTOR_ORDER order = DOF_LEFT, class AccessorPolicy = FE::default_accessor<T>>
        constexpr FE::fespan<T, dg_layout<T, ncomp, order>, AccessorPolicy> 
        generate_dg_fespan(T *data) const 
        {
            return FE::fespan<T, dg_layout<T, ncomp, order>, AccessorPolicy>(dg_offsets);
        } 

        /**
         * @brief get the number of dg degrees of freedom in the entire fespace 
         * multiply this by the nummber of components to get the size requirement for 
         * a dg fespan or use the built_in function in the dg_offsets member
         * @return the number of dg degrees of freedom
         */
        constexpr std::size_t ndof_dg() const noexcept
        {
            return dg_offsets.calculate_size_requirement(1);
        }
    };
    
}
