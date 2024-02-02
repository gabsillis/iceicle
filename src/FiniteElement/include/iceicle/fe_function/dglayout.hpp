/**
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief dglayout represents the memory layout of
 * a DG representation of a vector-valued fe_function
 */
#pragma once
#include <iceicle/fe_function/layout_enums.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/basis/basis.hpp>
#include <iceicle/element/finite_element.hpp>
#include <iceicle/fe_function/el_layout.hpp>

#include <type_traits>
#include <algorithm>
namespace FE {

    /** 
     * @brief represents the offsets of the degrees of freedom for a set of elements 
     * NOTE: this is independent of vector components;
     */
    struct dg_dof_offsets {
        std::vector<std::size_t> offsets;

        /** @brief default constructor */
        constexpr dg_dof_offsets() noexcept : offsets{0} {}

        /** @brief construct with uniform basis function */
        template<typename T, typename IDX, int ndim>
        constexpr dg_dof_offsets(
            IDX nelem,
            const BASIS::Basis<T, ndim> &basis
        ) noexcept : offsets(nelem + 1) {
            for(int i = 0; i <= nelem; ++i){
                offsets[i] = i * basis.nbasis();
            }
        }

        /** @brief construct with a list of FiniteElements */ 
        template<class T, class IDX, int ndim>
        constexpr dg_dof_offsets(
            const std::vector<ELEMENT::FiniteElement<T, IDX, ndim> > &elements
        ) noexcept : offsets(elements.size() + 1) {
            offsets[0] = 0;
            for(int i = 0; i < elements.size(); ++i){
                offsets[i + 1] = offsets[i] + elements[i].nbasis();
            }
        }

        /** get the ith offset */
        constexpr std::size_t operator[](std::size_t i) 
            const noexcept { return offsets[i]; }

        /** get reference to the last offset */ 
        constexpr auto back() const noexcept { return offsets.back(); }

        /** @brief get the size requirement for all degrees of freedom given
         * the number of vector components per dof 
         * @param nv_comp th number of vector components per dof 
         * @return the size requirement
         */
        constexpr std::size_t calculate_size_requirement( int nv_comp ) const noexcept {
            return offsets.back() * nv_comp;
        }

        /**
         * @brief calculate the largest size requirement for a single element 
         * @param nv_comp the number of vector components per dof 
         * @return the maximum size requirement 
         */
        constexpr std::size_t max_el_size_reqirement( int nv_comp ){
            std::size_t max_offset = 0;
            for(int i = 1; i < offsets.size(); ++i){
                max_offset = std::max(max_offset, offsets[i] - offsets[i - 1]);
            }
            return nv_comp * max_offset;
        }
    };

    /**
     * @brief LayoutPolicy for the DG representation of a vector-valued fe_function 
     */
    template<
        typename T,                           /// the element type
        int ncomp = dynamic_ncomp,             /// the number of vector components  TODO: switch to size_t and std::dymamic_extent
        LAYOUT_VECTOR_ORDER order = DOF_LEFT /// how dofs are organized wrt vector components
    >
    class dg_layout {
        const dg_dof_offsets &offsets;

        std::enable_if<is_dynamic_ncomp<ncomp>::value, int> ncomp_d;

        inline constexpr int get_ncomp() const {
            if constexpr(is_dynamic_ncomp<ncomp>::value){
                return ncomp_d;
            } else {
                return ncomp;
            }
        }
        public:
        /**
         * @brief create a dg_layout 
         * @param offsets the vector component independent dof offsets
         * @param ncomp_d needed if ncomp is set to dynamic (sets the number of vector components)
         */
        constexpr dg_layout(
            const dg_dof_offsets &offsets,
            int ncomp_d
        ) requires(is_dynamic_ncomp<ncomp>::value)
        : offsets(offsets), ncomp_d(ncomp_d) {}

        constexpr dg_layout(
            const dg_dof_offsets &offsets
        ) requires (!is_dynamic_ncomp<ncomp>::value) 
        : offsets(offsets) {}

        inline static constexpr bool is_global() { return true;  }
        inline static constexpr bool is_nodal()  { return false; }

        /**
         * @brief
         * consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same 
         */
        inline static constexpr bool local_dof_contiguous() { return true; }

        /**
         * @brief convert a multidimensional index to a single offset 
         * @param idx collection of the element, dof, and component indices
         *        TODO: ABIs tend to put 8+ byte structs on stack instead of passing
         *        through registers so maybe go away from this construct
         */
        constexpr std::size_t operator()(const fe_index &idx) const {
            std::size_t iel = idx.iel;
            std::size_t ildof = idx.idof;
            std::size_t iv = idx.iv;
            if constexpr (order == DOF_LEFT) {
                int component_mult = get_ncomp();
                return component_mult * offsets[iel] + component_mult * ildof + iv;
            } else {
                return iv * offsets.back() + offsets[iel] + ildof;
            }
        }

        /** @brief the upper bound of the index space */
        constexpr std::size_t size() const noexcept { 
            int component_mult = get_ncomp();
            return offsets.back() * component_mult;
        }

        /**
         * @brief create the element layout that matches this layout 
         * @param iel the element index 
         * @return the layout to construct 
         */
        constexpr auto create_element_layout(std::size_t iel){
            std::size_t ndof = offsets[iel + 1] - offsets[iel];
            if constexpr(is_dynamic_ncomp<ncomp>::value){
                return FE::compact_layout<T, ncomp, order>{ndof, ncomp_d};
            } else {
                return FE::compact_layout<T, ncomp, order>(ndof);
            }
        }
        
    };

    /** 
     * @brief with equivalent layout parameters 
     * a DG Layout is block copyable
     * so specialize the struct
     */
    template<
        typename T,
        int ncomp,
        LAYOUT_VECTOR_ORDER order
    >
    struct is_equivalent_el_layout<
        compact_layout<T, ncomp, order>,
        dg_layout<T, ncomp, order>
    > {
        static constexpr bool value = true;
    };
    
}
