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

#include <type_traits>
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
    };

    /**
     * @brief LayoutPolicy for the DG representation of a vector-valued fe_function 
     */
    template<
        typename T,                           /// the element type
        int ncomp = dynamic_ncomp,             /// the number of vector components
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

        inline static constexpr int is_global() { return true;  }
        inline static constexpr int is_nodal()  { return false; }

        /**
         * @brief convert a multidimensional index to a single offset 
         * @brief idx collection of the element, dof, and component indices
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
        
    };
    
}
