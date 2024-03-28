/**
 * @brief memory layout for element local vector valued function
 *
 * @author Gianni Absillis (gabsill)
 */
#pragma once
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/layout_enums.hpp"
#include <span>
#include <type_traits>
namespace FE {

    /**
     * @brief memory layout for data that is compact to a single element 
     * with vector-component fastest access
     *
     * This functions as a LayoutPolicy for elspan
     *
     * @tparam IDX the index type 
     * @tparam vextent the number of vector components per dof
     */
    template<typename IDX, std::size_t vextent>
    struct compact_layout_right {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        /// dynamic number of vector components 
        std::enable_if<is_dynamic_size<vextent>::value, index_type> nv_d;

        // the number of degrees of freedom
        index_type _ndof;

        // ================
        // = Constructors =
        // ================

        /**
         * @brief Constructor 
         * @param el the element this will represent compact data for 
         * @param nv_d the number of vector components per dof
         */
        template<class T, int ndim>
        constexpr compact_layout_right(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            index_type nv_d
        ) noexcept requires(is_dynamic_size<vextent>::value) 
        : nv_d(nv_d), _ndof{el.nbasis()}{}

        /**
         * @brief Constructor 
         * @param el the element this will represent compact data for 
         */
        template<typename T, int ndim>
        constexpr compact_layout_right(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el
        ) noexcept requires(!is_dynamic_size<vextent>::value)
        : _ndof{el.nbasis()}{}

        /**
         * @brief Constructor
         * @param ndof the number of degrees of freedom in this element 
         * @param nv_d the number of vector components per dof
         */
        constexpr compact_layout_right(index_type ndof, index_type nv_d) 
        noexcept requires(is_dynamic_size<vextent>::value)
        : nv_d(nv_d), _ndof{ndof} {}

        /**
         * @brief Constructor
         * @param ndof the number of degrees of freedom in this element 
         */
        constexpr compact_layout_right( index_type ndof ) noexcept
        requires(!is_dynamic_size<vextent>::value)
        : _ndof{ndof} {}

        /**
         * @brief Constructor
         * @param ndof the number of degrees of freedom in this element 
         * @param the extent for argument deduction
         */
        constexpr compact_layout_right( index_type ndof, std::integral_constant<std::size_t, vextent>) noexcept
        requires(!is_dynamic_size<vextent>::value)
        : _ndof{ndof} {}

        compact_layout_right(const compact_layout_right<IDX, vextent>& other) noexcept = default;
        compact_layout_right(compact_layout_right<IDX, vextent>&& other) noexcept = default;

        compact_layout_right<IDX, vextent>& operator=(
                const compact_layout_right<IDX, vextent>& other) noexcept = default;
        compact_layout_right<IDX, vextent>& operator=(
                compact_layout_right<IDX, vextent>&& other) noexcept = default;

        // ==============
        // = Properties =
        // ==============

        /**
         * @brief consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same
         */
        inline static constexpr bool local_dof_contiguous() noexcept { return true; }

        /// @brief static access to the extents 
        inline static constexpr std::size_t static_extent() noexcept {
            return vextent;
        }

        // =========
        // = Sizes =
        // =========

        /// @brief get the number of degrees of freedom
        [[nodiscard]] constexpr size_type ndof() const noexcept { 
            return _ndof;
        }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr size_type nv() const noexcept {
            if constexpr(is_dynamic_size<vextent>::value){
                return nv_d;
            } else {
                return vextent;
            }
        }
        /**
         * @brief get the size of the compact index space 
         * @return the size of the compact index space 
         */
        constexpr size_type size() const noexcept { return ndof() * nv(); }


        // ============
        // = Indexing =
        // ============
#ifndef NDEBUG 
        inline static constexpr bool index_noexcept = false;
#else 
        inline static constexpr bool index_noexcept = true;
#endif

        /**
         * Get the result of the mapping from an index pair 
         * to the one dimensional index of the elment 
         * @param idof the degree of freedom index 
         * @param iv the vector component index
         */
        [[nodiscard]] constexpr index_type operator[](
            index_type idof,
            index_type iv
        ) const noexcept(index_noexcept) {
#ifndef NDEBUG
            // Bounds checking version in debug
            if(idof < 0  || idof >= ndof()  ) throw std::out_of_range("Dof index out of range");
            if(iv < 0    || iv >= nv()      ) throw std::out_of_range("Vector compoenent index out of range");
#endif
           return idof * nv() + iv; 
        }
    };

    // Deduction Guides
    template<class T, class IDX, int ndim>
    compact_layout_right(const ELEMENT::FiniteElement<T, IDX, ndim> &, IDX) -> compact_layout_right<IDX, std::dynamic_extent>;
    template<class index_type>
    compact_layout_right( index_type )  -> compact_layout_right<index_type, std::dynamic_extent> ;

    template<class index_type, std::size_t vextent>
    compact_layout_right( index_type , std::integral_constant<std::size_t, vextent>) -> compact_layout_right<index_type, vextent> ;
}
