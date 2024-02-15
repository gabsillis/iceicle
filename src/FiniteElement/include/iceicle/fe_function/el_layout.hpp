/**
 * @brief memory layout for element local vector valued function
 *
 * @author Gianni Absillis (gabsill)
 */
#pragma once
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/layout_enums.hpp"
#include <type_traits>
namespace FE {

    /**
     * @brief memory layout for data that is compact to a single element 
     *
     * This functions as a LayoutPolicy for elspan
     * TODO: figure out how to have this double as a layoutpolicy for fespan
     * might need to remove size calculations from layout policy
     *
     * @tparam T the data type 
     * @tparam ncomp the number of vector components
     * @tparam order the order of the dofs vs vector components in memory 
     *               LEFT corresponds to the left index in C-style indexing
     */
    template<
        typename T,
        std::size_t ncomp = dynamic_ncomp,
        LAYOUT_VECTOR_ORDER order = DOF_LEFT 
    >
    class compact_layout {

        /// dynamic number of vector components 
        std::enable_if<is_dynamic_ncomp<ncomp>::value, int> ncomp_d;

        // the number of degrees of freedom
        std::size_t ndof;

        /** @brief accessor for the number of components that abstracts compile_time vs dynamic */
        inline constexpr std::size_t get_ncomp() const noexcept {
            if constexpr(is_dynamic_ncomp<ncomp>::value){
                return ncomp_d;
            } else {
                return ncomp;
            }
        }

        public:

        using extents_type = compact_index_extents<ncomp>;

        /**
         * @brief Constructor 
         * @param el the element this will represent compact data for 
         * @param ncomp_d (enabled if dynamic ncomponents) the number of vector components
         */
        template<typename IDX, int ndim>
        constexpr compact_layout(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el,
            std::enable_if<is_dynamic_ncomp<ncomp>::value, std::size_t> ncomp_d
        ) : ncomp_d(ncomp_d), ndof(el.nbasis()){}

        /**
         * @brief Constructor 
         * @param el the element this will represent compact data for 
         */
        template<typename IDX, int ndim>
        constexpr compact_layout(
            const ELEMENT::FiniteElement<T, IDX, ndim> &el
        ) requires(!is_dynamic_ncomp<ncomp>::value) : ndof(el.nbasis()){}

        /**
         * @brief Constructor
         * @param ndof the number of degrees of freedom in this element 
         * @param ncomp_d (enabled if dynamic ncomponents) the number of vector components
         */
        constexpr compact_layout(
            int ndof,
            std::enable_if<is_dynamic_ncomp<ncomp>::value, std::size_t> ncomp_d
        ) : ncomp_d(ncomp_d), ndof(ndof) {}

        /**
         * @brief Constructor
         * @param ndof the number of degrees of freedom in this element 
         * @param ncomp_d (enabled if dynamic ncomponents) the number of vector components
         */
        constexpr compact_layout(
            int ndof
        ) requires(!is_dynamic_ncomp<ncomp>::value): ndof(ndof) {}

        /**
         * @brief convert a multidimensional index to a single offset 
         * @brief idx the compact multidimensional index 
         * @return a 1d index offset into the element compact data
         */
        constexpr std::size_t operator()(const compact_index &idx) const noexcept {
            if constexpr (order == DOF_LEFT) {
                return idx.idof * get_ncomp() + idx.iv;
            } else {
                return ndof * idx.iv + idx.idof;
            }
        }

        /**
         * @brief get the size of the compact index space 
         * @return the size of the compact index space 
         */
        constexpr std::size_t size() const noexcept { return ndof * get_ncomp(); }

        /**
         * @brief get the extents of the compact index space 
         */
        constexpr extents_type extents() const noexcept {
            return extents_type{.ndof = ndof, .nv = (std::size_t) get_ncomp()};
        }

    };
}
