/**
 * @brief data layout for values contained in a local trace space 
 */

#pragma once
#include <type_traits>
#include "iceicle/element/TraceSpace.hpp"

namespace iceicle {

    /**
     * @brief memory layout for data that is compact to a single trace 
     * with vector-component fastest access 
     *
     * @tparam IDX the index type 
     * @param vextent the number of vector components per dof
     *        NOTE: currently don't support dynamic
     */
    template<typename IDX, std::size_t vextent>
    struct trace_layout_right {
        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        // the number of degrees of freedom
        index_type _ndof;

        /**
         * @brief Construct a trace layout 
         * over the trace basis of a given trace 
         * using the dimensionality as the number of vector components 
         */
        template<class T>
        constexpr trace_layout_right(
            const TraceSpace<T, index_type, (int) vextent> &trace
        ) noexcept : _ndof{trace.nbasis_trace()} {}

        trace_layout_right(const trace_layout_right<IDX, vextent>& other) noexcept = default;
        trace_layout_right(trace_layout_right<IDX, vextent>&& other) noexcept = default;

        trace_layout_right<IDX, vextent>& operator=(
                const trace_layout_right<IDX, vextent>& other) noexcept = default;
        trace_layout_right<IDX, vextent>& operator=(
                trace_layout_right<IDX, vextent>&& other) noexcept = default;


        // ==============
        // = Properties =
        // ==============

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
            return vextent;
        }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr size_type size() const noexcept {
            return nv() * ndof();
        }

        // ============
        // = Indexing =
        // ============
#ifndef NDEBUG 
        inline static constexpr bool index_noexcept = false;
#else 
        inline static constexpr bool index_noexcept = true;
#endif

        /**
         * Get the result of the mapping from an index double
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
    trace_layout_right(const TraceSpace<T, IDX, ndim>) -> trace_layout_right<IDX, (std::size_t) ndim>;
}
