#pragma once
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/fe_function/layout_enums.hpp>
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/basis/dof_mapping.hpp>
#include <stdexcept>
#include <type_traits>

namespace iceicle {

    /// @brief simple layout of the index space over a set of contiguous degrees of freedom 
    /// vector components are the fastest (analagous to std::layout_right)
    ///
    /// @tparam IDX the index type 
    /// @tparam vextent the extent of the vector component
    template< class IDX, std::size_t vextent >
    struct dof_layout_right {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        /// @brief the number of degrees of freedom represented by this layout
        std::size_t _ndof;

        // ==============
        // = Properties =
        // ==============

        /// @brief this makes no garuantees about contiguous degrees of freedom with respect to elements
        inline static constexpr auto local_dof_contiguous() -> bool { return false; }

        /// @brief static access to the extents 
        inline static constexpr auto static_extent() noexcept -> std::size_t 
        { return vextent; }

        // =========
        // = Sizes =
        // =========
        /// @brief get the number of degrees of freedom
        [[nodiscard]] inline constexpr auto ndof() const noexcept -> size_type 
        { return _ndof; }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr auto nv() const noexcept -> size_type { return vextent; }

        /// @brief the size of the compact index space
        [[nodiscard]] inline constexpr auto size() const noexcept -> size_type { return ndof() * nv(); }

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
        [[nodiscard]] constexpr auto operator[](
            index_type idof,
            index_type iv
        ) const noexcept(index_noexcept) -> index_type {
#ifndef NDEBUG
            // Bounds checking version in debug 
            // NOTE: allow indexing ndof()
            // for nodes that arent in inv_selected_nodes but still 
            // valid gdofs
            if(idof < 0  || idof >= ndof()  ) throw std::out_of_range("Dof index out of range");
            if(iv < 0    || iv >= nv()      ) throw std::out_of_range("Vector compoenent index out of range");
#endif
           return idof * nv() + iv; 
        }
    };

    /**
     * @brief a dg layout of the index space where the 
     * vector components are the fastest (analagous to std::layout_right)
     *
     * @tparam IDX the index type 
     * @tparam MapType the type that maps local to global dofs
     * @tparam vextent the extent of the vector component
     * @tparam include_ghost include "ghost" elements from adjoining processes
     */
    template<class IDX, class MapType, std::size_t vextent, bool include_ghost>
    struct fe_layout_right {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;
        using dof_mapping_type = MapType;

        static_assert(std::is_same_v<typename dof_mapping_type::index_type, index_type>, "index types are mismatched");

        // ===========
        // = Members =
        // ===========

        /// @brief the map from the element and local dof indices to gdof 
        /// heavy type: will require separate maps on host and device
        const dof_mapping_type& map_ref;

        /// @brief a map of parallel indices of elements that also stores ownership information
        const pindex_map<IDX>& element_partitioning;

        /// @brief a map of the parallel indices of dofs that stores ownership information
        const pindex_map<IDX>& dof_partitioning;

        /// @brief dynamic vector component if vextent is not specified
        std::enable_if<is_dynamic_size<vextent>::value, index_type> nv_d;

        // ================
        // = Constructors =
        // ================
        
        /// @brief Constructor from the three required mappings to define the layout 
        /// with static extent
        /// @param map_ref the element-wise dof mapping for the index set of this mpi rank 
        /// @param element_partitioning a map of parallel indices between mpi ranks for the elements
        /// @param dof_partitioning a map of parallel indices between mpi ranks for the dofs 
        /// @param ghost_arg set to true to include degrees of freedom corresponding to interprocess chost elements
        fe_layout_right(const MapType& map_ref, const pindex_map<IDX>& element_partitioning,
                const pindex_map<IDX>& dof_partitioning,
                std::integral_constant<bool, include_ghost> ghost_arg = std::false_type{}) 
        noexcept requires( !is_dynamic_size<vextent>::value )
        : map_ref{map_ref}, element_partitioning{element_partitioning}, dof_partitioning{dof_partitioning}
        {}

        /// @brief Constructor from the three required mappings to define the layout 
        /// with runtime size vector component extent
        /// @param map_ref the element-wise dof mapping for the index set of this mpi rank 
        /// @param element_partitioning a map of parallel indices between mpi ranks for the elements
        /// @param dof_partitioning a map of parallel indices between mpi ranks for the dofs 
        /// @param ghost_arg set to true to include degrees of freedom corresponding to interprocess chost elements
        fe_layout_right(const MapType& map_ref, const pindex_map<IDX>& element_partitioning,
                const pindex_map<IDX>& dof_partitioning, index_type nv,
                std::integral_constant<bool, include_ghost> ghost_arg = std::false_type{}) 
        noexcept requires( is_dynamic_size<vextent>::value )
        : map_ref{map_ref}, element_partitioning{element_partitioning}, dof_partitioning{dof_partitioning}, nv_d{nv}
        {}

        /// @brief Constructor from the three required mappings to define the layout 
        /// with integral constant for argument deduction
        /// @param map_ref the element-wise dof mapping for the index set of this mpi rank 
        /// @param element_partitioning a map of parallel indices between mpi ranks for the elements
        /// @param dof_partitioning a map of parallel indices between mpi ranks for the dofs 
        /// @param ghost_arg set to true to include degrees of freedom corresponding to interprocess chost elements
        fe_layout_right(const MapType& map_ref, const pindex_map<IDX>& element_partitioning,
                const pindex_map<IDX>& dof_partitioning,
                std::integral_constant<std::size_t, vextent>,
                std::integral_constant<bool, include_ghost> ghost_arg = std::false_type{})
        noexcept requires(!is_dynamic_size<vextent>::value) 
        : map_ref{map_ref}, element_partitioning{element_partitioning}, dof_partitioning{dof_partitioning} {}

        /// @brief construct a layout to map the degrees of freedom of the given FESpace 
        /// with integral constant for argument deduction
        /// @param fespace the finite element space 
        template< class T, int conformity, int ndim >
        fe_layout_right(
            const FESpace<T, IDX, ndim, conformity>& fespace,
            std::integral_constant<std::size_t, vextent>,
            std::integral_constant<bool, include_ghost> ghost_arg = std::false_type{}
        ) : map_ref{fespace.dofs}, element_partitioning{fespace.meshptr->element_partitioning}, 
            dof_partitioning{fespace.dof_partitioning} {}

        fe_layout_right(const fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>& other) noexcept = default;
        fe_layout_right(fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>&& other) noexcept = default;

        // reference members can't assign
//         fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>& operator=(
//                 const fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>& other) noexcept = default;
//         fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>& operator=(
//                 fe_layout_right<IDX, dof_mapping_type, vextent, include_ghost>&& other) noexcept = default;

        // ==============
        // = Properties =
        // ==============

        /**
         * @brief consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same
         */
        [[nodiscard]] inline static constexpr 
        auto local_dof_contiguous() noexcept
        -> bool
        { return dof_mapping_type::local_dof_contiguous(); }

        /// @brief static access to the extents 
        [[nodiscard]] inline static constexpr 
        auto static_extent() noexcept
        -> std::size_t 
        { return vextent; }

        /**
         * @brief if the layout is over a view including ghost elements 
         * then this is a read only view because writing to ghost elements 
         * requires explicit communication
         */
        [[nodiscard]] inline static constexpr
        auto read_only() noexcept
        -> bool 
        { return include_ghost; }

        // =========
        // = Sizes =
        // =========
       
        /// @brief the number of elements in the index space */
        [[nodiscard]] inline constexpr 
        auto nelem() const noexcept 
        -> size_type
        { 
            if constexpr (include_ghost) {
                return map_ref.nelem(); 
            } else {
                return element_partitioning.owned_range_size(mpi::mpi_world_rank());
            }
        }

        /// @brief get the number of global degrees of freedom
        [[nodiscard]] inline constexpr 
        auto ndof() const noexcept
        -> size_type
        { return map_ref.size(); }

        /// @brief get the number of degrees of freedom for the given element 
        /// @param ielem the element index
        [[nodiscard]] inline constexpr 
        auto ndof(index_type ielem) const noexcept
        -> size_type
        { return map_ref.ndof_el(ielem); }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr 
        auto nv() const noexcept
        -> size_type 
        {
            if constexpr(is_dynamic_size<vextent>::value){
                return nv_d;
            } else {
                return (size_type) vextent;
            }
        }

        /// @brief the total size of the global index space represented by this layout
        [[nodiscard]] inline constexpr 
        auto size() const noexcept 
        -> size_type 
        { return ndof() * nv(); }

        // ============
        // = Indexing =
        // ============
#ifndef NDEBUG 
        inline static constexpr bool index_noexcept = false;
#else 
        inline static constexpr bool index_noexcept = true;
#endif

        /**
         * Get the result of the mapping from an index triple 
         * to the global index 
         * @param ielem the element index 
         * @param idof the degree of freedom index 
         * @param iv the vector component index
         */
        [[nodiscard]] constexpr index_type operator[](
            index_type ielem,
            index_type idof,
            index_type iv
        ) const noexcept(index_noexcept) {
#ifndef NDEBUG
            // Bounds checking version in debug
            if(ielem < 0 || ielem >= nelem()    ) throw std::out_of_range("Element index out of range");
            if(idof  < 0 || idof  >= ndof(ielem)) throw std::out_of_range("Dof index out of range");
            if(iv    < 0 || iv    >= nv()       ) throw std::out_of_range("Vector compoenent index out of range");
#endif
            // the global degree of freedom index
            index_type gdof = map_ref[ielem, idof];
            return gdof * nv() + iv; 
        }

        /**
         * @brief get the result of mapping from a (idof, iv) index pair 
         * to the global index 
         * @param igdof the global degree of freedom index 
         * @param iv the vector component index
         */
        [[nodiscard]] constexpr index_type operator[](
            index_type igdof,
            index_type iv
        ) const noexcept(index_noexcept) {
#ifndef NDEBUG
            // Bounds checking version in debug
            if(igdof  < 0 || igdof  >= map_ref.size()) throw std::out_of_range("Dof index out of range");
            if(iv    < 0 || iv    >= nv()       ) throw std::out_of_range("Vector compoenent index out of range");
#endif
            // the global degree of freedom index
            return igdof * nv() + iv; 
        }

        /** 
         * @brief given a process-local global degree of freedom and vector componeent index,
         * get the parallel data index
         */
        [[nodiscard]] inline constexpr 
        auto get_pindex(index_type igdof, index_type iv) const noexcept(index_noexcept)
        -> index_type
        { 
#ifndef NDEBUG
            // Bounds checking version in debug
            if(igdof  < 0 || igdof  >= map_ref.size()) throw std::out_of_range("Dof index out of range");
            if(iv    < 0 || iv    >= nv()       ) throw std::out_of_range("Vector compoenent index out of range");
#endif
            return dof_partitioning.p_indices[igdof] * nv() + iv;
        }

        /** 
         * @brief given an index triple
         * get the parallel data index
         */
        [[nodiscard]] inline constexpr 
        auto get_pindex(index_type ielem, index_type ildof, index_type iv) const noexcept(index_noexcept)
        -> index_type 
        {
#ifndef NDEBUG
            // Bounds checking version in debug
            if(ielem < 0 || ielem >= nelem()    ) throw std::out_of_range("Element index out of range");
            if(ildof  < 0 || ildof  >= ndof(ielem)) throw std::out_of_range("Dof index out of range");
            if(iv    < 0 || iv    >= nv()       ) throw std::out_of_range("Vector compoenent index out of range");
#endif

            index_type igdof = map_ref[ielem, ildof];
            return dof_partitioning.p_indices[igdof] * nv() + iv;
        }
    };

    /// @brief cast the layout to a an index subset that excludes ghost interprocess elements
    /// @param layout the layout to cast
    template<class IDX, class MapT, std::size_t vextent, bool include_ghost>
    auto exclude_ghost(fe_layout_right<IDX, MapT, vextent, include_ghost> layout)
    noexcept -> fe_layout_right<IDX, MapT, vextent, false> 
    {
        if constexpr(vextent == dynamic_ncomp){
            return fe_layout_right{layout.map_ref, layout.element_partitioning,
                layout.dof_partitioning, layout.nv(), std::false_type{}};
        } else {
            return fe_layout_right{layout.map_ref, layout.element_partitioning,
                layout.dof_partitioning, tmp::to_size<vextent>{}, std::false_type{}};
        }
    }

    // ====================
    // = Deduction Guides =
    // ====================

    // dynamic extent, no ghost
    template<class IDX, class MapT>
    fe_layout_right(const MapT&, IDX nv, const pindex_map<IDX>&, const pindex_map<IDX>&, IDX)
    -> fe_layout_right<IDX, MapT, dynamic_ncomp, false>;

    // static extent, no ghost
    template<class IDX, class MapT, std::size_t vextent>
    fe_layout_right(const MapT&, const pindex_map<IDX>&, const pindex_map<IDX>&,
            std::integral_constant<std::size_t, vextent>) 
    -> fe_layout_right<IDX, MapT, vextent, false>;

    // static extent, ghost option
    template<class IDX, class MapT, std::size_t vextent, bool use_ghost>
    fe_layout_right(const MapT&, const pindex_map<IDX>&, const pindex_map<IDX>&,
            std::integral_constant<std::size_t, vextent>, std::integral_constant<bool, use_ghost>) 
    -> fe_layout_right<IDX, MapT, vextent, use_ghost>;

    // fespace constructor static extent, no ghost
    template<class T, class IDX, int ndim, int conformity, std::size_t vextent>
    fe_layout_right(const FESpace<T, IDX, ndim, conformity>&,
            std::integral_constant<std::size_t, vextent>)
    -> fe_layout_right<IDX, dof_map<IDX, ndim, conformity>, vextent, false>;

    // fespace constructor static extent, ghost option
    template<class T, class IDX, int ndim, int conformity, std::size_t vextent, bool use_ghost>
    fe_layout_right(const FESpace<T, IDX, ndim, conformity>&,
            std::integral_constant<std::size_t, vextent>, std::integral_constant<bool, use_ghost>) 
    -> fe_layout_right<IDX, dof_map<IDX, ndim, conformity>, vextent, use_ghost>;

    // === Type Aliases for clarity ===

    // Insert explicative laden rant about clang not supporting a c++20 feature in 2024
//
//    // layout right using dg map
//    template<class IDX, std::size_t vextent>
//    using dg_layout_right = fe_layout_right<IDX, dg_dof_map<IDX>, vextent>;
//
//    // default dg layout is layout right
//    template<class IDX, std::size_t vextent>
//    using dg_layout = dg_layout_right<IDX, vextent>;
}
