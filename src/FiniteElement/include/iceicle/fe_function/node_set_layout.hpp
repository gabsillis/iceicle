/**
 * @brief dof layout over a selected node set 
 * can be used to represent MDG unknowns and residuals
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fespace/fespace.hpp"
#include <ranges>
#include <type_traits>
#include <vector>
#include <stdexcept>
namespace iceicle {

    /**
     * @brief this maps a set of selected faces and their corresponding nodes to 
     * continuous degrees of freedom
     */
    template<class IDX>
    struct nodeset_dof_map {

        // ============
        // = Typedefs =
        // ============

        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;

        // =================
        // = Class Members =
        // =================

        /// @brief the indices of traces to add dof's for 
        std::vector<index_type> selected_traces{};

        /// @brief the global node indices of the nodes to represent dofs for 
        std::vector<index_type> selected_nodes{};

        /// @brief index in the selected node dofs each gdof maps to, or size() if not included
        std::vector<index_type> inv_selected_nodes{};

        // ================
        // = Constructors =
        // ================

        template<std::ranges::forward_range R1, typename T, int ndim>
        nodeset_dof_map(R1&& trace_indices, FESpace<T, IDX, ndim>& fespace)
#ifdef __cpp_lib_containers_ranges
        : selected_traces(std::from_range, trace_indices) 
#else 
        : selected_traces{std::ranges::begin(trace_indices), std::ranges::end(trace_indices)}
#endif
        {
            // helper array to keep track of which global node indices to select
            std::vector<bool> to_select(fespace.meshptr->n_nodes(), false);
            using trace_type = std::remove_reference_t<decltype(fespace)>::TraceType;

            // loop over selected faces and select nodes
            for(index_type trace_idx : trace_indices){
                const trace_type& trace = fespace.traces[trace_idx];
                for(index_type inode : trace.face->nodes_span()){
                    to_select[inode] = true;
                }
            }

            // loop over the boundary faces and deactivate all boundary nodes 
            // since some may be connected to an active interior face 
            for(const trace_type &trace : fespace.get_boundary_traces()){
                for(index_type inode : trace.face->nodes_span()){
                    to_select[inode] = false;
                }
            }

            // construct the selected nodes list 
            for(int inode = 0; inode < fespace.meshptr->n_nodes(); ++inode){
                if(to_select[inode]) selected_nodes.push_back(inode);
            }

            // default value for nodes that aren't selected is to map to selected_nodes.size()
            inv_selected_nodes = std::vector<index_type>(fespace.meshptr->n_nodes(), selected_nodes.size());
            for(int idof = 0; idof < selected_nodes.size(); ++idof){
                inv_selected_nodes[selected_nodes[idof]] = idof;
            }
        }

        nodeset_dof_map() = default;
        nodeset_dof_map(const nodeset_dof_map<IDX>& other) = default;
        nodeset_dof_map(nodeset_dof_map<IDX>&& other) = default;
        nodeset_dof_map<IDX>& operator=(const nodeset_dof_map<IDX>& other) = default;
        nodeset_dof_map<IDX>& operator=(nodeset_dof_map<IDX>&& other) = default;

    };

    template<std::ranges::forward_range R, typename T, int ndim>
    nodeset_dof_map(R&&, FESpace<T, std::ranges::range_value_t<R>, ndim>&) -> nodeset_dof_map<std::ranges::range_value_t<R>>;

    template<class IDX, std::size_t vextent>
    struct node_selection_layout {
        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;
        
        // ===========
        // = Members =
        // ===========
        const nodeset_dof_map<IDX>& nodeset;

        // ==============
        // = Properties =
        // ==============

        /**
         * @brief consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same
         */
        inline static constexpr auto local_dof_contiguous() noexcept -> bool { return false; }

        /// @brief static access to the extents 
        inline static constexpr auto static_extent() noexcept -> std::size_t {
            return vextent;
        }

        // =========
        // = Sizes =
        // =========

        /// @brief get the number of degrees of freedom
        [[nodiscard]] inline constexpr auto ndof() const noexcept -> size_type { return nodeset.selected_nodes.size(); }

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
}
