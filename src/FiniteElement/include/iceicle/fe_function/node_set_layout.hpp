/**
 * @brief dof layout over a selected node set 
 * can be used to represent MDG unknowns and residuals
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include <type_traits>
#include <vector>
namespace FE {

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
        std::vector<index_type> selected_traces;

        /// @brief the global node indices of the nodes to represent dofs for 
        std::vector<index_type> selected_nodes;

        /// @brief index in the selected node dofs each gdof maps to, or size() if not included
        std::vector<index_type> inv_selected_nodes;

    };


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
        inline static constexpr auto local_dof_contiguous() noexcept -> bool { return true; }

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
            if(idof < 0  || idof >= ndof()  ) throw std::out_of_range("Dof index out of range");
            if(iv < 0    || iv >= nv()      ) throw std::out_of_range("Vector compoenent index out of range");
#endif
           return idof * nv() + iv; 
        }
    };
}
