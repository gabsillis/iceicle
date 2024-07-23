#include <vector>
namespace iceicle {

    /// @brief A collection of index sets that define a geometric subset
    /// of the mesh when considering interface conservation
    template<class index_type>
    struct mdg_selection {

        /// @brief the indices of traces to consider interface conditions for
        std::vector<index_type> selected_traces{};
        
        /// @brief the global node indices of the nodes to represent dofs for 
        std::vector<index_type> selected_nodes{};

        /// @brief index in the selected node dofs each gdof maps to, or size() if not included
        std::vector<index_type> inv_selected_nodes{};
    };
}
