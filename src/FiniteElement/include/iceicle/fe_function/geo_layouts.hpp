/// @brief data layouts for geometry degrees of freedom
/// @author Gianni Absillis (gabsill@ncsu.edu)
#pragma once
#include <type_traits>
#include <map>
#include <span>
#include <memory>
#include <vector>
#include <iceicle/fespace/fespace.hpp>


namespace iceicle {

    // Parametric Functions 

    /// @brief parametric mapping that fixes the x coordinate and maps s -> y
    template<class T>
    struct wall_2d_free_y {
        static constexpr int ndim = 2;

        T x_constraint;

        auto operator()(std::span<T> svec, std::span<T, ndim> xvec) -> void {
            xvec[0] = x_constraint;
            xvec[1] = svec[0];
        }
    };

    template<class T, int ndim>
    class ParametricCoordTransformation {
        public:
        virtual ~ParametricCoordTransformation() = default;

        virtual void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const = 0;

        virtual void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const = 0;

        virtual auto s_size() const -> int = 0;
    };

    template<class T, int ndim>
    class fixed_component_constraint : public ParametricCoordTransformation<T, ndim> {
        public:
        /// The value that the fixed component takes
        T component_value;

        /// the index of the component that is fixed
        int component_index;

        fixed_component_constraint(T component_value, int component_index)
        : ParametricCoordTransformation<T, ndim>{}, component_value(component_value), component_index(component_index) {}

        fixed_component_constraint() = default;

        void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const override {
            for(int idim = 0; idim < component_index; ++idim)
                xvec[idim] = svec[idim];

            xvec[component_index] = component_value;

            for(int idim = component_index + 1; idim < ndim; ++idim)
                xvec[idim] = svec[idim - 1];
        };

        void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const override {
            for(int idim = 0; idim < component_index; ++idim)
                svec[idim] = xvec[idim];

            for(int idim = component_index + 1; idim < ndim; ++idim)
                svec[idim - 1] = xvec[idim];
        };

        auto s_size() const -> int override {
            return ndim - 1;
        }

    };

    template<class T, class IDX, int ndim>
    struct geo_dof_map {

        // ============
        // = Typedefs =
        // ============

        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;

        // ================
        // = Data Members =
        // ================

        /// @brief the indices of traces to add dof's for 
        std::vector<index_type> selected_traces{};

        /// @brief the global node indices of the nodes to represent dofs for 
        std::vector<index_type> selected_nodes{};

        /// @brief index in the selected node dofs each gdof maps to, or size() if not included
        std::vector<index_type> inv_selected_nodes{};

        /// @brief the indices that represent the start of the data for each represented dof
        std::vector<index_type> cols{};

        /// @brief for degrees of freedom where the coordinates are parameterized 
        /// store the parametric_function that represents this map
        std::map<index_type, std::unique_ptr<ParametricCoordTransformation<T, ndim>> > parametric_accessors;

        /// @brief true if the given index is a parametric dof
        std::vector<bool> is_parametric;

        /// @brief flag that gets set to true when the data structures are all in usable state
        bool finalized = false;

        // ================
        // = Constructors =
        // ================

        template<std::ranges::forward_range R1 >
        geo_dof_map(R1&& trace_indices, FESpace<T, IDX, ndim>& fespace, bool remove_boundary_dofs = false)
        : selected_traces{std::ranges::begin(trace_indices), std::ranges::end(trace_indices)}
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

            if(remove_boundary_dofs){
                // loop over the boundary faces and deactivate all boundary nodes 
                // since some may be connected to an active interior face 
                for(const trace_type &trace : fespace.get_boundary_traces()){
                    for(index_type inode : trace.face->nodes_span()){
                        to_select[inode] = false;
                    }
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

            // initialize the is_parametric array
            is_parametric = std::vector<bool>(selected_nodes.size(), false);

            // this is finalized until parametric nodes are registered
            cols.reserve(ndof()+1);
            cols.push_back(0);
            for(int idof = 0; idof < ndof(); ++idof){
                cols.push_back(cols[idof] + ndim);
            }
            finalized = true;
        }

        geo_dof_map() = default;
        geo_dof_map(const geo_dof_map<T, IDX, ndim>& other) = default;
        geo_dof_map(geo_dof_map<T, IDX, ndim>&& other) = default;
        geo_dof_map<T, IDX, ndim>& operator=(const geo_dof_map<T, IDX, ndim>& other) = default;
        geo_dof_map<T, IDX, ndim>& operator=(geo_dof_map<T, IDX, ndim>&& other) = default;

        /// @brief register a node as a parametrically controlled node 
        /// the vector components in parametric space s get mapped to x (size = ndim) 
        /// @param inode the index of the node in the mesh (will get converted internally to the ldof)
        /// if inode doesn't have an ldof this entry is just ignored
        /// @param parametric_transform the set of functions that governs the invertible s -> x mapping
        template<class parametric_t>
        auto register_parametric_node(index_type inode, parametric_t parametric_transform) -> void 
        requires(std::is_base_of_v<ParametricCoordTransformation<T, ndim>, parametric_t>)
        {
            finalized = false;
            index_type idof = inv_selected_nodes[inode];
            if(idof != ndof()){
                parametric_accessors[idof] = std::make_unique<ParametricCoordTransformation<T, ndim>>(
                        std::move(parametric_transform));
                is_parametric[idof] = true;
            }
        }

        /// @brief the number of degrees of freedom represented
        auto ndof() const -> size_type { return selected_nodes.size(); }

        /// @brief put all data structures in usable state
        auto finalize() -> void {
            // compute the columns array;
            cols.clear();
            cols.reserve(ndof()+1);
            cols.push_back(0);
            for(int idof = 0; idof < ndof(); ++idof){
                if(is_parametric[idof]){
                    cols.push_back(cols[idof - 1] + parametric_accessors[idof]->s_size());
                } else {
                    cols.push_back(cols[idof - 1] + ndim);
                }
            }

            finalized = true;
        }


    };

    // Deduction guides
    template<std::ranges::forward_range R, typename T, int ndim>
    geo_dof_map(R&&, FESpace<T, std::ranges::range_value_t<R>, ndim>&) -> geo_dof_map<T, std::ranges::range_value_t<R>, ndim>;
    template<std::ranges::forward_range R, typename T, int ndim>
    geo_dof_map(R&&, FESpace<T, std::ranges::range_value_t<R>, ndim>&, bool) -> geo_dof_map<T, std::ranges::range_value_t<R>, ndim>;

    template<class T, class IDX, int ndim>
    struct geo_data_layout {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        const geo_dof_map<T, IDX, ndim>& geo_map;

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

        // =========
        // = Sizes =
        // =========

        /// @brief get the number of degrees of freedom
        [[nodiscard]] inline constexpr auto ndof() const noexcept -> size_type { return geo_map.ndof(); }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr auto nv(index_type idof) const noexcept -> size_type { 
            return geo_map.cols[idof + 1] - geo_map.cols[idof];
        }

        /// @brief the size of the compact index space
        [[nodiscard]] inline constexpr auto size() const noexcept -> size_type { return geo_map.cols[geo_map.ndof()]; }

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
            if(iv < 0    || iv >= nv(idof)  ) throw std::out_of_range("Vector compoenent index out of range");
            if(!geo_map.finalized) throw std::out_of_range("indices have not been finalized");
#endif
           return geo_map.cols[idof]; 
        }
    };

}
