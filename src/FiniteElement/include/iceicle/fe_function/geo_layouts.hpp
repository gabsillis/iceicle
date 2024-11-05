/// @brief data layouts for geometry degrees of freedom
/// @author Gianni Absillis (gabsill@ncsu.edu)
#pragma once
#include "iceicle/anomaly_log.hpp"
#include <type_traits>
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


    /// @brief a ParametricCoordTransformation defines a parametric transformation from 
    /// a a parametric coordinate space (s) to the physical coordinate space (x)
    template<class T, int ndim>
    class ParametricCoordTransformation {
        public:

        /// @brief virtual destructor
        virtual ~ParametricCoordTransformation() = default;

        /// @brief convert from the parametric coordinate space to the physical coordinate space
        virtual void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const = 0;

        /// @brief convert from the physical coordinate space to the parametric coordinate space
        virtual void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const = 0;

        /// @brief the dimensionality of the parametric coordinate space
        virtual auto s_size() const -> int = 0;

        /// @brief which indices in the parametric coordinate space are shared 
        /// between nodes 
        /// e.g. parameterize the top of a time slab by one parameter: t_f
        /// if coordinates are (x, y, t_f)
        ///                    (0, 1, 2  )
        /// then the shared indices vector would be {2}
        std::vector<int> shared_s_indices{};
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

    namespace parametric_transformations {

        /// @brief the identity parametric transformation (x = s)
        template<class T, int ndim>
        class Identity final : public ParametricCoordTransformation<T, ndim> {
            public:
            /// @brief convert from the parametric coordinate space to the physical coordinate space
            void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const override {
                std::ranges::copy(svec, xvec.begin());
            }

            /// @brief convert from the physical coordinate space to the parametric coordinate space
            void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const override {
                std::ranges::copy(xvec, svec.begin());
            }

            /// @brief the dimensionality of the parametric coordinate space
            auto s_size() const -> int override { return ndim; };
        };

        /// @brief fix a subset of the coordinates 
        template<class T, int ndim>
        class FixedCoordinateSubset final : public ParametricCoordTransformation<T, ndim> {
            /// @brief fixed coordinate index and value pairs
            std::vector<std::pair<int, T>> fixed_coordinates;

            public:
            FixedCoordinateSubset(std::vector<std::pair<int, T>> fixed_coordinates)
                : fixed_coordinates(fixed_coordinates) {};

            /// @brief convert from the parametric coordinate space to the physical coordinate space
            void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const override {
                auto fixed_iterator = fixed_coordinates.begin();
                auto s_iterator = svec.begin();

                for(int idim = 0; idim < ndim; ++idim){
                    if(fixed_iterator != fixed_coordinates.end() && fixed_iterator->first == idim){
                        xvec[idim] = fixed_iterator->second;
                        ++fixed_iterator;
                    } else {
                        xvec[idim] = *s_iterator;
                        ++s_iterator;
                    }
                }
            }

            /// @brief convert from the physical coordinate space to the parametric coordinate space
            void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const override {
                auto fixed_iterator = fixed_coordinates.begin();
                auto s_iterator = svec.begin();

                for(int idim = 0; idim < ndim; ++idim){
                    if(fixed_iterator != fixed_coordinates.end() && fixed_iterator->first == idim){
                        ++fixed_iterator;
                    } else {
                        *s_iterator = xvec[idim];
                        ++s_iterator;
                    }
                }
            }

            /// @brief the dimensionality of the parametric coordinate space
            auto s_size() const -> int override { return ndim - fixed_coordinates.size(); };

        };

        /// @brief fix a subset of the coordinates subject to bounds on coordinate values
        template<class T, int ndim>
        class BoundedFixedCoordinateSubset final : public ParametricCoordTransformation<T, ndim> {
            static constexpr std::size_t u_ndim = ndim; // unsigned
            /// @brief fixed coordinate index and value pairs
            std::vector<std::pair<int, T>> fixed_coordinates;
            std::array<T, u_ndim> xmin;
            std::array<T, u_ndim> xmax;

            public:
            BoundedFixedCoordinateSubset(std::vector<std::pair<int, T>> fixed_coordinates, std::array<T, u_ndim> xmin, std::array<T, u_ndim> xmax)
                : fixed_coordinates(fixed_coordinates), xmin{xmin}, xmax{xmax} {};

            /// @brief convert from the parametric coordinate space to the physical coordinate space
            void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const override {
                auto fixed_iterator = fixed_coordinates.begin();
                auto s_iterator = svec.begin();

                for(int idim = 0; idim < ndim; ++idim){
                    if(fixed_iterator != fixed_coordinates.end() && fixed_iterator->first == idim){
                        xvec[idim] = fixed_iterator->second;
                        ++fixed_iterator;
                    } else {
                        T val = *s_iterator;
                        val = std::max(xmin[idim], val);
                        val = std::min(xmax[idim], val);
                        xvec[idim] = val;
                        ++s_iterator;
                    }
                }
            }

            /// @brief convert from the physical coordinate space to the parametric coordinate space
            void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const override {
                auto fixed_iterator = fixed_coordinates.begin();
                auto s_iterator = svec.begin();

                for(int idim = 0; idim < ndim; ++idim){
                    if(fixed_iterator != fixed_coordinates.end() && fixed_iterator->first == idim){
                        ++fixed_iterator;
                    } else {
                        *s_iterator = xvec[idim];
                        ++s_iterator;
                    }
                }
            }

            /// @brief the dimensionality of the parametric coordinate space
            auto s_size() const -> int override { return ndim - fixed_coordinates.size(); };

        };

        /// @brief fix all the coordinates
        template<class T, int ndim>
        class Fixed final : public ParametricCoordTransformation<T, ndim> {
            /// @brief fixed coordinate index and value pairs
            std::array<T, ndim> fixed_coordinates;

            public:
            Fixed(std::array<T, ndim> fixed_coordinates)
                : fixed_coordinates(fixed_coordinates) {};

            /// @brief convert from the parametric coordinate space to the physical coordinate space
            void s_to_x(std::span<const T> svec, std::span<T, ndim> xvec) const override {
                std::ranges::copy(fixed_coordinates, xvec.begin());
            }

            /// @brief convert from the physical coordinate space to the parametric coordinate space
            void x_to_s(std::span<const T, ndim> xvec, std::span<T> svec) const override {
                // no-op
            }

            /// @brief the dimensionality of the parametric coordinate space
            auto s_size() const -> int override { return 0; };

        };
    }

    template<class T, class IDX, int ndim>
    struct geo_param_map {
        // ============
        // = Typedefs =
        // ============

        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;
        using fcn_list = 
            std::vector< 
                std::unique_ptr< ParametricCoordTransformation<T, ndim> >  
            >;

        // ================
        // = Data Members =
        // ================

        /// @brief a list of all the parametric functions
        fcn_list parametric_functions;

        /// the index of the parameterization in parametric_functions for each node
        std::vector< std::size_t > node_param_map;

        /// @brief true if the given node index is the identity parameterization
        std::vector<bool> is_identity;

        /// @brief for parametric_function
        /// a list of the shared parametric coordinate indices
        std::vector< std::vector< index_type > > shared_coord_param_map;

        /// @brief map of the 1D index for each set of idof (node index) 
        /// and iv (vector component index);
        util::crs<IDX> map;

        index_type n_shared_params;

        /// @brief 
        bool finalized = false;


        /// @brief register a set of nodes with a given parametric transformation 
        /// if there are shared parameters, these will be shared among all nodes in the list given 
        ///
        /// TODO: give a general form of this that takes pairs of {vector<index_type, parametric_ts>...}
        /// and shares parametric indices between all of the transformations given
        ///
        /// @param node_indices the node indices to apply the parameterization to 
        /// @param parametric_transform the transformation
        template<class parametric_t>
        auto register_parametric_nodes(std::vector<index_type> node_indices, parametric_t parametric_transform)
        -> void 
        requires(std::is_base_of_v<ParametricCoordTransformation<T, ndim>, parametric_t>)
        {
            finalized = false;

            // we will be adding one to the list so this size will be the new index
            std::size_t param_function_idx = parametric_functions.size();

            // check for shared parametric coordinates
            int n_shared = parametric_transform.shared_s_indices.size();
            std::vector< index_type > shared_coord_indices(n_shared);
            if(n_shared > 0) 
                std::iota(shared_coord_indices.begin(), shared_coord_indices.end(), n_shared_params);
            n_shared_params += n_shared;
            shared_coord_param_map.push_back(shared_coord_indices);

            // add the parametric function to the list
            parametric_functions.push_back(
                    std::make_unique<parametric_t>(std::move(parametric_transform)));

            // map all the nodes to this parametric function
            for(index_type inode : node_indices){
                const ParametricCoordTransformation<T, ndim>& previous_parameterization =
                    *(parametric_functions[node_param_map[inode]]);
                if(previous_parameterization.shared_s_indices.size() > 0){
                    util::AnomalyLog::log_anomaly(
                        "Previous parameterization had shared indices and was overwritten,"
                        "this could lead to redundant parameters");
                }
                is_identity[inode] = false;
                node_param_map[inode] = param_function_idx;
            }
        }

        auto finalize() -> void 
        {
            std::vector< std::vector<IDX> > map_ragged(node_param_map.size());

            index_type iparam = n_shared_params;
            for(index_type idof = 0; idof < node_param_map.size(); ++idof){
                const ParametricCoordTransformation<T, ndim>& parameterization =
                    *(parametric_functions[node_param_map[idof]]);
                // set the sizes for each node
                map_ragged[idof].resize(parameterization->s_size());
                std::ranges::fill(map_ragged[idof], (index_type) -1);

                // set the shared coordinate indices
                for(int ilshared = 0; ilshared < parameterization.shared_s_indices.size(); ++ilshared){
                    map_ragged[idof][parameterization.shared_s_indices[ilshared]] =
                        shared_coord_param_map[ilshared];
                }

                // set the nonshared indices
                for(int is = 0; is < parameterization.s_size(); ++is){
                    if(map_ragged[idof][is] == (index_type) -1){
                        map_ragged[idof][is] = iparam;
                        ++iparam;
                    }
                }
            }

            map = util::crs<IDX>(map_ragged);
            finalized = true;
        }

        [[nodiscard]] inline constexpr
        auto size() const noexcept -> size_type 
        { return map.nnz(); }
        
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

        /// store the parametric_function that represents this map
        std::vector< std::unique_ptr< ParametricCoordTransformation<T, ndim> > > parametric_accessors;

        /// @brief true if the given index is a parametric dof
        std::vector<bool> is_parametric;

        /// @brief flag that gets set to true when the data structures are all in usable state
        bool finalized = false;

        // ================
        // = Constructors =
        // ================

        geo_dof_map(std::ranges::forward_range auto trace_indices, FESpace<T, IDX, ndim>& fespace, bool remove_boundary_dofs = false)
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

            // initialize the parametric accessors to the identity accessor 
            parametric_accessors = std::vector< std::unique_ptr< ParametricCoordTransformation<T, ndim> > >(
                    selected_nodes.size());
            for(size_type idof = 0; idof < selected_nodes.size(); ++idof)
                parametric_accessors[idof] = std::make_unique<parametric_transformations::Identity<T, ndim>>();

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
                parametric_accessors[idof] = std::make_unique<parametric_t>(
                        std::move(parametric_transform));
                is_parametric[idof] = true;
            }
        }

        /// @brief the number of degrees of freedom represented
        auto ndof() const -> size_type { return selected_nodes.size(); }

        /// @brief the total number of components represented
        auto size() const -> size_type { return cols[ndof()]; }

        /// @brief put all data structures in usable state
        auto finalize() -> void {
            // compute the columns array;
            cols.clear();
            cols.reserve(ndof()+1);
            cols.push_back(0);
            for(int idof = 0; idof < ndof(); ++idof){
                if(is_parametric[idof]){
                    cols.push_back(cols[idof] + parametric_accessors[idof]->s_size());
                } else {
                    cols.push_back(cols[idof] + ndim);
                }
            }

            finalized = true;
        }
    };

    // Deduction guides
    template<std::ranges::forward_range R, typename T, typename IDX, int ndim>
    geo_dof_map(R&&, FESpace<T, IDX, ndim>&) -> geo_dof_map<T, IDX, ndim>;
    template<std::ranges::forward_range R, typename T, typename IDX, int ndim>
    geo_dof_map(R&&, FESpace<T, IDX, ndim>&, bool) -> geo_dof_map<T, IDX, ndim>;


    namespace mesh_parameterizations {

        /// @brief fix the boundary nodes to only be able to slide along the boundary 
        template<class T, class IDX, int ndim>
        auto hyper_rectangle(
                std::array<IDX, (std::size_t) ndim> nelem,
                std::array<T, (std::size_t) ndim> xmin,
                std::array<T, (std::size_t) ndim> xmax,
                geo_dof_map<T, IDX, ndim>& geo_map
        ) -> void 
        {
            std::array<IDX, ndim> nnode;
            std::ranges::transform(nelem, nnode.begin(), [](IDX el_count){ return el_count + 1; });

            // TODO: be more efficient than entire cartesian product
            for(IDX inode{0}; const std::array<IDX, ndim>& ijk : cartesian_index_product(nnode)){

                std::vector<std::pair<int, T>> fixed_coordinates{};
                for(int idim = 0; idim < ndim; ++idim){
                    if(ijk[idim] == 0)
                        fixed_coordinates.push_back(std::pair{idim, xmin[idim]});
                    else if(ijk[idim] == nelem[idim])
                        fixed_coordinates.push_back(std::pair{idim, xmax[idim]});
                }
                if(fixed_coordinates.size() != 0){
                    parametric_transformations::BoundedFixedCoordinateSubset<T, ndim> 
                        parameterization{fixed_coordinates, xmin, xmax};
                    geo_map.register_parametric_node(inode, parameterization);
                }
                ++inode;
            }

            geo_map.finalize();
        }

        /// @brief fix any nodes lying on a bounding box to stay on the given bounding box
        template<class T, class IDX, int ndim>
        auto bounding_box(
            AbstractMesh<T, IDX, ndim> &mesh,
            std::array<T, (std::size_t) ndim> xmin,
            std::array<T, (std::size_t) ndim> xmax,
            geo_dof_map<T, IDX, ndim>& geo_map
        ) -> void 
        {
            auto near = [](T a, T b) -> bool 
            { return std::abs(a - b) < 1e-8; };

            for(IDX inode = 0; inode < mesh.n_nodes(); ++inode){
                std::vector<std::pair<int, T>> fixed_coordinates{};
                for(int idim = 0; idim < ndim; ++idim){
                    if(near(mesh.coord[inode][idim], xmin[idim]) )
                        fixed_coordinates.push_back(std::pair{idim, xmin[idim]});
                    else if(near(mesh.coord[inode][idim], xmax[idim]))
                        fixed_coordinates.push_back(std::pair{idim, xmax[idim]});
                }
                if(fixed_coordinates.size() != 0){
                    parametric_transformations::BoundedFixedCoordinateSubset<T, ndim> 
                        parameterization{fixed_coordinates, xmin, xmax};
                    geo_map.register_parametric_node(inode, parameterization);
                }
            }
        }

        /// @brief fix all the nodes given in the list
        template<class T, class IDX, int ndim>
        auto fixed_nodelist(
            std::span<IDX> nodelist,
            AbstractMesh<T, IDX, ndim> &mesh,
            geo_dof_map<T, IDX, ndim>& geo_map
        ) -> void 
        {
            for(IDX inode : nodelist){
                std::array<T, ndim> pt;
                std::ranges::copy(mesh.coord[inode], pt.begin());
                parametric_transformations::Fixed<T, ndim> parameterization{pt};
                geo_map.register_parametric_node(inode, parameterization);
            }
        }
    }

    /// @brief layout that represents the layout of parametric geometric coordinate data in memory
    template<class T, class IDX, int ndim>
    struct geo_param_layout {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        const geo_param_map<T, IDX, ndim>& geo_map;

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
        [[nodiscard]] inline constexpr auto ndof() const noexcept -> size_type { return geo_map.map.nrow(); }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr auto nv(index_type idof) const noexcept -> size_type { 
            return geo_map.rowsize(idof);
        }

        /// @brief the size of the compact index space
        [[nodiscard]] inline constexpr auto size() const noexcept -> size_type { return geo_map.size(); }

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
            if(idof < 0  || idof >= std::max((size_type) 1, ndof())  ) throw std::out_of_range("Dof index out of range");
            if(iv < 0    || iv >= std::max((size_type) 1, nv(idof))  ) throw std::out_of_range("Vector compoenent index out of range");
            if(!geo_map.finalized) throw std::out_of_range("indices have not been finalized");
#endif
           return geo_map.map[idof, iv];
        }
    };

    /// @brief layout that represents the layout of parametric geometric coordinate data in memory
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
            if(idof < 0  || idof >= std::max((size_type) 1, ndof())  ) throw std::out_of_range("Dof index out of range");
            if(iv < 0    || iv >= std::max((size_type) 1, nv(idof))  ) throw std::out_of_range("Vector compoenent index out of range");
            if(!geo_map.finalized) throw std::out_of_range("indices have not been finalized");
#endif
           return geo_map.cols[idof] + iv; 
        }
    };

    /// @brief represents the layout of memeory for interface conservation residual
    template<class T, class IDX, int ndim, int _nv>
    struct ic_residual_layout {
        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        // ===========
        // = Members =
        // ===========

        /// @brief the mapping of selected degrees of freedom for MDG-ICE
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
        inline static constexpr auto local_dof_contiguous() noexcept -> bool { return false; }

        /// @brief this is purely dynamic extent
        inline static constexpr auto static_extent() { return _nv; }

        // =========
        // = Sizes =
        // =========

        /// @brief get the number of degrees of freedom
        [[nodiscard]] inline constexpr auto ndof() const noexcept -> size_type { return geo_map.ndof(); }

        /// @brief get the number of vector components
        [[nodiscard]] inline constexpr auto nv() const noexcept -> size_type { 
            return _nv;
        }

        /// @brief the size of the compact index space
        [[nodiscard]] inline constexpr auto size() const noexcept -> size_type 
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
            if(iv < 0    || iv >= nv()  ) throw std::out_of_range("Vector compoenent index out of range");
            if(!geo_map.finalized) throw std::out_of_range("indices have not been finalized");
#endif
           return idof * nv() + iv; 
        }
    };

}
