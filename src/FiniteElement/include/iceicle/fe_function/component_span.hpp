/// @brief non-owning view over selected vector components
/// @author Gianni Absillis (gabsill@ncsu.edu)
#pragma once
#include <iceicle/fe_function/fespan.hpp>
#include <iceicle/fe_function/geo_layouts.hpp>

namespace iceicle {

    template<
        class T,
        class LayoutPolicy,
        class AccessorPolicy = default_accessor<T>
    >
    class component_span {
        public:
            // ============
            // = Typedefs =
            // ============
            using value_type = T;
            using layout_type = LayoutPolicy;
            using accessor_type = AccessorPolicy;
            using pointer = AccessorPolicy::data_handle_type;
            using reference = AccessorPolicy::reference;
            using index_type = LayoutPolicy::index_type;
            using size_type = std::make_unsigned_t<index_type>;

        private:
            /// The pointer to the data being accessed
            pointer _ptr;

            /// the layout policy
            LayoutPolicy _layout;

            /// the accessor policy
            AccessorPolicy _accessor;

        public:

            template<typename... LayoutArgsT>
            constexpr component_span(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(data), _layout{layout_args...}, _accessor{} 
            {}

            template<std::ranges::contiguous_range R, typename... LayoutArgsT>
            constexpr component_span(R&& data_range, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(std::ranges::data(data_range)), _layout{std::forward<LayoutArgsT>(layout_args)...}, 
                     _accessor{} 
            {}

            template<typename... LayoutArgsT>
            constexpr component_span(pointer data, LayoutArgsT&&... layout_args, const AccessorPolicy &_accessor) 
            noexcept : _ptr(data), _layout{layout_args...}, _accessor{_accessor} 
            {}

            // ===================
            // = Size Operations =
            // ===================

            /** @brief get the upper bound of the 1D index space */
            [[nodiscard]] constexpr 
            auto size() const noexcept -> size_type { return _layout.size(); }

            /** @brief get the number of degrees of freedom for a given element represented in the layout */
            [[nodiscard]] constexpr 
            auto ndof() const noexcept -> size_type { return _layout.ndof(); }

            /** @brief get the number of vector components */
            [[nodiscard]] constexpr 
            auto nv(index_type idof) const noexcept -> size_type { return _layout.nv(idof); }

            // ===========
            // = Utility =
            // ===========

            /** @brief get the layout */
            constexpr inline LayoutPolicy& get_layout() { return _layout; }

            /**
             * @brief set the value at every index 
             * in the index space to the value 
             * @return reference to this
             */
            constexpr dofspan<T, LayoutPolicy, AccessorPolicy>& operator=( T value )
            {
                // TODO: be more specific about the index space
                // maybe by delegating to the LayoutPolicy
                for(int i = 0; i < size(); ++i){
                    _ptr[i] = value;
                }
                return *this;
            }

            // ===============
            // = Data Access =
            // ===============

            /** @brief index into the data using the set order
             * @param idof the degree of freedom index 
             * @param iv the vector index
             * @return a reference to the data 
             */
            constexpr reference operator[](index_type idof, index_type iv){
                return _accessor.access(_ptr, _layout[idof, iv]);
            }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr pointer data() noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return _ptr; }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr const pointer data() const noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return _ptr; }

            /**
             * @brief access the underlying data as a std::span 
             * at the given dof 
             * @param idof the degree of freedom to index at 
             * @return std::span over the vector component data at idof 
             */
            constexpr inline 
            auto span_at_dof(index_type idof) -> std::span<value_type> {
                return std::span{_ptr + _layout[idof, 0], _ptr + _layout[idof, 0] + _layout.nv(idof)};
            }

            /**
             * @brief access the underlying data as a std::span 
             * at the given dof 
             * @param idof the degree of freedom to index at 
             * @return std::span over the vector component data at idof 
             */
            constexpr inline 
            auto span_at_dof(index_type idof) const -> std::span<const value_type> {
                return std::span{_ptr + _layout[idof, 0], _ptr + _layout[idof, 0] + _layout.nv(idof)};
            }
    };


    // deduction guides
    template<typename T, class LayoutPolicy>
    component_span(T* data, LayoutPolicy &) -> component_span<T, LayoutPolicy>;
    template<typename T, class LayoutPolicy>
    component_span(T* data, const LayoutPolicy &) -> component_span<T, LayoutPolicy>;
    template<std::ranges::contiguous_range R, class LayoutPolicy>
    component_span(R&&, const LayoutPolicy&) -> component_span<std::ranges::range_value_t<R>, LayoutPolicy>;

    namespace impl {
        template<class spantype>
        static constexpr bool is_geospan = false;

        template<class T, class IDX, class AccessorPolicy, int ndim>
        static constexpr bool is_geospan<component_span<T, geo_data_layout<T, IDX, ndim>, AccessorPolicy>> = true; 

        template<class spantype>
        static constexpr bool is_icespan = false;

        template<class T, class IDX, class AccessorPolicy, int ndim, int _nv>
        static constexpr bool is_icespan<dofspan<T, ic_residual_layout<T, IDX, ndim, _nv>, AccessorPolicy>> = true;
    }

    /// @brief This span uses the geo_dof_map
    template<class spantype>
    concept geospan = impl::is_geospan<spantype>;

    /// @brief this span uses the ic_residual_layout
    template<class spantype>
    concept icespan = impl::is_icespan<spantype>;

    template<class T, class IDX, int ndim>
    void extract_geospan(AbstractMesh<T, IDX, ndim>& mesh, geospan auto geo_data){
        const geo_dof_map<T, IDX, ndim>& geo_map = geo_data.get_layout().geo_map;
        for(int idof = 0; idof < geo_data.ndof(); ++idof){
            IDX inode = geo_map.selected_nodes[idof];
            std::span<const T, ndim> xcoord{mesh.coord[inode]};
            if(geo_map.is_parametric[idof]){
                geo_map.parametric_accessors.at(idof)->x_to_s(xcoord, geo_data.span_at_dof(idof));
            } else {
                for(int idim = 0; idim < ndim; ++idim){
                    geo_data[idof, idim] = xcoord[idim];
                }
            }
        }
    }

    template<class T, class IDX, int ndim>
    void update_mesh(geospan auto geo_data, AbstractMesh<T, IDX, ndim>& mesh){
        const geo_dof_map<T, IDX, ndim>& geo_map = geo_data.get_layout().geo_map;
        for(int idof = 0; idof < geo_data.ndof(); ++idof){
            IDX inode = geo_map.selected_nodes[idof];
            std::span<T, ndim> xcoord{mesh.coord[inode]};
            if(geo_map.is_parametric[idof]){
                geo_map.parametric_accessors.at(idof)->s_to_x(geo_data.span_at_dof(idof), xcoord);
            } else {
                for(int idim = 0; idim < ndim; ++idim){
                    xcoord[idim] = geo_data[idof, idim];
                }
            }
        }

        // NOTE: moving the nodes requires updating the replciated per-element coordinates
        mesh.update_coord_els();
        // TODO: mesh validation and consistency operations
    }

    /**
     * @brief BLAS-like add scaled version of one dofspan to another with the same layout policy 
     * y <= y + alpha * x
     * @param [in] alpha the multipier for x
     * @param [in] x the component_span to add 
     * @param [in/out] y the component_span to add to
     */
    template<class T, class LayoutPolicy>
    auto axpy(T alpha, component_span<T, LayoutPolicy> x, component_span<T, LayoutPolicy> y) -> void 
    {
        using index_type = decltype(y)::index_type;
        for(index_type idof = 0; idof < x.ndof(); ++idof){
            for(index_type iv = 0; iv < x.nv(idof); ++iv){
                y[idof, iv] += alpha * x[idof, iv];
            }
        }
    }

    /**
    * @brief perform a scatter operation to incorporate face data 
    * back into the global data array 
    *
    * follows the y = alpha * x + beta * y convention
    * inspired by BLAS interface 
    *
    * @param [in] trace the trace who's data is being represented
    * @param [in] alpha the multiplier for element data 
    * @param [in] fac_data the fac local data 
    * @param [in] beta the multiplier for values in the global data array 
    * @param [in/out] global_data the global data array to scatter to 
    */
    template< class value_type, class index_type, int ndim>
    inline auto scatter_facspan(
        TraceSpace<value_type, index_type, ndim> &trace,
        value_type alpha,
        facspan auto fac_data,
        value_type beta,
        icespan auto global_data
    ) -> void {
        static_assert(std::is_same_v<index_type, typename decltype(fac_data)::index_type> , "index_types must match");
        static_assert(std::is_same_v<index_type, typename decltype(global_data)::index_type> , "index_types must match");
        static_assert(decltype(fac_data)::static_extent() == decltype(global_data)::static_extent(), "static_extents must match" );

        // node selection data structure 
        const geo_dof_map<value_type, index_type, ndim>& geo_map = global_data.get_layout().geo_map;

        // maps from full set of nodes -> restricted set of nodes
        const std::vector<index_type>& inv_selected_nodes = geo_map.inv_selected_nodes;

        for(index_type inode = 0; inode < trace.face->n_nodes(); ++inode){
            index_type ignode = inv_selected_nodes[trace.face->nodes()[inode]];
            if(ignode != geo_map.selected_nodes.size()){ // safeguard against boundary nodes
                for(index_type iv = 0; iv < fac_data.nv(); ++iv){
                    global_data[ignode, iv] = alpha * fac_data[inode, iv]
                        + beta * global_data[ignode, iv];
                }
            }
        }
    }
}
