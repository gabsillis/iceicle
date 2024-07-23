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

        template<class T, class IDX, class AccessorPolicy, int ndim>
        static constexpr bool is_icespan<component_span<T, ic_residual_layout<T, IDX, ndim>, AccessorPolicy>> = true;
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
            std::span<const T, ndim> xcoord{mesh.nodes[inode]};
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
            std::span<T, ndim> xcoord{mesh.nodes[inode]};
            if(geo_map.is_parametric[idof]){
                geo_map.parametric_accessors.at(idof)->s_to_x(geo_data.span_at_dof(idof), xcoord);
            } else {
                for(int idim = 0; idim < ndim; ++idim){
                    xcoord[idim] = geo_data[idof, idim];
                }
            }
        }

        // TODO: mesh validation and consistency operations
    }
}
