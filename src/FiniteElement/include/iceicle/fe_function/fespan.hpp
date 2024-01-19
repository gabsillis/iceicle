/**
 * @author Gianni Absillis
 * @brief a non-owning lightweight view for finite element data 
 * reminiscent of mdspan
 */
#pragma once
#include <cstdlib>
#include <type_traits>
#include <iceicle/fe_function/layout_enums.hpp>

namespace FE {
    
    /**
     * @brief implementation equivalent of std::default_accessor
     */
    template< class ElementType >
    class default_accessor {
        public:

        using offset_policy = default_accessor;
        using element_type = ElementType;
        using reference = ElementType&;
        using data_handle_type = ElementType*;

        constexpr reference access(data_handle_type p, std::size_t i) const noexcept {
            return p[i];
        } 

        constexpr data_handle_type offset(data_handle_type p, std::size_t i) const noexcept {
            return p + i;
        }
    };

    /**
     * @brief and fespan represents a non-owning view for finite element data 
     *
     * This partitions the data with a 3 dimensional index space: the indices are 
     * iel - the index of the finite element 
     * idof - the index of the local degree of freedom to the finite element 
     *        a degree of freedom is a coefficent to a finite element basis function
     * icomp - the index of the vector component
     * 
     * @tparam T the data type being stored 
     * @tparam LayoutPolicy policy for how the data is laid out in memory
     *         Layout Poicies need to implement the following functions 
     *         is_global - boolean that determines if this can access all elements data 
     *         is_nodal - boolean that determines if this is a CG data structure 
     *         operator()(fe_index) - indexing with the 3 indices above
     *         size() - the extent of the 1d index space
     * @tparam AccessorPolicy policy to dispatch to when accessing data 
     */
    template<
        class T,
        class LayoutPolicy,
        class AccessorPolicy = default_accessor<T>
    >
    class fespan {
        public:
            using pointer = T*;
            using const_pointer = const T*;
            using reference = AccessorPolicy::reference;
            using const_reference = const reference;

        private:
            /// The pointer to the data being accessed
            pointer __ptr;

            /// the layout policy
            LayoutPolicy __layout;

            /// the accessor policy
            AccessorPolicy __accessor;

        public:

            template<typename... LayoutArgsT>
            constexpr fespan(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : __ptr(data), __layout{layout_args...}, __accessor{} 
            {}

            template<typename... LayoutArgsT>
            constexpr fespan(pointer data, LayoutArgsT&&... layout_args, const AccessorPolicy &__accessor) 
            noexcept : __ptr(data), __layout{layout_args...}, __accessor{__accessor} 
            {}

            /** @brief get the upper bound of the 1D index space */
            constexpr std::size_t size() const noexcept { return __layout.size(); }

            /** @brief index into the data using a fe_index 
             * @param fe_index represents the element, dof, and vector component indices 
             * @return a reference to the data 
             */
            constexpr reference operator[](const fe_index &feidx) const {
                return __accessor.access(__ptr, __layout.operator()(feidx));
            }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr pointer data() noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return __ptr; }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr const pointer data() const noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return __ptr; }
    };


    /**
     * @brief elspan represents a non-owning view for the data of a single element
     * A Finite Element is a partition of the full domain with compact basis functions
     *
     * This partitions the data with a 2 dimensional index space: the indices are 
     * idof - the index of the local degree of freedom to the finite element 
     *        a degree of freedom is a coefficent to a finite element basis function
     * icomp - the index of the vector component
     * 
     * @tparam T the data type being stored 
     * @tparam LayoutPolicy policy for how the data is laid out in memory
     *         Layout Poicies need to implement the following functions 
     *         is_global - boolean that determines if this can access all elements data 
     *         is_nodal - boolean that determines if this is a CG data structure 
     *         operator()(fe_index) - indexing with the 3 indices above
     *         size() - the extent of the 1d index space
     * @tparam AccessorPolicy policy to dispatch to when accessing data 
     */
    template<
        class T,
        class LayoutPolicy,
        class AccessorPolicy = default_accessor<T>
    >
    class elspan{
        public:
            using pointer = T*;
            using const_pointer = const T*;
            using reference = AccessorPolicy::reference;
            using const_reference = const reference;

        private:
            /// The pointer to the data being accessed
            pointer __ptr;

            /// the layout policy
            LayoutPolicy __layout;

            /// the accessor policy
            AccessorPolicy __accessor;

        public:

            template<typename... LayoutArgsT>
            constexpr elspan(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : __ptr(data), __layout{layout_args...}, __accessor{} 
            {}

            template<typename... LayoutArgsT>
            constexpr elspan(pointer data, LayoutArgsT&&... layout_args, const AccessorPolicy &__accessor) 
            noexcept : __ptr(data), __layout{layout_args...}, __accessor{__accessor} 
            {}

            /** @brief get the upper bound of the 1D index space */
            constexpr std::size_t size() const noexcept { return __layout.size(); }

            /** @brief index into the data using a compact_index
             * @param idx the compact_index which represents a multidimensional index into element local data
             * @return a reference to the data 
             */
            constexpr reference operator[](const compact_index &idx) const {
                return __accessor.access(__ptr, __layout.operator()(idx));
            }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr pointer data() noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return __ptr; }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            constexpr const pointer data() const noexcept 
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return __ptr; }
    };
}
