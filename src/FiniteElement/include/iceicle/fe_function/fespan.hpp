/**
 * @author Gianni Absillis
 * @brief a non-owning lightweight view for finite element data 
 * reminiscent of mdspan
 */
#pragma once
#include "Numtool/tmp_flow_control.hpp"
#include <cstdlib>
#include <span>
#include <type_traits>
#include <iceicle/fe_function/layout_enums.hpp>
#include <mdspan/mdspan.hpp>

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

            /** @brief get the extents of the multidimensional index space */
            constexpr LayoutPolicy::extents_type extents() const noexcept { return __layout.extents(); }

            /** @brief index into the data using a compact_index
             * @param idx the compact_index which represents a multidimensional index into element local data
             * @return a reference to the data 
             */
            constexpr reference operator[](const compact_index &idx) const {
                return __accessor.access(__ptr, __layout.operator()(idx));
            }

            /** @brief index into the data using the set order
             * @param idof the degree of freedom index 
             * @param iv the vector index
             * @return a reference to the data 
             */
            constexpr reference operator[](std::size_t idof, std::size_t iv){
                return __accessor.access(__ptr, __layout.operator()(FE::compact_index{.idof = idof, .iv = iv}));
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

            /**
             * @brief contract this with another vector 
             * along the dof index 
             * @param [in] dof_vec the vector of degrees of freedom to contract with 
             *  usually basis function evaluations 
             * @param [out] eqn_out the values for each equation after contracting with dof_vec
             * WARNING: must be zero'd out
             */
            void contract_dofs(const T *__restrict__ dof_vec, T *__restrict__ eqn_out){
                compact_index_extents extents = __layout.extents();
                for(std::size_t idof = 0; idof < extents.ndof; ++idof){
                    for(std::size_t iv = 0; iv < extents.nv; ++iv){
                        eqn_out[iv] += this->operator[](compact_index{.idof = idof, .iv = iv}) * dof_vec[idof];
                    }
                }
            }

        public:

            /**
             * @brief contract along the first index dimension with the dof index 
             * with an mdspan type object
             *
             * specialized for rank 2 and 3 mdspan objects
             * Assumes the first index into the mdspan is the dof index 
             *
             * TODO: currently asssumes all non-dof extents are static extents
             *
             * @param [in] dof_mdspan
             * @param [out] result_data pointer to memory 
             *      where the result of the contraction will be stored 
             *      WARNING: must be zero'd out beforehand 
             *
             * @return an mdspan view of the result data with the contraction indices
             *  the first index of the contraction result will be the vector component index from 
             *  this elspan 
             *  the rest will be the indices from the dof_mdspan
             */
            template<class in_mdspan>
            auto contract_mdspan(const in_mdspan &dof_mdspan, T *result_data){
                static_assert(dof_mdspan.rank() == 2 || dof_mdspan.rank() == 3, "only defined for ranks 2 and 3");
                // get the equation extent
                static constexpr int eq_extent = (is_dynamic_ncomp<LayoutPolicy::extents_type::get_ncomp()>::value)
                ? std::dynamic_extent : LayoutPolicy::extents_type::get_ncomp();

                static constexpr int rank_dynamic = ((eq_extent == std::dynamic_extent) ? 1 : 0);

                // build up the array of dynamic extents
                std::array<int, rank_dynamic> dynamic_extents{};
                if constexpr(eq_extent == std::dynamic_extent) dynamic_extents[0] = extents().nv;
//                int iarr = 1;
//                NUMTOOL::TMP::constexpr_for_range<1, dof_mdspan.rank()>([&]<int iextent>{
//                    if constexpr (dof_mdspan.static_extent(iextent) == std::dynamic_extent){
//                        dynamic_extents[iarr++] = dof_mdspan.extent(iextent);
//                    }
//                });

                if constexpr(dof_mdspan.rank() == 2){
                    static constexpr int ext_1 = dof_mdspan.static_extent(1);

                    // set up the extents and construc the mdspan
                    std::experimental::extents<
                        int,
                        eq_extent,
                        ext_1
                    > result_extents{dynamic_extents};
                    std::experimental::mdspan eq_mdspan{result_data, result_extents};

                    // perform the contraction 
                    for(int idof = 0; idof < extents().ndof; ++idof){
                        for(int iv = 0; iv < extents().nv; ++iv){
                            for(int iext1 = 0; iext1 < ext_1; ++iext1){
                                eq_mdspan[iv, iext1] += 
                                    operator[](idof, iv) * dof_mdspan[idof, iext1];
                            }
                        }
                    }
                    return eq_mdspan;
                } else if constexpr (dof_mdspan.rank() == 3) {

                    static constexpr int ext_1 = dof_mdspan.static_extent(1);
                    static constexpr int ext_2 = dof_mdspan.static_extent(2);

                    // set up the extents and construc the mdspan
                    std::experimental::extents<
                        int,
                        eq_extent,
                        ext_1,
                        ext_2
                    > result_extents{dynamic_extents};
                    std::experimental::mdspan eq_mdspan{result_data, result_extents};

                    // perform the contraction 
                    for(int idof = 0; idof < extents().ndof; ++idof){
                        for(int iv = 0; iv < extents().nv; ++iv){
                            for(int iext1 = 0; iext1 < ext_1; ++iext1){
                                for(int iext2 = 0; iext2 < ext_2; ++iext2){
                                    eq_mdspan[iv, iext1] += 
                                        operator[](idof, iv) * dof_mdspan[idof, iext1, iext2];
                                }
                            }
                        }
                    }
                    return eq_mdspan;
                } else {

                    std::experimental::mdspan ret{result_data};
                    return ret;
                }
            }
    };
}
