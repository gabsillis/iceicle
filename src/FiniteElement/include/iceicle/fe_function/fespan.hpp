/**
 * @author Gianni Absillis
 * @brief a non-owning lightweight view for finite element data 
 * reminiscent of mdspan
 */
#pragma once
#include "iceicle/fe_function/el_layout.hpp"
#include <cstdlib>
#include <ostream>
#include <span>
#include <format>
#include <cmath>
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
            // ============
            // = Typedefs =
            // ============
            using value_type = T;
            using pointer = AccessorPolicy::data_handle_type;
            using reference = AccessorPolicy::reference;
            using index_type = LayoutPolicy::index_type;
            using size_type = std::make_unsigned_t<index_type>;
            using dof_mapping_type = LayoutPolicy::dof_mapping_type;

        private:
            /// The pointer to the data being accessed
            pointer _ptr;

            /// the layout policy
            LayoutPolicy _layout;

            /// the accessor policy
            AccessorPolicy _accessor;

        public:

            constexpr fespan(pointer data, const dof_mapping_type &dof_map) 
            noexcept : _ptr(data), _layout{dof_map}, _accessor{} 
            {}

            constexpr fespan(pointer data, const dof_mapping_type &dof_map, const AccessorPolicy &_accessor) 
            noexcept : _ptr(data), _layout{dof_map}, _accessor{_accessor} 
            {}

            template<typename... LayoutArgsT>
            constexpr fespan(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(data), _layout{std::forward<LayoutArgsT>(layout_args)...}, _accessor{} 
            {}
            // ===================
            // = Size Operations =
            // ===================

            /** @brief get the upper bound of the 1D index space */
            constexpr size_type size() const noexcept { return _layout.size(); }

            /** @brief get the number of elements represented in the layout */
            [[nodiscard]] constexpr size_type nelem() const noexcept { return _layout.nelem(); }

            /** @brief get the number of degrees of freedom for a given element represented in the layout */
            [[nodiscard]] constexpr size_type ndof(index_type ielem) const noexcept { return _layout.ndof(ielem); }

            /** @brief get the number of vector components */
            [[nodiscard]] constexpr size_type nv() const noexcept { return _layout.nv(); }

            // ===============
            // = Data Access =
            // ===============

            /** @brief index into the data using a fe_index 
             * @param fe_index represents the element, dof, and vector component indices 
             * @return a reference to the data 
             */
            constexpr reference operator[](index_type ielem, index_type idof, index_type iv) const {
                return _accessor.access(_ptr, _layout.operator[](ielem, idof, iv));
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

            // ===========
            // = Utility =
            // ===========

            /**
             * @brief create an element local layout 
             * from the global layout and element index 
             * This can be used for gather and scatter operations
             * @param iel the element index 
             * @return the element layout 
             */
            constexpr auto create_element_layout(std::size_t iel){
                constexpr int nv_static = LayoutPolicy::static_extent();
                if constexpr (is_dynamic_size<nv_static>::value){
                    return compact_layout_right{
                        static_cast<decltype(LayoutPolicy::index_type)>(_layout.ndof()),
                        static_cast<decltype(LayoutPolicy::index_type)>(_layout.nv())
                    };
                } else {
                    return compact_layout_right{ 
                        static_cast<index_type>(_layout.ndof(iel)),
                        std::integral_constant<std::size_t, nv_static>{}
                    };

                }
            }

            /**
             * @brief set the value at every index 
             * in the index space to the value 
             * @return reference to this
             */
            constexpr fespan<T, LayoutPolicy, AccessorPolicy> &operator=( T value )
            {
                for(int i = 0; i < size(); ++i){
                    _ptr[i] = value;
                }
                return *this;
            }

            /** @brief get a const reference too the layout policy */
            constexpr const LayoutPolicy &get_layout() const { return _layout; }

            /**
             * @brief get the norm of the vector data components 
             * NOTE: this is not a finite element norm 
             *
             * @tparam order the lp polynomial order 
             * @return the vector lp norm 
             */
            template<int order = 2>
            constexpr T vector_norm(){

                T sum = 0;
                // TODO: be more specific about the index space
                // maybe by delegating to the LayoutPolicy
                for(int i = 0; i < size(); ++i){
                    sum += std::pow(_ptr[i], order);
                }
                
                if constexpr (order == 2){
                    return std::sqrt(sum);
                } else {
                    return std::pow(sum, 1.0 / order);
                }
            }
           
            /// @brief output to a stream all of the data 
            /// donote each element 
            /// each basis function gets its own row 
            /// vector components for each column
            friend std::ostream& operator<< ( std::ostream& os, const fespan<T, LayoutPolicy, AccessorPolicy>& fedata ) {
                constexpr int field_width = 20;
                constexpr int precision = 10;
                os << "=== Fespan Output ===" << std::endl;
                using index_type = LayoutPolicy::index_type;
                for(index_type ielem = 0; ielem < fedata.nelem(); ++ielem){
                    os << " - Element " << ielem << ":" << std::endl;
                    for(index_type idof = 0; idof < fedata.ndof(ielem); ++idof){
                        for(index_type iv = 0; iv < fedata.nv(); ++iv){
                            os << std::format("{:{}.{}e}", fedata[ielem, idof, iv], field_width, precision);
                        }
                        os << std::endl;
                    }
                }
                os << std::endl;
                return os;
            }
    };

    // deduction guides
    template<typename T, class LayoutPolicy>
    fespan(T *data, const LayoutPolicy &) -> fespan<T, LayoutPolicy>;

    /**
     * @brief cmpute a vector scalar product and add to a vector 
     * y <= alpha * x + y
     *
     * @param [in] alpha the scalar to multiply x by
     * @param [in] x the fespan to add 
     * @param [in/out] y the fespan to add to
     */
    template<typename T, class LayoutPolicyx, class LayoutPolicyy>
    void axpy(T alpha, const fespan<T, LayoutPolicyx> &x, fespan<T, LayoutPolicyy> y){
        if constexpr(std::is_same_v<LayoutPolicyy, LayoutPolicyx>) {
            // do in a single loop over the 1d index space 
            T *ydata = y.data();
            T *xdata = x.data();
            for(int i = 0; i < x.size(); ++i){
                ydata[i] += alpha * xdata[i];
            }
        } else {
            for(int ielem = 0; ielem < x.nelem(); ++ielem){
                for(int idof = 0; idof < x.ndof(ielem); ++idof){
                    for(int iv = 0; iv < x.nv(); ++iv){
                        y[ielem, idof, iv] += alpha * x[ielem, idof, iv];
                    }
                }
            }
        }
    }

    /**
     * @brief copy the data from fespan x to fespan y
     *
     * @param [in] x the fespan to copy from
     * @param [out] y the fespan to copy to
     */
    template<typename T, class LayoutPolicyx, class LayoutPolicyy>
    void copy_fespan(const fespan<T, LayoutPolicyx> &x, fespan<T, LayoutPolicyy> y){
        if constexpr(std::is_same_v<LayoutPolicyy, LayoutPolicyx>) {
            // do in a single loop over the 1d index space 
            std::copy_n(x.data(), x.size(), y.data());
        } else {
            // TODO: more assurances that x and y still share a space
            for(int ielem = 0; ielem < x.nelem(); ++ielem){
                for(int idof = 0; idof < x.ndof(ielem); ++idof){
                    for(int iv = 0; iv < x.nv(); ++iv){
                        y[ielem, idof, iv] = x[ielem, idof, iv];
                    }
                }
            }
        }
    }

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
            // ============
            // = Typedefs =
            // ============
            using value_type = T;
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
            constexpr elspan(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(data), _layout{layout_args...}, _accessor{} 
            {}

            template<typename... LayoutArgsT>
            constexpr elspan(pointer data, LayoutArgsT&&... layout_args, const AccessorPolicy &_accessor) 
            noexcept : _ptr(data), _layout{layout_args...}, _accessor{_accessor} 
            {}

            // ===================
            // = Size Operations =
            // ===================

            /** @brief get the upper bound of the 1D index space */
            constexpr size_type size() const noexcept { return _layout.size(); }

            /** @brief get the number of degrees of freedom for a given element represented in the layout */
            [[nodiscard]] constexpr size_type ndof() const noexcept { return _layout.ndof(); }

            /** @brief get the number of vector components */
            [[nodiscard]] constexpr size_type nv() const noexcept { return _layout.nv(); }

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


            // ===========
            // = Utility =
            // ===========

            /** @brief get the layout */
            constexpr inline LayoutPolicy &get_layout() { return _layout; }

            /**
             * @brief set the value at every index 
             * in the index space to the value 
             * @return reference to this
             */
            constexpr elspan<T, LayoutPolicy, AccessorPolicy> &operator=( T value )
            {
                // TODO: be more specific about the index space
                // maybe by delegating to the LayoutPolicy
                for(int i = 0; i < size(); ++i){
                    _ptr[i] = value;
                }
                return *this;
            }

            /**
             * @brief get the norm of the vector data components 
             * NOTE: this is not a finite element norm 
             *
             * @tparam order the lp polynomial order 
             * @return the vector lp norm 
             */
            template<int order = 2>
            constexpr T vector_norm(){

                T sum = 0;
                // TODO: be more specific about the index space
                // maybe by delegating to the LayoutPolicy
                for(int i = 0; i < size(); ++i){
                    sum += std::pow(_ptr[i], order);
                }
                
                if constexpr (order == 2){
                    return std::sqrt(sum);
                } else {
                    return std::pow(sum, 1.0 / order);
                }
            }

            /**
             * @brief contract this with another vector 
             * along the dof index 
             * @param [in] dof_vec the vector of degrees of freedom to contract with 
             *  usually basis function evaluations 
             * @param [out] eqn_out the values for each equation after contracting with dof_vec
             * WARNING: must be zero'd out
             */
            void contract_dofs(const T *__restrict__ dof_vec, T *__restrict__ eqn_out){
                for(std::size_t idof = 0; idof < ndof(); ++idof){
                    for(std::size_t iv = 0; iv < nv(); ++iv){
                        eqn_out[iv] += this->operator[](idof, iv) * dof_vec[idof];
                    }
                }
            }

            /**
             * @brief contract along the first index dimension with the dof index 
             * with an mdspan type object
             *
             * NOTE: zero's out result_data before use
             *
             * specialized for rank 2 and 3 mdspan objects
             * Assumes the first index into the mdspan is the dof index 
             *
             * TODO: currently asssumes all non-dof extents are static extents
             *
             * @param [in] dof_mdspan
             * @param [out] result_data pointer to memory 
             *      where the result of the contraction will be stored 
             *
             * @return an mdspan view of the result data with the contraction indices
             *  the first index of the contraction result will be the vector component index from 
             *  this elspan 
             *  the rest will be the indices from the dof_mdspan
             */
            template<class in_mdspan>
            auto contract_mdspan(const in_mdspan &dof_mdspan, T *result_data){
                static_assert(
                       in_mdspan::rank() == 1 
                    || in_mdspan::rank() == 2 
                    || in_mdspan::rank() == 3, 
                "only defined for ranks 2 and 3");
                // get the equation extent
                static constexpr std::size_t eq_static_extent = LayoutPolicy::static_extent();
                static constexpr std::size_t rank_dynamic = ((eq_static_extent == std::dynamic_extent) ? 1 : 0);

                // build up the array of dynamic extents
                std::array<int, rank_dynamic> dynamic_extents{};
                if constexpr(eq_static_extent == std::dynamic_extent) dynamic_extents[0] = nv();
//                int iarr = 1;
//                NUMTOOL::TMP::constexpr_for_range<1, dof_mdspan.rank()>([&]<int iextent>{
//                    if constexpr (dof_mdspan.static_extent(iextent) == std::dynamic_extent){
//                        dynamic_extents[iarr++] = dof_mdspan.extent(iextent);
//                    }
//                });
                if constexpr(in_mdspan::rank() == 1){
                    std::experimental::extents<int, eq_static_extent> result_extents{dynamic_extents};
                    std::experimental::mdspan eq_mdspan{result_data, result_extents};

                    // zero fill
                    std::fill_n(result_data, eq_mdspan.size(), 0.0);

                    for(int idof = 0; idof < ndof(); ++idof){
                        for(int iv = 0; iv < nv(); ++iv){
                            eq_mdspan[iv] +=
                                operator[](idof, iv) * dof_mdspan[idof];
                        }
                    }

                } else if constexpr(in_mdspan::rank() == 2){
                    static constexpr int ext_1 = in_mdspan::static_extent(1);

                    // set up the extents and construc the mdspan
                    std::experimental::extents<
                        int,
                        eq_static_extent,
                        ext_1
                    > result_extents{dynamic_extents};
                    std::experimental::mdspan eq_mdspan{result_data, result_extents};

                    // zero fill
                    std::fill_n(result_data, eq_mdspan.size(), 0.0);

                    // perform the contraction 
                    for(int idof = 0; idof < ndof(); ++idof){
                        for(int iv = 0; iv < nv(); ++iv){
                            for(int iext1 = 0; iext1 < ext_1; ++iext1){
                                eq_mdspan[iv, iext1] += 
                                    operator[](idof, iv) * dof_mdspan[idof, iext1];
                            }
                        }
                    }
                    return eq_mdspan;
                } else if constexpr (in_mdspan::rank() == 3) {

                    static constexpr int ext_1 = in_mdspan::static_extent(1);
                    static constexpr int ext_2 = in_mdspan::static_extent(2);

                    // set up the extents and construc the mdspan
                    std::experimental::extents<
                        int,
                        eq_static_extent,
                        ext_1,
                        ext_2
                    > result_extents{dynamic_extents};
                    std::experimental::mdspan eq_mdspan{result_data, result_extents};

                    // zero fill
                    std::fill_n(result_data, eq_mdspan.size(), 0.0);

                    // perform the contraction 
                    for(int idof = 0; idof < ndof(); ++idof){
                        for(int iv = 0; iv < nv(); ++iv){
                            for(int iext1 = 0; iext1 < ext_1; ++iext1){
                                for(int iext2 = 0; iext2 < ext_2; ++iext2){
                                    eq_mdspan[iv, iext1, iext2] += 
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

    // deduction guides
    template<typename T, class LayoutPolicy>
    elspan(T * data, LayoutPolicy &) -> elspan<T, LayoutPolicy>;
    template<typename T, class LayoutPolicy>
    elspan(T * data, const LayoutPolicy &) -> elspan<T, LayoutPolicy>;

    /**
     * @brief extract the data for a specific element 
     * from a global fespan 
     * to a local elspan 
     * @param iel the element index 
     * @param fedata the fespan to get data from
     * @param eldata the elspan to copy data to
     */
    template<
        class T,
        class GlobalLayoutPolicy,
        class GlobalAccessorPolicy,
        class LocalLayoutPolicy
    > inline void extract_elspan(
        std::size_t iel,
        const fespan<T, GlobalLayoutPolicy, GlobalAccessorPolicy> fedata,
        elspan<T, LocalLayoutPolicy> eldata
    ){
        // if the default accessor is used 
        // and the layout is block copyable 
        // this is the most efficient operation
        if constexpr (
            is_equivalent_el_layout<LocalLayoutPolicy, GlobalLayoutPolicy>::value 
            && std::is_same<GlobalAccessorPolicy, default_accessor<T>>::value
        ) {
            // memcopy/memmove poggers
            std::copy_n(
                &( fedata[fe_index{iel, 0, 0}] ),
                eldata.ndof() * eldata.nv(),
                eldata.data());
        } else {
            // NOTE: assuming extents are equivalent
            for(std::size_t idof = 0; idof < eldata.ndof(); ++idof){
                for(std::size_t iv = 0; iv < eldata.nv(); ++iv){
                    eldata[idof, iv] = fedata[iel, idof, iv];
                }
            }
        }
    }

    /**
     * @brief perform a scatter operation to incorporate element data 
     * back into the global data array 
     *
     * follows the y = alpha * x + beta * y convention
     * inspired by BLAS interface 
     *
     * @param [in] iel the element index of the element data 
     * @param [in] alpha the multiplier for element data 
     * @param [in] eldata the element local data 
     * @param [in] beta the multiplier for values in the global data array 
     * @param [in/out] fedata the global data array to scatter to 
     */
    template< 
        class T,
        class GlobalLayoutPolicy,
        class LocalLayoutPolicy,
        class LocalAccessorPolicy
    > inline void scatter_elspan(
        std::size_t iel,
        T alpha,
        elspan<T, LocalLayoutPolicy, LocalAccessorPolicy> eldata,
        T beta,
        fespan<T, GlobalLayoutPolicy> fedata
    ) {
        using index_type = LocalLayoutPolicy::index_type;

        if constexpr (
            is_equivalent_el_layout<LocalLayoutPolicy, GlobalLayoutPolicy>::value 
            && std::is_same<LocalAccessorPolicy, default_accessor<T>>::value
        ) {
            T *fe_data_block = &( fedata[fe_index{iel, 0, 0}] );
            T *el_data_block = eldata.data();
            std::size_t blocksize = eldata.extents().ndof * eldata.extents().nv;
            for(int i = 0; i < blocksize; ++i){
                fe_data_block[i] = alpha * el_data_block[i] + beta * fe_data_block[i];
            }
        } else {
            // NOTE: assuming extents are equivalent 
            for(index_type idof = 0; idof < eldata.ndof(); ++idof){
                for(index_type iv = 0; iv < eldata.nv(); ++iv){
                    fedata[iel, idof, iv] = 
                        alpha * eldata[idof, iv]
                        + beta * fedata[iel, idof, iv];
                }
            }
        }
    }
}
