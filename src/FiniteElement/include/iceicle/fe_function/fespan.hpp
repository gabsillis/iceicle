/**
 * @author Gianni Absillis
 * @brief a non-owning lightweight view for finite element data 
 * reminiscent of mdspan
 */
#pragma once
#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/fe_function/el_layout.hpp"
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include "iceicle/fe_function/trace_layout.hpp"
#include "iceicle/fe_function/node_set_layout.hpp"
#include "iceicle/fespace/fespace.hpp"
#include <cstdlib>
#include <ostream>
#include <ranges>
#include <set>
#include <span>
#include <ranges>
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
            using layout_type = LayoutPolicy;
            using accessor_type = AccessorPolicy;
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
     * @brief compute a vector scalar product and add to a vector 
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
     * @brief compute a vector scalar product and add to a scaled vector
     * y <= alpha * x + beta * y
     *
     * @param [in] alpha the scalar to multiply x by
     * @param [in] x the fespan to add 
     * @param [in/out] y the fespan to add to
     */
    template<typename T, class LayoutPolicyx, class LayoutPolicyy>
    void axpby(T alpha, const fespan<T, LayoutPolicyx> &x, T beta, fespan<T, LayoutPolicyy> y){
        if constexpr(std::is_same_v<LayoutPolicyy, LayoutPolicyx>) {
            // do in a single loop over the 1d index space 
            T *ydata = y.data();
            T *xdata = x.data();
            for(int i = 0; i < x.size(); ++i){
                ydata[i] = alpha * xdata[i] + beta * ydata[i];
            }
        } else {
            for(int ielem = 0; ielem < x.nelem(); ++ielem){
                for(int idof = 0; idof < x.ndof(ielem); ++idof){
                    for(int iv = 0; iv < x.nv(); ++iv){
                        y[ielem, idof, iv] = alpha * x[ielem, idof, iv] + beta * y[ielem, idof, iv];
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
     * @brief dofspan represents a non-owning view for the data over a set of degreees of freedom 
     * and vector components
     * This partitions the data with a 2 dimensional index space: the indices are 
     * idof - the index of the degree of freedom 
     *        a degree of freedom is a coefficent to a finite element basis function
     * iv - the index of the vector component
     * 
     * @tparam T the data type being stored 
     * @tparam LayoutPolicy policy for how the data is laid out in memory
     * @tparam AccessorPolicy policy to dispatch to when accessing data 
     */
    template<
        class T,
        class LayoutPolicy,
        class AccessorPolicy = default_accessor<T>
    >
    class dofspan{
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
            constexpr dofspan(pointer data, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(data), _layout{layout_args...}, _accessor{} 
            {}

            template<std::ranges::contiguous_range R, typename... LayoutArgsT>
            constexpr dofspan(R&& data_range, LayoutArgsT&&... layout_args) 
            noexcept : _ptr(std::ranges::data(data_range)), _layout{layout_args...}, _accessor{} 
            {}

            template<typename... LayoutArgsT>
            constexpr dofspan(pointer data, LayoutArgsT&&... layout_args, const AccessorPolicy &_accessor) 
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

            /** @brief get the static vector extent */
            [[nodiscard]] inline static constexpr std::size_t static_extent() noexcept { return LayoutPolicy::static_extent(); }

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
             *  this dofspan 
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
    dofspan(T * data, LayoutPolicy &) -> dofspan<T, LayoutPolicy>;
    template<typename T, class LayoutPolicy>
    dofspan(T * data, const LayoutPolicy &) -> dofspan<T, LayoutPolicy>;
    template<std::ranges::contiguous_range R, class LayoutPolicy>
    dofspan(R&&, LayoutPolicy &) -> dofspan<std::ranges::range_value_t<R>, LayoutPolicy>;

    // ================================
    // = dofspan concept restrictions =
    // ================================

    /**
     * @brief an elspan represents local dofs in an element in a compact layout.
     * and constrains the layout policy type to represent this 
     *
     * NOTE: This will also work with reference types 
     * though we discourage passing dofspan as a reference because it is a view
     */
    template<class spantype>
    concept elspan = std::same_as<
        std::remove_cv_t<std::remove_reference_t<spantype>>,
        dofspan<
            typename spantype::value_type,
            compact_layout_right<typename spantype::index_type, spantype::static_extent()>, 
            typename spantype::accessor_type 
        >
        /* || compact_layout_left dofspan*/
    >;

    /**
     * @brief a facspan represents local dofs over a face in a compact layout.
     * and constrains the layout policy type to represent this 
     *
     * NOTE: This will also work with reference types 
     * though we discourage passing dofspan as a reference because it is a view
     */
    template<class spantype>
    concept facspan = std::same_as<
        std::remove_cv_t<std::remove_reference_t<spantype>>,
        dofspan<
            typename spantype::value_type,
            trace_layout_right<typename spantype::index_type, spantype::static_extent()>, 
            typename spantype::accessor_type 
        >
        /* || trace_layout_left dofspan*/
    >;

    /**
     * @brief a facspan represents local dofs over a face in a compact layout.
     * and constrains the layout policy type to represent this 
     *
     * NOTE: This will also work with reference types 
     * though we discourage passing dofspan as a reference because it is a view
     */
    template<class spantype>
    concept node_selection_span = std::same_as<
        std::remove_cv_t<std::remove_reference_t<spantype>>,
        dofspan<
            typename spantype::value_type,
            node_selection_layout<typename spantype::index_type, spantype::static_extent()>, 
            typename spantype::accessor_type 
        >
        /* || trace_layout_left dofspan*/
    >;

    // ================
    // = Span Utility =
    // ================

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
        class GlobalAccessorPolicy
    > inline void extract_elspan(
        std::size_t iel,
        const fespan<T, GlobalLayoutPolicy, GlobalAccessorPolicy> fedata,
        elspan auto eldata
    ){
        // if the default accessor is used 
        // and the layout is block copyable 
        // this is the most efficient operation
        if constexpr (
            has_equivalent_el_layout<decltype(eldata), decltype(fedata)>::value 
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
        class GlobalLayoutPolicy
    > inline void scatter_elspan(
        std::size_t iel,
        T alpha,
        elspan auto eldata,
        T beta,
        fespan<T, GlobalLayoutPolicy> fedata
    ) {
        using index_type = decltype(fedata)::index_type;
        using local_accessor_policy = decltype(eldata)::accessor_type;

        if constexpr (
            has_equivalent_el_layout<decltype(eldata), decltype(fedata)>::value 
            && std::is_same<local_accessor_policy, default_accessor<T>>::value
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

    /**
        * @brief extract the data from a trace 
        * into a facspan from a global nodal data structure 
        * @param trace the trace to extract from 
        * @param global_data the data to extract from 
        * @param facdata the span to extract to 
        */
    template<
        class T,
        class LocalLayoutPolicy
    > inline void extract_facspan(
        ELEMENT::TraceSpace<T, typename LocalLayoutPolicy::index_type, LocalLayoutPolicy::static_extent()> &trace,
        FE::NodalFEFunction<T, LocalLayoutPolicy::static_extent()> &global_data,
        dofspan<T, LocalLayoutPolicy> facdata
    ) requires facspan<decltype(facdata)> {
        using index_type = LocalLayoutPolicy::index_type;
        for(index_type inode = 0; inode < trace.face->n_nodes();  ++inode){
            index_type ignode = trace.face->nodes()[inode];
            for(index_type iv = 0; iv < LocalLayoutPolicy::static_extent(); ++iv){
                facdata[inode, iv] = global_data[ignode][iv];
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
    * @param [in] iel the element index of the element data 
    * @param [in] alpha the multiplier for element data 
    * @param [in] facdata the fac local data 
    * @param [in] beta the multiplier for values in the global data array 
    * @param [in/out] global_data the global data array to scatter to 
    */
    template<
        class T,
        class LocalLayoutPolicy,
        class LocalAccessorPolicy
    > inline void scatter_facspan(
        ELEMENT::TraceSpace<T, typename LocalLayoutPolicy::index_type, LocalLayoutPolicy::static_extent()> &trace,
        T alpha, 
        dofspan<T, LocalLayoutPolicy, LocalAccessorPolicy> facdata,
        T beta,
        FE::NodalFEFunction<T, LocalLayoutPolicy::static_extent()> &global_data 
    ) requires facspan<decltype(facdata)> {
        using index_type = LocalLayoutPolicy::index_type;
        for(index_type inode = 0; inode < trace.face->n_nodes();  ++inode){
            index_type ignode = trace.face->nodes()[inode];
            for(index_type iv = 0; iv < facdata.nv(); ++iv){
                global_data[ignode][iv] = alpha * facdata[inode, iv]
                    + beta * global_data[ignode][iv];
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
        ELEMENT::TraceSpace<value_type, index_type, ndim> &trace,
        value_type alpha,
        facspan auto fac_data,
        value_type beta,
        node_selection_span auto global_data
    ) -> void {
        static_assert(std::is_same_v<index_type, typename decltype(fac_data)::index_type> , "index_types must match");
        static_assert(std::is_same_v<index_type, typename decltype(global_data)::index_type> , "index_types must match");
        static_assert(decltype(fac_data)::static_extent() == decltype(global_data)::static_extent(), "static_extents must match" );

        // node selection data structure 
        const nodeset_dof_map<index_type>& nodeset = global_data.get_layout().nodeset;

        // maps from full set of nodes -> restricted set of nodes
        const std::vector<index_type>& inv_selected_nodes = nodeset.inv_selected_nodes;

        for(index_type inode = 0; inode < trace.face->n_nodes(); ++inode){
            index_type ignode = inv_selected_nodes[trace.face->nodes()[inode]];
            for(index_type iv = 0; iv < fac_data.nv(); ++iv){
                global_data[ignode, iv] = alpha * fac_data[inode, iv]
                    + beta * global_data[ignode, iv];
            }
        }
    }

    /**
    * @brief perform a scatter operation to incorporate data from a 
    * restricted set of nodes to all nodes
    *
    * follows the y = alpha * x + beta * y convention
    * inspired by BLAS interface 
    *
    * @param [in] iel the element index of the element data 
    * @param [in] alpha the multiplier for element data 
    * @param [in] facdata the fac local data 
    * @param [in] beta the multiplier for values in the global data array 
    * @param [in/out] global_data the global data array to scatter to 
    */
    template<class value_type>
    inline auto scatter_node_selection_span(
        value_type alpha,
        node_selection_span auto node_selection_data,
        value_type beta, 
        FE::NodalFEFunction<value_type, decltype(node_selection_data)::static_extent()>& all_nodes_data
    ) -> void {
        using index_type = decltype(node_selection_data)::index_type;

        // the global node index of each dof in order
        const std::vector<index_type>& selected_nodes = node_selection_data.get_layout().nodeset.selected_nodes;

        for(index_type idof = 0; idof < node_selection_data.ndof(); ++idof){
            index_type ignode = selected_nodes[idof];
            if(ignode != selected_nodes.size()){ // safeguard against boundary nodes
                for(index_type iv = 0; iv < node_selection_data.nv(); ++iv){
                    all_nodes_data[ignode][iv] = alpha * node_selection_data[idof, iv]
                        + beta * all_nodes_data[ignode][iv];
                }
            }
        }
    }

    /**
     * @brief restrict the set of nodes to the nodes on interior faces 
     * that have a interface conservation vector norm above the given threshold 
     *
     * @return the nodeset based on the threshold
     */
    template<class T, class IDX, int ndim, class disc_type, class uLayout, class uAccessor, std::size_t vextent>
    auto select_nodeset(
        FE::FESpace<T, IDX, ndim> &fespace,              /// [in] the finite elment space
        disc_type disc,                                  /// [in] the discretization
        fespan<T, uLayout, uAccessor> u,                 /// [in] the current finite element solution
        T residual_threshold,                            /// [in] residual threshhold for selecting a trace
        std::integral_constant<std::size_t, vextent> nv  /// [in] the number of vector components for Interface Conservation
    ) -> nodeset_dof_map<IDX> {
        using index_type = IDX;
        using trace_type = FE::FESpace<T, IDX, ndim>::TraceType;

        nodeset_dof_map<IDX> nodeset{};

        // we will be filling the selected traces, nodes, 
        // and selected nodes -> gnode index map respectively
        std::vector<index_type>& selected_traces{nodeset.selected_traces};
        std::vector<index_type>& selected_nodes{nodeset.selected_nodes};
        std::vector<index_type>& inv_selected_nodes{nodeset.inv_selected_nodes};

        // helper array to keep track of which global node indices to select
        std::vector<bool> to_select(fespace.meshptr->nodes.n_nodes(), false);

        std::vector<T> res_storage{};
        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dg_map.max_el_size_reqirement(disc_type::dnv_comp);
        const std::size_t ncomp = disc_type::dnv_comp;
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);

        // loop over the traces and select traces and nodes based on IC residual
        for(const trace_type &trace : fespace.get_interior_traces()){
            // compact data views 
            dofspan uL{uL_storage.data(), u.create_element_layout(trace.elL.elidx)};
            dofspan uR{uR_storage.data(), u.create_element_layout(trace.elR.elidx)};

            // trace data view
            trace_layout_right<IDX, vextent> ic_res_layout{trace};
            res_storage.resize(ic_res_layout.size());
            dofspan ic_res{res_storage, ic_res_layout};

            // extract the compact values from the global u view 
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out and then get interface conservation
            ic_res = 0.0;
            disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, ic_res);

            std::cout << "Interface nr: " << trace.facidx; 
            std::cout << " | nodes:";
            for(index_type inode : trace.face->nodes_span()){
                std::cout << " " << inode;
            }
            std::cout << " | ic residual: " << ic_res.vector_norm() << std::endl; 

            // if interface conservation residual is high enough,
            // add the trace and nodes of the trace
            if(ic_res.vector_norm() > residual_threshold){
                selected_traces.push_back(trace.facidx);
                for(index_type inode : trace.face->nodes_span()){
                    to_select[inode] = true;
                }
            }
        }

        // loop over the boundary faces and deactivate all boundary nodes 
        // since some may be connected to an active interior face 
        for(const trace_type &trace : fespace.get_boundary_traces()){
            for(index_type inode : trace.face->nodes_span()){
                to_select[inode] = false;
            }
        }

        // finish setting up the map arrays

        // add all the selected nodes
        for(int ignode = 0; ignode < fespace.meshptr->nodes.n_nodes(); ++ignode){
            if(to_select[ignode]){
                selected_nodes.push_back(ignode);
            }
        }

        // default value for nodes that aren't selected is to map to selected_nodes.size()
        inv_selected_nodes = std::vector<index_type>(fespace.meshptr->nodes.n_nodes(), selected_nodes.size());
        for(int idof = 0; idof < selected_nodes.size(); ++idof){
            inv_selected_nodes[selected_nodes[idof]] = idof;
        }

        return nodeset;
    }
}
