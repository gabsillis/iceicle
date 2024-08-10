/// @brief owning view of data over a set of degrees of freedom
#include "iceicle/fe_function/fespan.hpp"
namespace iceicle {

    /**
     * @brief dofarray represents a owning view for the data over a set of degreees of freedom 
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
    class dofarray{
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
            constexpr dofarray(LayoutArgsT&&... layout_args) 
            noexcept : _layout{layout_args...}, _accessor{} 
            { _ptr = new T[size()]; }

            template<typename... LayoutArgsT>
            constexpr dofarray(LayoutArgsT&&... layout_args, const AccessorPolicy &_accessor) 
            noexcept : _layout{layout_args...}, _accessor{_accessor} 
            { _ptr = new T[size()]; }

            ~dofarray(){ delete[] _ptr; }

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

            /** @brief index into the data using the set order
             * @param idof the degree of freedom index 
             * @param iv the vector index
             * @return a reference to the data 
             */
            constexpr const reference operator[](index_type idof, index_type iv) const {
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
                return std::span{_ptr + _layout[idof, 0], _ptr + _layout[idof, 0] + _layout.nv()};
            }

            /**
             * @brief access the underlying data as a std::span 
             * at the given dof 
             * @param idof the degree of freedom to index at 
             * @return std::span over the vector component data at idof 
             */
            constexpr inline 
            auto span_at_dof(index_type idof) const -> std::span<const value_type> {
                return std::span{_ptr + _layout[idof, 0], _ptr + _layout[idof, 0] + _layout.nv()};
            }

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
            constexpr dofarray<T, LayoutPolicy, AccessorPolicy>& operator=( T value )
            {
                // TODO: be more specific about the index space
                // maybe by delegating to the LayoutPolicy
                for(int i = 0; i < size(); ++i){
                    _ptr[i] = value;
                }
                return *this;
            }


            /**
             * @brief add another dofarray with the same layout to this 
             */
            template<class otherAccessor>
            constexpr inline 
            auto operator+=(const dofarray<T, LayoutPolicy, otherAccessor>& other)
            -> dofarray<T, LayoutPolicy, AccessorPolicy>& {
                for(index_type idof = 0; idof < ndof(); ++idof){
                    for(index_type iv = 0; iv < nv(); ++iv){
                        operator[](idof, iv) += other[idof, iv];
                    }
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
             *  this dofarray 
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
}
