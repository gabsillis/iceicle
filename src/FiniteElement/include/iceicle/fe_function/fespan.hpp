/**
 * @author Gianni Absillis
 * @brief a non-owning lightweight view for finite element data 
 * reminiscent of mdspan
 */
#pragma once
#include "iceicle/anomaly_log.hpp"
#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/fe_function/el_layout.hpp"
#include "iceicle/fe_function/trace_layout.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/fe_function/node_set_layout.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include <mpi.h>
#include <ostream>
#include <ranges>
#include <span>
#include <ranges>
#include <fmt/core.h>
#include <cmath>
#include <type_traits>
#include <iceicle/fe_function/layout_enums.hpp>
#include <mdspan/mdspan.hpp>

namespace iceicle {
    
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

        constexpr reference access(data_handle_type p, std::size_t i) const noexcept 
        { return p[i]; } 

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
            using pointer = typename std::conditional<LayoutPolicy::read_only(), 
                  const typename AccessorPolicy::data_handle_type,
                  typename AccessorPolicy::data_handle_type>::type;
            using reference = typename std::conditional<LayoutPolicy::read_only(), 
                  const typename AccessorPolicy::reference,
                  typename AccessorPolicy::reference>::type;
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

            template<std::ranges::contiguous_range R>
            constexpr fespan(R&& data_range, const LayoutPolicy &dof_map)
            noexcept : _ptr(std::ranges::data(data_range)), _layout{dof_map}, _accessor{}
            {
                static_assert(std::is_same_v<std::ranges::range_value_t<decltype(data_range)>, T>, "value type must match");
                T sz = std::ranges::size(data_range);
                if(sz < dof_map.size()){
                    util::AnomalyLog::log_anomaly(util::Anomaly{
                        "Provided data range cannot support the extent of the layout",
                        util::general_anomaly_tag{}});
                }
            }

            constexpr fespan(
                    std::ranges::contiguous_range auto data_range,
                    const LayoutPolicy &dof_map,
                    const AccessorPolicy &_accessor
            ) noexcept : _ptr(std::ranges::data(data_range)),
                      _layout{dof_map}, _accessor{_accessor}
            {
                static_assert(std::is_same_v<std::ranges::range_value_t<decltype(data_range)>, T>, "value type must match");
                util::AnomalyLog::check(std::ranges::size(data_range) < dof_map.size(),
                    util::Anomaly{"Provided data range cannot support the extent of the layout", util::general_anomaly_tag{}});
            }

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

            /** @brief get the number of global degrees of freedom */
            [[nodiscard]] constexpr size_type ndof() const noexcept { return _layout.ndof(); }

            /** @brief get the number of vector components */
            [[nodiscard]] constexpr size_type nv() const noexcept { return _layout.nv(); }

            /** @brief get the static vector extent */
            [[nodiscard]] inline static constexpr std::size_t static_extent() noexcept { return LayoutPolicy::static_extent(); }

            // ====================
            // = Index Operations =
            // ====================

            /** @brief given a process-local global degree of freedom, get the parallel dof index */
            [[nodiscard]] inline constexpr 
            auto get_pindex(index_type igdof) const noexcept 
            -> index_type 
            { return _layout.dof_partitioning.p_indices[igdof]; }

            /** @brief given the parallel dof index, get the mpi rank that owns this index */
            [[nodiscard]] inline constexpr 
            auto owning_rank(index_type pdof) const noexcept
            -> int 
            { return _layout.dof_partitioning.owning_rank(pdof); }

            //** @brief get the number of dofs owned by this process */
            [[nodiscard]] inline constexpr 
            auto owned_ndof(mpi::communicator_type comm) const noexcept 
            -> size_type
            { return _layout.dof_partitioning.owned_range_size(mpi::rank(comm)); }

            // ===============
            // = Data Access =
            // ===============

            /** 
             * @brief index into the data using a finite element index triple 
             * the element index, the element local dof index, the vector component index
             * @param ielem the element index 
             * @param idof the element local dof index
             * @param iv the vector component index
             * @return a reference to the data at the given index triple
             */
            [[nodiscard]] inline constexpr
            auto operator[](index_type ielem, index_type idof, index_type iv) const
            -> reference
            { return _accessor.access(_ptr, _layout.operator[](ielem, idof, iv)); }

            /** 
             * @brief index into the data using a index pair
             * the global dof index, the vector component index
             * @param igdof the element local dof index
             * @param iv the vector component index
             * @return a reference to the data at the given index pair
             */
            [[nodiscard]] inline constexpr
            auto operator[](index_type igdof, index_type iv) const
            -> reference
            { return _accessor.access(_ptr, _layout.operator[](igdof, iv)); }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            [[nodiscard]] inline constexpr 
            auto data() const noexcept 
            -> pointer
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return _ptr; }

            // ===========
            // = Utility =
            // ===========

            /**
             * @brief synchronize data from the owning ranks 
             * after this call all data at each parallel dof index should match the value 
             * on the owning rank for that parallel dof index
             * @param comm the mpi communicator
             */
            inline constexpr
            auto sync_mpi(
                mpi::communicator_type comm = mpi::comm_world
            ) -> void
            {
                const pindex_map<index_type> &dof_partitioning = _layout.dof_partitioning;
                
                int nrank = mpi::size(comm), myrank = mpi::rank(comm);

                // for each mpi rank, list the dofs we need to recieve
                std::vector< std::vector< index_type > > to_recieve(nrank);
                // start loop after owned range
                for(index_type lindex = dof_partitioning.owned_range_size(myrank); 
                        lindex < ndof(); ++lindex) {
                    index_type pindex = dof_partitioning.p_indices[lindex];
                    to_recieve[dof_partitioning.owning_rank(pindex)].push_back(pindex);
                }

                // for each mpi rank, list the dofs we need to send
                std::vector< std::vector< index_type > > to_send(nrank);
                std::vector< std::vector< T > > send_data(nrank);

                std::vector<MPI_Request> requests;
                for(int irank = 0; irank < nrank; ++irank){
                    if(irank != myrank){ 
                        requests.emplace_back();
                        MPI_Isend(to_recieve[irank].data(), to_recieve[irank].size(), 
                                mpi_get_type(to_recieve[irank].data()), irank, 0, comm, &requests.back());
                    }
                }

                for(int irank = 0; irank < nrank; ++irank){
                    if(irank != myrank){
                        MPI_Status status;
                        MPI_Probe(irank, 0, comm, &status);
                        int recv_sz;
                        MPI_Get_count(&status, mpi_get_type<index_type>(), &recv_sz);
                        to_send[irank].resize(recv_sz);
                        MPI_Recv(to_send[irank].data(), recv_sz, mpi_get_type<index_type>(), 
                                irank, 0, comm, MPI_STATUS_IGNORE);
                    }
                }

                // wait for isends
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                requests.clear();

                // build the data vectors to send
                for(int irank = 0; irank < nrank; ++irank){
                    if(irank != myrank){
                        send_data[irank].reserve(to_send[irank].size());
                        for(index_type pidx : to_send[irank]){
                            index_type igdof = dof_partitioning.inv_p_indices.at(pidx);
                            for(int iv = 0; iv < nv(); ++iv)
                                send_data[irank].push_back(operator[](igdof, iv));
                        }

                        requests.emplace_back();
                        MPI_Isend(send_data[irank].data(), send_data[irank].size(),
                                mpi_get_type(send_data[irank].data()), 
                                irank, 1, comm, &requests.back());
                    }
                }

                for(int irank = 0; irank < nrank; ++irank){
                    if(irank != myrank){
                        MPI_Status status;
                        MPI_Probe(irank, 1, comm, &status);
                        int recv_sz;
                        MPI_Get_count(&status, mpi_get_type<index_type>(), &recv_sz);
                        std::vector<T> recv_data(recv_sz);
                        MPI_Recv(recv_data.data(), recv_sz, mpi_get_type<T>(), 
                                irank, 1, comm, MPI_STATUS_IGNORE);

                        auto recieve_it = recv_data.begin();
                        for(index_type pidx : to_recieve[irank]){
                            index_type igdof = dof_partitioning.inv_p_indices.at(pidx);
                            for(int iv = 0; iv < nv(); ++iv){
                                operator[](igdof, iv) = *recieve_it;
                                ++recieve_it;
                            }
                        }
                    }
                }

                // wait for isends
                MPI_Waitall(requests.size(), requests.data(), MPI_STATUSES_IGNORE);
                requests.clear();

            }

            /**
             * @brief create an element local layout 
             * from the global layout and element index 
             * This can be used for gather and scatter operations
             * @param iel the element index 
             * @return the element layout 
             */
            constexpr auto create_element_layout(std::size_t iel) const {
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
            requires(!LayoutPolicy::read_only())
            {
                for(int i = 0; i < size(); ++i){
                    _ptr[i] = value;
                }
                return *this;
            }

            /** @brief get a const reference to the layout policy */
            constexpr const LayoutPolicy& get_layout() const { return _layout; }

            /** @brief get a const reference to the accessor policy */
            constexpr const AccessorPolicy& get_accessor() const { return _accessor; }

            /**
             * @brief get the norm of the vector data components 
             * NOTE: this is not a finite element norm 
             *
             * @tparam order the p polynomial order 
             * @return the vector L^p norm 
             */
            template<int order = 2>
            constexpr T vector_norm(mpi::communicator_type comm = mpi::comm_world){

                T sum = 0;
                for(index_type idof = 0; idof < owned_ndof(comm); ++idof){
                    for(index_type iv = 0; iv < nv(); ++iv){
                        sum += std::pow(operator[](idof, iv), order);
                    }
                }
#ifdef ICEICLE_USE_MPI
                MPI_Allreduce(MPI_IN_PLACE, &sum, 1, mpi_get_type(sum), MPI_SUM, comm);
#endif
                
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
                            os << fmt::format("{:{}.{}e}", fedata[ielem, idof, iv], field_width, precision);
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

    template<std::ranges::contiguous_range R, class LayoutPolicy>
    fespan(R&& data_range, const LayoutPolicy &) -> fespan< std::ranges::range_value_t<R>, LayoutPolicy >;

    template<std::ranges::contiguous_range R, class LayoutPolicy, class AccessorPolicy>
    fespan(R&& data_range, const LayoutPolicy &, const AccessorPolicy&)
    -> fespan< std::ranges::range_value_t<R>, LayoutPolicy, AccessorPolicy>;

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
     * @brief cast the view u, to one that excludes the additional dofs from interprocess ghost elements 
     * This ensures that the span meets the invariants enforced for writeability
     * @param u the data view to cast 
     * @return a new fespan with over the dofs without the interprocess ghost elements
     */
    template<class T, class LayoutPolicy, class AccessorPolicy>
    auto exclude_ghost(fespan<T, LayoutPolicy, AccessorPolicy> u)
    { 
        return fespan{std::span{u.data(), u.data() + u.size()},
                exclude_ghost(u.get_layout()), u.get_accessor()}; 
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
            noexcept : _ptr(std::ranges::data(data_range)), _layout{std::forward<LayoutArgsT>(layout_args)...}, 
                     _accessor{} 
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
            [[nodiscard]] inline constexpr 
            auto operator[](index_type idof, index_type iv) const 
            -> reference
            { return _accessor.access(_ptr, _layout[idof, iv]); }

            /**
             * @brief if using the default accessor, allow access to the underlying storage
             * @return the underlying storage 
             */
            [[nodiscard]] inline constexpr 
            auto data() const noexcept 
            -> pointer
            requires(std::is_same_v<AccessorPolicy, default_accessor<T>>) 
            { return _ptr; }

            /**
             * @brief access the underlying data as a std::span 
             * at the given dof 
             * @param idof the degree of freedom to index at 
             * @return std::span over the vector component data at idof 
             */
            [[nodiscard]] constexpr inline 
            auto span_at_dof(index_type idof)
            -> std::span<value_type>
            { return std::span{_ptr + _layout[idof, 0], _ptr + _layout[idof, 0] + _layout.nv()}; }

            /**
             * @brief get the equivalent 1D index of the multi-dimensional indices 
             * @return the 1D index determined by the layout 
             */
            [[nodiscard]] constexpr inline 
            auto index_1d(index_type idof, index_type iv) const 
            -> index_type 
            { return _layout[idof, iv]; }

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
             * @brief add another dofspan with the same layout to this 
             */
            template<class otherAccessor>
            constexpr inline 
            auto operator+=(const dofspan<T, LayoutPolicy, otherAccessor>& other)
            -> dofspan<T, LayoutPolicy, AccessorPolicy>& 
            {
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

                    // set up the extents and construct the mdspan
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

                    // set up the extents and construct the mdspan
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

                    std::mdspan ret{result_data};
                    return ret;
                }
            }
    };

    // deduction guides
    template<typename T, class LayoutPolicy>
    dofspan(T* data, LayoutPolicy &) -> dofspan<T, LayoutPolicy>;
    template<typename T, class LayoutPolicy>
    dofspan(T* data, const LayoutPolicy &) -> dofspan<T, LayoutPolicy>;
    template<std::ranges::contiguous_range R, class LayoutPolicy>
    dofspan(R&&, const LayoutPolicy&) -> dofspan<std::ranges::range_value_t<R>, LayoutPolicy>;

    // forward declaration
    template<
        class T,
        class LayoutPolicy,
        class AccessorPolicy
    >
    class dofarray;

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
    > || std::same_as<
        std::remove_cv_t<std::remove_reference_t<spantype>>,
        dofarray<
            typename spantype::value_type,
            compact_layout_right<typename spantype::index_type, spantype::static_extent()>, 
            typename spantype::accessor_type 
        >
    >;
    /* || compact_layout_left dofspan*/

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

    template<class T, class IDX, std::size_t vextent>
    using simple_dof_span = dofspan<T, dof_layout_right<IDX, vextent>>;

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
    
    /// @brief from a view over finite element data get a view per dof 
    /// of the same data 
    /// WARNING: data must not be strided (currently not a thing)
    ///
    /// @tparam T the data type 
    /// @tparam LayoutPolicyFespan the LayoutPolicy of the fespan 
    ///
    /// @param fedata the fespan 
    /// @return a simple_dof_span over the same data
    template<class T, class LayoutPolicyFespan>
    auto dof_view(fespan<T, LayoutPolicyFespan> fedata)
    -> simple_dof_span<T, typename LayoutPolicyFespan::index_type, LayoutPolicyFespan::static_extent()>
    {
        static constexpr std::size_t vextent = LayoutPolicyFespan::static_extent();
        using index_type = LayoutPolicyFespan::index_type;
        // compute the number of degrees of freedom
        std::size_t ndof = fedata.size() / fedata.nv();
        return simple_dof_span<T, index_type, vextent> 
            {fedata.data(), dof_layout_right<index_type, vextent>{ndof}};
    }

    /// @brief copy the data from dofspan a to dofspan b 
    /// NOTE: not bounds checked in release mode
    ///
    /// @param a the dofspan to copy from 
    /// @parma b the dofspan to copy to
    template<typename T, class LayoutPolicyA, class LayoutPolicyB>
    auto copy(dofspan<T, LayoutPolicyA> a, dofspan<T, LayoutPolicyB> b) -> void {
        using index_type = decltype(a)::index_type;
        for(index_type idof = 0; idof < a.ndof(); ++idof){
            for(index_type iv = 0; iv < a.nv(); ++iv){
                b[idof, iv] = a[idof, iv];
            }
        }
    }
   
    /**
     * @brief BLAS-like add scaled version of one dofspan to another with the same layout policy 
     * y <= y + alpha * x
     * @param [in] alpha the multipier for x
     * @param [in] x the dofspan to add 
     * @param [in/out] y the dofspan to add to
     */
    template<typename T, class LayoutPolicy>
    auto axpy(T alpha, dofspan<T, LayoutPolicy> x, dofspan<T, LayoutPolicy> y) -> void {
        using index_type = decltype(y)::index_type;
        for(index_type idof = 0; idof < x.ndof(); ++idof){
            for(index_type iv = 0; iv < x.nv(); ++iv){
                y[idof, iv] += alpha * x[idof, iv];
            }
        }
    }

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
        TraceSpace<T, typename LocalLayoutPolicy::index_type, LocalLayoutPolicy::static_extent()> &trace,
        NodeArray<T, LocalLayoutPolicy::static_extent()> &global_data,
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
        TraceSpace<T, typename LocalLayoutPolicy::index_type, LocalLayoutPolicy::static_extent()> &trace,
        T alpha, 
        dofspan<T, LocalLayoutPolicy, LocalAccessorPolicy> facdata,
        T beta,
        NodeArray<T, LocalLayoutPolicy::static_extent()> &global_data 
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
        TraceSpace<value_type, index_type, ndim> &trace,
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
            if(ignode != nodeset.selected_nodes.size()){ // safeguard against boundary nodes
                for(index_type iv = 0; iv < fac_data.nv(); ++iv){
                    global_data[ignode, iv] = alpha * fac_data[inode, iv]
                        + beta * global_data[ignode, iv];
                }
            }
        }
    }

    /**
     * @brief get the node coordinates into a node selection data 
     * @param [in] all_nodes_data the node coordinates 
     * @param [out] node_selection_data the selected coordinates
     */
    template<class T, int ndim>
    inline auto extract_node_selection_span(
        const NodeArray<T, ndim>& all_nodes_data,
        node_selection_span auto node_selection_data
    ) -> void {

        using index_type = decltype(node_selection_data)::index_type;

        // the global node index of each dof in order
        const std::vector<index_type>& selected_nodes = node_selection_data.get_layout().nodeset.selected_nodes;

        for(index_type idof = 0; idof < node_selection_data.ndof(); ++idof){
            index_type ignode = selected_nodes[idof];
            for(index_type iv = 0; iv < ndim; ++iv){ // TODO: bounds check against nv()
                node_selection_data[idof, iv] = all_nodes_data[ignode][iv];
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
        NodeArray<value_type, decltype(node_selection_data)::static_extent()>& all_nodes_data
    ) -> void {
        using index_type = decltype(node_selection_data)::index_type;

        // the global node index of each dof in order
        const std::vector<index_type>& selected_nodes = node_selection_data.get_layout().nodeset.selected_nodes;

        for(index_type idof = 0; idof < node_selection_data.ndof(); ++idof){
            index_type ignode = selected_nodes[idof];
            for(index_type iv = 0; iv < node_selection_data.nv(); ++iv){
                all_nodes_data[ignode][iv] = alpha * node_selection_data[idof, iv]
                    + beta * all_nodes_data[ignode][iv];
            }
        }
    }
}
