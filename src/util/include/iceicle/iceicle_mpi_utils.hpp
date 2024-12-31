// @brief compile macro protected calls to mpi utilities
#pragma once
#include <optional>
#include <ranges>
#ifdef ICEICLE_USE_MPI
#include <mpi.h>
#include <iceicle/mpi_type.hpp>
#endif
#include <utility>
namespace iceicle {

    /// @brief a parallel index pair 
    /// contains the rank where the item being indexed is 
    /// and the process-local index
    template< class IDX >
    struct p_index {
        IDX rank;
        IDX index;
    };

    namespace mpi {

        /// @brief broadcast a contiguous range using MPI_Bcast  
        /// After calling this the range on all processes will contain 
        /// the data of the "root"
        /// @param range the data to broadcast 
        /// @param root the process that we want to broadcast the data from
        template<std::ranges::contiguous_range R>
        inline 
        auto mpi_bcast_range(R&& range, int root){
            using value_type = std::ranges::range_value_t<R>;
#ifdef ICEICLE_USE_MPI 
            MPI_Bcast(std::ranges::data(range), std::ranges::size(range), 
                mpi_get_type<value_type>(), root, MPI_COMM_WORLD);
#endif
        }

        /// @brief check if mpi has been initialized
        inline
        auto mpi_initialized() -> bool 
        {
#ifdef ICEICLE_USE_MPI
            int initialized = (int) false;
            MPI_Initialized(&initialized);
            return static_cast<bool>(initialized);
#else 
            return false;
#endif
        }

        /// @brief execute the function fcn with arguments args only on rank irank
        template<class F, class... ArgsT>
        inline constexpr
        auto execute_on_rank(int irank, const F& fcn, ArgsT&&... args) -> void {
#ifdef ICEICLE_USE_MPI
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            if(myrank == irank){
                fcn(std::forward<ArgsT>(args)...);
            }
#else 
            fcn(std::forward<ArgsT>(args)...);
#endif
        }

        inline 
        auto mpi_world_rank() -> int 
        {
#ifdef ICEICLE_USE_MPI
            if(!mpi_initialized()) return 0;
            int myrank;
            MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
            return myrank;
#else 
            return 0;
#endif
        }

        inline 
        auto mpi_world_size() -> int 
        {
#ifdef ICEICLE_USE_MPI
            if(!mpi_initialized()) return 0;
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
#else 
            return 1;
#endif
        }

        inline 
        void mpi_sync()
        {
#ifdef ICEICLE_USE_MPI 
            MPI_Barrier(MPI_COMM_WORLD);
#endif
        }

        /// @brief mark that data only satisfies invariants on a given mpi_rank 
        template<class T>
        class on_rank {
            private:
            T _value;
            int rank;

            public:
            inline constexpr 
            on_rank(auto&& value, int rank) 
            : _value{value}, rank{rank} {}

            [[nodiscard]] inline constexpr 
            auto valid_rank() const -> int { return rank; }

            [[nodiscard]] inline constexpr 
            bool has_value() const { return mpi_world_rank() == rank; }

            [[nodiscard]] inline constexpr 
            operator bool() const { return has_value(); }

            [[nodiscard]] inline constexpr 
            auto value()& -> T&
            {
                if(has_value()){
                    return _value;
                } else {
                    throw std::bad_optional_access{};
                }
            }

            [[nodiscard]] inline constexpr 
            auto value()const& -> const T&
            {
                if(has_value()){
                    return _value;
                } else {
                    throw std::bad_optional_access{};
                }
            }

            [[nodiscard]] inline constexpr 
            auto value()&& -> T&&
            {
                if(has_value()){
                    return _value;
                } else {
                    throw std::bad_optional_access{};
                }
            }

            [[nodiscard]] inline constexpr 
            auto value()const&& -> const T&&
            {
                if(has_value()){
                    return _value;
                } else {
                    throw std::bad_optional_access{};
                }
            }

            template< class U >
            [[nodiscard]] inline constexpr 
            auto value_or(U&& other) const& -> T
            {
                if(has_value()){
                    return _value;
                } else {
                    return other;
                }

            }

            template< class U >
            [[nodiscard]] inline constexpr 
            auto value_or(U&& other)&& -> T&
            {
                if(has_value()){
                    return _value;
                } else {
                    return other;
                }

            }
        };

        template<class T>
        on_rank(T&& val, int) -> on_rank<std::remove_reference_t<T>>;
    }
}

