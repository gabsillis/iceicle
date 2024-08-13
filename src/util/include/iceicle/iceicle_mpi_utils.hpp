// @brief compile macro protected calls to mpi utilities
#pragma once
#ifdef ICEICLE_USE_MPI
#include <mpi.h>
#endif
#include <utility>
namespace iceicle {
    namespace mpi {

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
            int size;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            return size;
#else 
            return 1;
#endif
        }
    }
}

