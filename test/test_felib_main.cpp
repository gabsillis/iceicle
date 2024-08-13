
#include <gtest/gtest.h>
#ifdef ICEICLE_USE_MPI
#include <mpi.h>
#endif

int main(int argc, char **argv){
#ifdef ICEICLE_USE_MPI
    MPI_Init(&argc, &argv);
#endif
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
#ifdef ICEICLE_USE_MPI
    MPI_Finalize();
#endif
    return result;
}
