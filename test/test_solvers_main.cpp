
#include <gtest/gtest.h>
#include "HYPRE_utilities.h"
#include "mpi.h"

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);
    HYPRE_Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    HYPRE_Finalize();
    MPI_Finalize();
    return result;
}
