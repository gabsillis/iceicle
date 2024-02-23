
#include <gtest/gtest.h>
#include <petscsys.h>
#include "HYPRE_utilities.h"

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    HYPRE_Initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    HYPRE_Finalize();
    PetscFinalize();
    return result;
}
