
#include <gtest/gtest.h>
#include <petscsys.h>

int main(int argc, char **argv){
    PetscInitialize(&argc, &argv, nullptr, nullptr);
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    PetscFinalize();
    return result;
}
