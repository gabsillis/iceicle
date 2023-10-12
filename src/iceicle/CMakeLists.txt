add_executable(iceicle_1d iceicle_1d.cpp)
target_link_libraries(iceicle_1d iceicle_lib)
target_link_libraries(iceicle_1d ${MPI_CXX_LIBRARIES})
