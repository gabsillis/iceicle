add_library(iceicle_fe segment_mesh.cpp)
target_link_libraries(iceicle_fe PUBLIC iceicle_util)
target_link_libraries(iceicle_fe PUBLIC iceicle_math)
target_link_libraries(iceicle_fe PUBLIC ${MPI_C_LIBRARIES})
target_include_directories(iceicle_fe PUBLIC ${MPI_INCLUDE_PATH})
target_include_directories(iceicle_fe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(iceicle_fe PUBLIC cxx_std_20)

