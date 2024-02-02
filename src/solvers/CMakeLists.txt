add_library(iceicle_solvers INTERFACE)
target_link_libraries(iceicle_solvers INTERFACE iceicle_fe)


target_include_directories(iceicle_solvers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
