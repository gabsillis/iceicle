add_library(iceicle_fe INTERFACE)
target_link_libraries(iceicle_fe INTERFACE iceicle_math)
target_include_directories(iceicle_fe
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(iceicle_fe INTERFACE cxx_std_20)
