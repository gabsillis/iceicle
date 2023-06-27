add_library(iceicle_math INTERFACE)
target_include_directories(iceicle_math
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(iceicle_math INTERFACE cxx_std_20)
