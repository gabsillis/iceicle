add_library(iceicle_util INTERFACE)
target_include_directories(iceicle_util
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(iceicle_util INTERFACE cxx_std_20)
