if(ICEICLE_USE_OPENGL)
    add_library(iceicle_opengl_utils load_shaders.cpp opengl_drawer.cpp)
    target_include_directories(iceicle_opengl_utils
        PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
    target_compile_features(iceicle_opengl_utils INTERFACE cxx_std_20)
endif()
add_library(iceicle_util INTERFACE)
target_include_directories(iceicle_util
    INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(iceicle_util INTERFACE cxx_std_20)
