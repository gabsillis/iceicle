# subdirectory src/FiniteElement

# create the library target for finite element features 
set(ICEICLE_FELIB_SOURCES mesh.cpp)
add_library(iceicle_fe ${ICEICLE_FELIB_SOURCES})
target_compile_features(iceicle_fe INTERFACE cxx_std_23)

# link libraries
target_link_libraries(iceicle_fe PUBLIC iceicle_util)
target_link_libraries(iceicle_fe PUBLIC nt_general)
target_link_libraries(iceicle_fe PUBLIC nt_tensor)
target_link_libraries(iceicle_fe PUBLIC nt_polynomial)
target_link_libraries(iceicle_fe PUBLIC nt_tmp)
target_link_libraries(iceicle_fe PUBLIC nt_geometry)
target_link_libraries(iceicle_fe PUBLIC ${MPI_C_LIBRARIES})
target_link_libraries(iceicle_fe PUBLIC ${MFEM_LIBRARIES})
target_link_libraries(iceicle_fe PUBLIC mdspan)

# OPTIONAL LIBRARIES
if(MPI_FOUND)
    target_compile_definitions(iceicle_fe PUBLIC ICEICLE_USE_MPI)
endif()
if(ICEICLE_USE_MFEM)
    target_include_directories(iceicle_fe PUBLIC ${MFEM_INCLUDE_DIRS})
    message("MFEM include dirs: ${MFEM_INCLUDE_DIRS}")
endif()
if(ICEICLE_USE_LUA)
    target_link_libraries(iceicle_fe PUBLIC ${LUA_LIBRARIES})
    target_include_directories(iceicle_fe PUBLIC ${LUA_INCLUDE_DIR})

    target_link_libraries(iceicle_fe PUBLIC sol2)
endif()
# include interface
target_include_directories(iceicle_fe PUBLIC ${MPI_INCLUDE_PATH})
target_include_directories(iceicle_fe PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

if(ICEICLE_USE_OPENGL)
    target_link_libraries(iceicle_fe PUBLIC OpenGL::OpenGL)
    target_link_libraries(iceicle_fe PUBLIC OpenGL::GLU)
    target_link_libraries(iceicle_fe PUBLIC GLEW::glew)
    target_link_libraries(iceicle_fe PUBLIC glfw)
endif()

