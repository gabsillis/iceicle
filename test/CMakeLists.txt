if(ICEICLE_USE_MATPLOTPLUSPLUS)
    add_executable(simplex_trans_ex SimplexTransformExample.cpp)
    target_link_libraries(simplex_trans_ex PUBLIC matplot)
    target_link_libraries(simplex_trans_ex PUBLIC iceicle_fe) 
endif()

if(ICEICLE_USE_OPENGL)
    SET(IMGUI_SOURCES 
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    add_executable(element_visualizer element_visualizer.cpp ${IMGUI_SOURCES})
    target_link_libraries(element_visualizer PUBLIC iceicle_fe)
    target_link_libraries(element_visualizer PUBLIC iceicle_opengl_utils)
    target_include_directories(element_visualizer PUBLIC ${imgui_SOURCE_DIR})
    target_include_directories(element_visualizer PUBLIC ${imgui_SOURCE_DIR}/backends)

    add_executable(drawtest opengl_tester.cpp ${IMGUI_SOURCES})
    target_link_libraries(drawtest PUBLIC iceicle_fe)
    target_link_libraries(drawtest PUBLIC iceicle_opengl_utils)
    target_include_directories(drawtest PUBLIC ${imgui_SOURCE_DIR})
    target_include_directories(drawtest PUBLIC ${imgui_SOURCE_DIR}/backends)
endif()


# ==================
# === UNIT TESTS ===
# ==================

include(GoogleTest)

# === Unit Tests for the Finite Element Library ===
set(FELIB_TEST_SOURCES
    test_felib_main.cpp
    test_geometry.cpp
    test_polytope.cpp
    test_face_utils.cpp
    TestTraceSpace.cpp
    TestFespan.cpp
    TestFESpace.cpp
    TestHypercubeFace.cpp
    TestQuadrature1d.cpp
    TestHypercubeTransform.cpp
    TestSimplexTransform.cpp
    TestSimplexQuadrature.cpp
    TestSimplexLagrange.cpp
    test_spacetime_utils.cpp
    test_utils.cpp
    test_mesh.cpp
    test_evaluation.cpp
    test_jacobian_utils.cpp
    test_ns.cpp
    )

add_executable(test_felib ${FELIB_TEST_SOURCES})
target_link_libraries( test_felib PUBLIC iceicle_fe )
target_link_libraries( test_felib PUBLIC iceicle_solvers )
target_link_libraries( test_felib PUBLIC nt_autodiff) 
target_link_libraries( test_felib PUBLIC gtest gtest_main) 

if(NOT ICEICLE_ASAN)
    gtest_discover_tests(test_felib)
endif()

if(ICEICLE_USE_METIS AND ICEICLE_USE_VTK)
    add_executable(test_mesh_partition test_mesh_partition.cpp)
    target_link_libraries( test_mesh_partition PUBLIC iceicle_fe )
    target_link_libraries( test_mesh_partition PUBLIC iceicle_io )
    #    target_link_libraries( test_mesh_partition PUBLIC iceicle_solvers )
    target_link_libraries( test_mesh_partition PUBLIC gtest )
    target_include_directories( test_mesh_partition PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries( test_mesh_partition PUBLIC ${MPI_CXX_LIBRARIES} )
    target_link_libraries( test_mesh_partition PUBLIC ${VTK_LIBRARIES} )
endif()

# === Unit Tests for the Solver Library ===
#  ( currently not in CI )

#if(ICEICLE_USE_HYPRE)
#set(SOLVER_TEST_SOURCES
#    test_solvers_main.cpp
#    test_petsc_solvers.cpp
#    test_hypre_interface.cpp)
#add_executable( test_solvers ${SOLVER_TEST_SOURCES})
#target_link_libraries( test_solvers PUBLIC iceicle_fe )
#target_link_libraries( test_solvers PUBLIC iceicle_solvers )
#target_link_libraries( test_solvers PUBLIC gtest )
#endif()

if(ICEICLE_USE_PETSC)
    set(SOLVER_TEST_SOURCES
        test_solvers_main.cpp
        test_petsc_jacobian.cpp 
    )
    add_executable(test_solvers ${SOLVER_TEST_SOURCES})
    target_link_libraries( test_solvers PUBLIC iceicle_fe )
    target_link_libraries( test_solvers PUBLIC iceicle_solvers )
    target_link_libraries( test_solvers PUBLIC gtest )
    target_include_directories( test_solvers PUBLIC ${MPI_CXX_INCLUDE_PATH})
    target_link_libraries( test_solvers PUBLIC ${MPI_CXX_LIBRARIES} )
endif()

add_executable(test_linalg test_linalg.cpp)
target_link_libraries( test_linalg PUBLIC iceicle_fe )
