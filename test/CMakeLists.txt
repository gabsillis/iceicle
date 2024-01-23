if(ENABLE_MATPLOTLIB_CPP)
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
#    add_executable(element_visualizer element_visualizer.cpp ${IMGUI_SOURCES})
#    target_link_libraries(element_visualizer PUBLIC iceicle_fe)
#    target_link_libraries(element_visualizer PUBLIC iceicle_opengl_utils)
#    target_include_directories(element_visualizer PUBLIC ${imgui_SOURCE_DIR})
#    target_include_directories(element_visualizer PUBLIC ${imgui_SOURCE_DIR}/backends)
#
    add_executable(drawtest opengl_tester.cpp ${IMGUI_SOURCES})
    target_link_libraries(drawtest PUBLIC iceicle_fe)
    target_link_libraries(drawtest PUBLIC iceicle_opengl_utils)
    target_include_directories(drawtest PUBLIC ${imgui_SOURCE_DIR})
    target_include_directories(drawtest PUBLIC ${imgui_SOURCE_DIR}/backends)
endif()


# ==================
# === UNIT TESTS ===
# ==================

enable_testing()

include(GoogleTest)
add_executable(test_felib 
    TestFespan.cpp
    TestFESpace.cpp
    TestHypercubeFace.cpp
    TestQuadrature1d.cpp
    TestHypercubeTransform.cpp
    TestSimplexTransform.cpp
    TestSimplexQuadrature.cpp
    TestSimplexLagrange.cpp
  #  TestHypercubeFE.cpp
  #  TestSimplexFE.cpp
)
target_link_libraries( test_felib PUBLIC iceicle_fe )
target_link_libraries( test_felib PUBLIC nt_autodiff) 
target_link_libraries( test_felib PUBLIC gtest gtest_main) 

gtest_discover_tests(test_felib)

