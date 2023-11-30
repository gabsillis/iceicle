if(ENABLE_MATPLOTLIB_CPP)
    add_executable(simplex_trans_ex SimplexTransformExample.cpp)
    target_link_libraries(simplex_trans_ex PUBLIC matplot)
    target_link_libraries(simplex_trans_ex PUBLIC iceicle_fe) 
endif()

if(ICEICLE_USE_OPENGL)
    add_executable(element_visualizer element_visualizer.cpp)
    target_link_libraries(element_visualizer PUBLIC iceicle_fe)
    target_link_libraries(element_visualizer PUBLIC iceicle_opengl_utils)

endif()

add_executable(test_felib 
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
