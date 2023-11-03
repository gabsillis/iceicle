add_executable(simplex_trans_ex SimplexTransformExample.cpp)
target_link_libraries(simplex_trans_ex PUBLIC matplot)
target_link_libraries(simplex_trans_ex PUBLIC iceicle_fe) 

add_executable(test_felib 
  TestHypercubeTransform.cpp
  TestSimplexTransform.cpp
  TestSimplexQuadrature.cpp
  TestSimplexLagrange.cpp
  TestSimplexFE.cpp
)
target_link_libraries( test_felib PUBLIC iceicle_fe )
target_link_libraries( test_felib PUBLIC nt_autodiff) 
target_link_libraries( test_felib PUBLIC gtest gtest_main) 
