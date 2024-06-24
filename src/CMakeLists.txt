# Subdirectory src 
add_library(iceicle_lib INTERFACE)

# add subdirectories of code in order of dependency
add_subdirectory(util)
add_subdirectory(math)
add_subdirectory(FiniteElement)
add_subdirectory(io)
add_subdirectory(solvers)
add_subdirectory(vtk)

target_link_libraries(iceicle_lib INTERFACE iceicle_fe)
target_link_libraries(iceicle_lib INTERFACE iceicle_solvers)
target_link_libraries(iceicle_lib INTERFACE iceicle_io)

# executables
add_subdirectory(iceicle)
