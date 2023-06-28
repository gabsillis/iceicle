add_library(iceicle_lib INTERFACE)

add_subdirectory(util)
add_subdirectory(math)
add_subdirectory(FiniteElement)

target_link_libraries(iceicle_lib INTERFACE iceicle_fe)

# executables
add_subdirectory(iceicle)
