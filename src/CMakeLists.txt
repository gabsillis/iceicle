add_library(iceicle_lib INTERFACE)

add_subdirectory(util)
add_subdirectory(math)
add_subdirectory(FiniteElement)
add_subdirectory(io)

target_link_libraries(iceicle_lib INTERFACE iceicle_fe)
target_link_libraries(iceicle_lib INTERFACE iceicle_io)

# executables
add_subdirectory(iceicle)
