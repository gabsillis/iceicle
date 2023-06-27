add_library(iceicle_lib INTERFACE)

add_subdirectory(math)
add_subdirectory(FiniteElement)

target_link_libraries(iceicle_lib INTERFACE iceicle_fe)

#Petsc
find_package(PkgConfig REQUIRED)
set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
pkg_search_module(PETSC PETSc>3.5.0 IMPORTED_TARGET REQUIRED)
message(STATUS "Using PETSc version: ${PETSC_VERSION}")
target_link_libraries(iceicle_lib INTERFACE PkgConfig::PETSC)
target_include_directories(iceicle_lib INTERFACE ${PETSC_INCLUDE_DIRS})

# executables
#add_subdirectory(iceicle)
