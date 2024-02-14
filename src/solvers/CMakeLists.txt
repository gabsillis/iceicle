add_library(iceicle_solvers INTERFACE)
target_link_libraries(iceicle_solvers INTERFACE iceicle_fe)
if(ICEICLE_USE_HYPRE)
    message("Using HYPRE in solvers")
    target_link_libraries(iceicle_solvers INTERFACE HYPRE::HYPRE)
endif()
if(ICEICLE_USE_PETSC)
    message("Using PETSc in solvers")
    target_link_libraries(iceicle_solvers INTERFACE PkgConfig::PETSC)
    target_include_directories(iceicle_solvers INTERFACE ${PETSC_INCLUDE_DIRS})
endif()
target_include_directories(iceicle_solvers INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
