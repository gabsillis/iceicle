cmake_minimum_required(VERSION 3.20.0) # support for c++23
project(ICEICLE_PROJECT VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 23)

# FLOATING POINT AND DOUBLE TYPES
# in order of priority
option(USE_DOUBLE_PRECISION "Use double precision floating point number" ON)
option(USE_SINGLE_PRECISION "Use single precision floating point number" OFF)
option(USE_QUAD_PRECISION "Use quad precision floating point number" OFF)
option(USE_32_BIT_INT "Use 32 bit integers for large arrays" ON)
option(USE_64_BIT_INT "Use 64 bit integers for large arrays" OFF)

if(USE_DOUBLE_PRECISION)
  MESSAGE("Using Double Precision.")
  add_definitions(-DT_DOUBLE_PRECISION)
endif()
if(USE_SINGLE_PRECISION)
  MESSAGE("Using Single Precision.")
  add_definitions(-DT_SINGLE_PRECISION)
endif()
if(USE_QUAD_PRECISION)
  MESSAGE("Using Quad Precision.")
  add_definitions(-DT_QUAD_PRECISION)
endif()
if(USE_32_BIT_INT)
  MESSAGE("Using 32 bit integers.")
  add_definitions(-DIDX_32_BIT)
endif()
if(USE_64_BIT_INT)
  MESSAGE("Using 64 bit integers.")
  add_definitions(-DIDX_64_BIT)
endif()

# ====================
# = Compiler Options =
# ====================
option(ICEICLE_MARCH "Build with the march=native flag if in release mode" ON)
option(ICEICLE_MTUNE "Build with the mtune=native flag if in release mode" ON)
if(ICEICLE_MARCH)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
endif()
if(ICEICLE_MTUNE)
  SET(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mtune=native")
endif()


# ===============
# = TPL OPTIONS =
# ===============
option(ICEICLE_USE_PETSC "Builds ICEicle with Petsc integration" OFF)
option(ICEICLE_USE_LUA "Builds ICEicle with Lua integration integration provided through sol3" OFF)
#option(ICEICLE_USE_IREP "Builds ICEicle with the irep library for Lua mixed-language" OFF)
option(ICEICLE_USE_HYPRE "Builds ICEicle with Hypre integration" OFF)
option(ICEICLE_USE_MFEM  "Enables MFEM as a third party library" OFF)
option(ICEICLE_USE_OPENGL "Enables OpenGL visualization with imgui" OFF)
option(ICEICLE_USE_MATPLOTPLUSPLUS "Enables matplotlib-cpp" OFF)

# ==================
# = CMake includes =
# ==================
include(FetchContent)
find_package(MPI)
find_package(PkgConfig REQUIRED)

# custom modules
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Generate Compile Commands (good for LSPs)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)	

# ================================
# = Cmake Output Directory Setup =
# ================================
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_custom_target(OUTPUTDIR ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/output)

# =========================
# = Third Party Libraries =
# =========================

# === REQUIRED Third Party Library: mdspan
add_library(mdspan INTERFACE)
target_include_directories(mdspan INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/mdspan/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)


# MPI Dependent 3rd party libraries 
if(MPI_FOUND)
  # === Third Party Library: Hypre ===
  if(ICEICLE_USE_HYPRE)
    add_subdirectory(./hypre/src)
  endif()

# === Third Party Library: Petsc ===
# NOTE: we use mpi functions in most cases with petsc 
# so even though petsc supports non-MPI builds, we do not
  if(ICEICLE_USE_PETSC)
    set(ENV{PKG_CONFIG_PATH} "$ENV{PETSC_DIR}/$ENV{PETSC_ARCH}/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")
    pkg_search_module(PETSC PETSc>3.5.0 IMPORTED_TARGET REQUIRED)
    message(STATUS "Using PETSc version: ${PETSC_VERSION}")
  endif()
else()
  if(ICEICLE_USE_HYPRE)
    message(SEND_ERROR "ICEICLE_USE_HYPRE requires MPI_FOUND")
  endif()
  if(ICEICLE_USE_PETSC)
    message(SEND_ERROR "ICEICLE_USE_PETSC requires MPI_FOUND")
  endif()
endif()

# === Third Party Library: Lua ===
if(ICEICLE_USE_LUA)
  # first get the base lua library
  find_package(Lua REQUIRED)
  message("Using Lua version: ${LUA_VERSION_STRING}")

  # sol3 
  FetchContent_Declare(
    sol2
    GIT_REPOSITORY https://github.com/ThePhD/sol2.git 
    GIT_TAG origin/develop # main didnt work with Lua 5.4.4 fedora but develop did
    GIT_PROGRESS TRUE
  )
  FetchContent_GetProperties(sol2)
  FetchContent_MakeAvailable(sol2)

endif()

# ==== Third Party Library: MFEM ===
set(ICEICLE_TPL_DIR "${CMAKE_SOURCE_DIR}/../iceicle_tpl")
set(MFEM_BUILD_DIR "${ICEICLE_TPL_DIR}/mfem/build")
if(ICEICLE_USE_MFEM)
  add_definitions(-DICEICLE_HAS_MFEM)
  message("Searching for mfem in: ${MFEM_BUILD_DIR}")
  if(MFEM_BUILD_DIR)
      # searches for MFEMConfig.cmake in the given path
      find_package(mfem REQUIRED NAMES MFEM HINTS ${MFEM_BUILD_DIR} NO_DEFAULT_PATH)
  else()
      message("Searching for mfem using normal find_package")
      find_package(mfem REQUIRED NAMES MFEM)
  endif()
endif()

# === Third Party Library: OpenGL ===
if(ICEICLE_USE_OPENGL)
  find_package(OpenGL REQUIRED)
  find_package(GLEW REQUIRED)
  find_package(glfw3 REQUIRED)

  # imgui
  set(imgui_SOURCE_DIR "${CMAKE_SOURCE_DIR}/imgui")

  # copy shaders
  file(COPY ${CMAKE_SOURCE_DIR}/shaders DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
endif()

  # === Third Party Library matplotlib-cpp ===
if(ICEICLE_USE_MATPLOTPLUSPLUS)
  FetchContent_Declare(
    matplotplusplus
    GIT_REPOSITORY https://github.com/alandefreitas/matplotplusplus
    GIT_TAG origin/master
  )

  FetchContent_GetProperties(matplotplusplus)
  if(NOT matplotplusplus_POPULATED)
      FetchContent_Populate(matplotplusplus)
      add_subdirectory(${matplotplusplus_SOURCE_DIR} ${matplotplusplus_BINARY_DIR} EXCLUDE_FROM_ALL)
  endif()
endif()

# Include subdirectories
add_subdirectory(numericaltoolbox)
add_subdirectory(src)

# ===========
# = TESTING =
# ===========

option(ENABLE_TESTING_ICEICLE "Enables testing and downlod gtest for ICEicle" ON)
IF(ENABLE_TESTING_ICEICLE)
  enable_testing()
  include(CTest)
# replacing find_package with a download
#  find_package(GTest REQUIRED)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
  )

  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
  add_subdirectory(test)

  # =================
  # = Documentation =
  # =================
  add_subdirectory(doc)
ENDIF()
