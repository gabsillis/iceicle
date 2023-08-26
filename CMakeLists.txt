cmake_minimum_required(VERSION 3.12.0)
project(ICEICLE_PROJECT VERSION 0.1.0)
set(CMAKE_CXX_STANDARD 20)

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

# Generate Compile Commands
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)	

find_package(MPI)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/obj)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin/obj)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
add_custom_target(OUTPUTDIR ALL COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/output)

# Include subdirectories
add_subdirectory(numericaltoolbox)
add_subdirectory(src)

#TESTING
option(ENABLE_TESTING_ICEICLE "Enables testing and downlod gtest for ICEicle" ON)
IF(ENABLE_TESTING_ICEICLE)
# replacing find_package with a download
#  find_package(GTest REQUIRED)
  include(FetchContent)
  FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        7274ec186442c96e4c7dcc99c684a03e5db2ff48 # main branch as of Aug 15
  )

  FetchContent_MakeAvailable(googletest)
  add_subdirectory(test)

ENDIF()
