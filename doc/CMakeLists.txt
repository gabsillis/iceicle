# Source: Sy Brand
# https://devblogs.microsoft.com/cppblog/clear-functional-c-documentation-with-sphinx-breathe-doxygen-cmake/

find_package(Doxygen)

if(Doxygen_FOUND)
  get_target_property(ICEICLE_PUBLIC_HEADER_DIR iceicle_lib INTERFACE_INCLUDE_DIRECTORIES)
  file(GLOB_RECURSE ICEICLE_PUBLIC_HEADERS ${ICEICLE_PUBLIC_HEADER_DIR}/*.hpp)

  set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
  set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doc/doxygen)
#This will be the main output of our command
  set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/html/index.html)
  set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
  set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

#Replace variables inside @@ with the current values
  configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

  file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR}) #Doxygen won't create this for us
  add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                    DEPENDS ${CAT_CUTIFIER_PUBLIC_HEADERS}
                    COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                    MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                    COMMENT "Generating docs")

  add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

  find_package(Sphinx)
    if(Sphinx_FOUND)
    set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
    set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/doc/sphinx)
    set(SPHINX_INDEX_FILE ${SPHINX_BUILD}/index.html)

# Only regenerate Sphinx when:
# - Doxygen has rerun
# - Our doc files have been updated
# - The Sphinx config has been updated
    add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                      COMMAND 
                        ${SPHINX_EXECUTABLE} -b html
                        # Tell Breathe where to find the Doxygen output
                        -Dbreathe_projects.ICEicle=${DOXYGEN_OUTPUT_DIR}/xml
                      ${SPHINX_SOURCE} ${SPHINX_BUILD}
                      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                      DEPENDS
                      # Other docs files you want to track should go here (or in some variable)
                      ${CMAKE_CURRENT_SOURCE_DIR}/index.rst
                      ${DOXYGEN_INDEX_FILE}
                      MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                      COMMENT "Generating documentation with Sphinx")

    add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})


# Add an install target to install the docs
    include(GNUInstallDirs)
    install(DIRECTORY ${SPHINX_BUILD}
    DESTINATION ${CMAKE_INSTALL_DOCDIR})
  endif()
endif()
