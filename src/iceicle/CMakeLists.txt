add_executable(iceicle_1d iceicle_1d.cpp)
target_link_libraries(iceicle_1d iceicle_lib)
target_link_libraries(iceicle_1d ${MPI_CXX_LIBRARIES})

if(ICEICLE_USE_OPENGL) 
    SET(IMGUI_SOURCES 
        ${imgui_SOURCE_DIR}/imgui.cpp
        ${imgui_SOURCE_DIR}/imgui_draw.cpp
        ${imgui_SOURCE_DIR}/imgui_tables.cpp
        ${imgui_SOURCE_DIR}/imgui_widgets.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
        ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
    )
    add_executable(mesh_visualizer mesh_visualizer.cpp ${IMGUI_SOURCES})
    target_link_libraries(mesh_visualizer PUBLIC iceicle_fe)
    target_link_libraries(mesh_visualizer PUBLIC iceicle_io)
    target_link_libraries(mesh_visualizer PUBLIC iceicle_opengl_utils)
    target_include_directories(mesh_visualizer PUBLIC ${imgui_SOURCE_DIR})
    target_include_directories(mesh_visualizer PUBLIC ${imgui_SOURCE_DIR}/backends)
endif()


# =========================
# = Heat Equation Miniapp =
# =========================
if(ICEICLE_USE_LUA)
    add_executable(heat_eqn_miniapp heat_eqn_miniapp.cpp)
    target_link_libraries(heat_eqn_miniapp iceicle_lib)
    target_link_libraries(heat_eqn_miniapp ${MPI_CXX_LIBRARIES})
endif()
