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

# ============================
# = Conservation Law Miniapp =
# ============================
if(ICEICLE_USE_LUA)
    add_executable(conservation_law conservation_law.cpp)
    target_link_libraries(conservation_law iceicle_lib)
    target_link_libraries(conservation_law ${MPI_CXX_LIBRARIES})

    # TODO: maybe add opengl for mesh visualization on the fly

endif()

add_executable(sine_diffusion_1d sine_diffusion_1d.cpp)
target_link_libraries(sine_diffusion_1d iceicle_lib)


add_executable(mdgbl_1d mdgbl_1d.cpp)
target_link_libraries(mdgbl_1d iceicle_lib)
