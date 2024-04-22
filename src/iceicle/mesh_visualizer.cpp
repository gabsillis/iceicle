#include <iceicle/mesh/mesh.hpp>
#include <iceicle/pvd_writer.hpp>
#include <iceicle/fe_function/opengl_fe_function_utils.hpp>
#include <iceicle/opengl_utils.hpp>
#include <iceicle/geometry/face_utils.hpp>
#include <iceicle/opengl_drawer.hpp>
#include <iceicle/load_shaders.hpp>
#include <iceicle/opengl_utils.hpp>
#include "iceicle/geometry/face.hpp"

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include "Numtool/point.hpp"
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <iostream>
#include <random>

using namespace glm;
using namespace std;
using namespace iceicle;
using namespace iceicle::transformations;
using namespace iceicle::gl;


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int argc, char**argv){
    glfwSetErrorCallback(glfw_error_callback);
    glewExperimental = true;
    if (!glfwInit())
        return 1;


    // ===========================
    // = OpenGL 3.3 window hints =
    // ===========================
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

    // ========================
    // = Create OpenGL window =
    // ========================
    GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
    window = glfwCreateWindow( 1024, 768, "Element Visualizer", NULL, NULL);

    if(window == nullptr) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }
    // =======================
    // = Setup Imgui Context =
    // =======================
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    
    // ===============
    // = Setup Style =
    // ===============
    ImGui::StyleColorsDark();

    // ===========================
    // = Setup Renderer backends =
    // ===========================
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init("#version 330 core");

    // =========
    // = State =
    // =========
    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);
    int nx = 1, ny = 1;
    float xmin = -1, ymin = -1;
    float xmax = 1, ymax = 1;
    bool number_elements = false;
    bool number_faces = false;
    bool draw_normals = false;

    AbstractMesh<double, int, 2> *mesh = nullptr;
    io::PVDWriter<double, int, 2> mesh_writer{mesh};

    // ======================
    // = Setup Face Drawing =
    // ======================
    ShapeDrawer<ArrowGenerated> normal_drawer{};
    BufferedShapeDrawer<Curve> face_drawer{};

    // ================
    // = FrameBuffers =
    // ================
    FrameBuffer fbo1(800, 600);

    while(!glfwWindowShouldClose(window)){
        // Poll and handle events (inputs, window resize, etc.)
        // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to use your inputs.
        // - When io.WantCaptureMouse is true, do not dispatch mouse input data to your main application, or clear/overwrite your copy of the mouse data.
        // - When io.WantCaptureKeyboard is true, do not dispatch keyboard input data to your main application, or clear/overwrite your copy of the keyboard data.
        // Generally you may always pass all inputs to dear imgui, and hide them from your application based on those two flags.
        glfwPollEvents();

        // Start the Dear ImGui frame
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        { // simple window 
            static float f = 0.0f;
            static int counter = 0;

            ImGui::Begin("2D Uniform Mesh Generation");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)

            // mesh display options
            ImGui::Checkbox("Number Elements", &number_elements);
            ImGui::Checkbox("Number Faces", &number_faces);
            ImGui::Checkbox("Draw Normals", &draw_normals);

            // BG color edit
            ImGui::ColorEdit3("bg color", (float*)&clear_color); // Edit 3 floats representing a color

            // Number of elements in the x and y directions
            ImGui::InputInt("nx", &nx);
            ImGui::InputInt("ny", &ny);

            // Define the corners
            ImGui::InputFloat("xmin", &xmin);
            ImGui::InputFloat("ymin", &ymin);

            ImGui::InputFloat("xmax", &xmax);
            ImGui::InputFloat("ymax", &ymax);

            // Generate the mesh when clicked
            if (ImGui::Button("Generate Mesh")){
                if(mesh != nullptr) delete mesh;

                double corner1[2] = {xmin, ymin};
                double corner2[2] = {xmax, ymax};
                int directional_nelem[2] = {nx, ny};

                mesh = new AbstractMesh<double, int, 2>(
                    {xmin, ymin}, {xmax, ymax}, {nx, ny}, 1
                );

                face_drawer.clear();
                normal_drawer.clear_list();

                // get the bounding box and apply to the vertex shaders
                BoundingBox mesh_bounds = get_mesh_bounds(*mesh);
                face_drawer.shader.load();
                face_drawer.shader.set3Float("xmin", mesh_bounds.xmin);
                face_drawer.shader.set3Float("xmax", mesh_bounds.xmax);

                normal_drawer.shader.load();
                normal_drawer.shader.set3Float("xmin", mesh_bounds.xmin);
                normal_drawer.shader.set3Float("xmax", mesh_bounds.xmax);


                for(int iface = mesh->interiorFaceStart; iface < mesh->interiorFaceEnd; ++iface){
                    // add the face to draw
                    face_drawer.add_shape(create_curve(mesh->faces[iface], mesh->nodes));

                    // get the normal to draw 
                    auto centroid = face_centroid(*(mesh->faces[iface]), mesh->nodes);

                    auto normal = calc_normal(
                        *(mesh->faces[iface]),
                        mesh->nodes,
                        ref_face_centroid(*(mesh->faces[iface]))
                    );

                    ArrowGenerated normal_arrow = {
                        to_vec3(centroid),
                        to_vec3(normal)
                    };

                    normal_drawer.add_shape(normal_arrow);
                }

                for(int iface = mesh->bdyFaceStart; iface < mesh->bdyFaceEnd; ++iface){
                    // add the face to draw
                    face_drawer.add_shape(create_curve(mesh->faces[iface], mesh->nodes));

                    // get the normal to draw 
                    auto centroid = face_centroid(*(mesh->faces[iface]), mesh->nodes);

                    auto normal = calc_normal(
                        *(mesh->faces[iface]),
                        mesh->nodes,
                        ref_face_centroid(*(mesh->faces[iface]))
                    );

                    ArrowGenerated normal_arrow = {
                        to_vec3(centroid),
                        to_vec3(normal)
                    };

                    normal_drawer.add_shape(normal_arrow);
                }

                // update the buffers
                face_drawer.update();
                normal_drawer.update();
                mesh_writer.register_mesh(mesh);

            } // Buttons return true when clicked (most widgets return true when edited/activated)
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            if (ImGui::Button("Save Mesh")){
                mesh_writer.write_mesh();
            }
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }

        // Draw our Mesh
        {
            ImGui::Begin("Mesh View");
            ImGui::BeginChild("Render");
            ImVec2 wSize = ImGui::GetContentRegionAvail();
            ImVec2 wpos = ImGui::GetWindowPos();

            // resize viewport and framebuffer 
            // TODO: use a callback
            GLfloat aspect_ratio = 1.0;
            set_viewport_maintain_aspect_ratio(
                    wSize.x, wSize.y, aspect_ratio);
            fbo1.rescale_frame_buffer(wSize.x, wSize.y);

            // bind the framebuffer and draw
            fbo1.bind();
            // TODO: draw stuffs
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
            //normal_drawer.draw();
            if(mesh) face_drawer.draw();

            if(mesh && draw_normals) normal_drawer.draw();

            fbo1.unbind();

            ImGui::Image((ImTextureID) fbo1.texture, wSize, 
                    ImVec2(0, 1), ImVec2(1, 0));
            ImGui::EndChild();
            ImGui::End();
        }

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ===========
    // = Cleanup =
    // ===========
    if(mesh != nullptr) delete mesh;
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
