
#include "iceicle/geometry/hypercube_element.hpp"
#include <iceicle/geometry/hypercube_element_utility.hpp>
#include <iceicle/fe_function/opengl_fe_function_utils.hpp>
#include <iceicle/load_shaders.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

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

struct Example {
    virtual void draw_in_loop() = 0;

    virtual ~Example() = default;
};

struct Example0 : public Example {
    GLuint VertexArrayID;
    GLuint vertexbuffer;
    GLuint programID; // id for shader 
    GLuint fbo;

    // An array of 3 vectors which represents 3 vertices
    inline static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
    };

    Example0(){
        // Create and compile our GLSL program from the shaders
        programID = LoadShaders( "./shaders/SimpleVertexShader.vertexshader", "./shaders/SimpleFragmentShader.fragmentshader" );

        glGenVertexArrays(1, &VertexArrayID);
        glBindVertexArray(VertexArrayID);

        // Generate 1 buffer, put the resulting identifier in vertexbuffer
        glGenBuffers(1, &vertexbuffer);
        // The following commands will talk about our 'vertexbuffer' buffer
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        // Give our vertices to OpenGL.
        glBufferData(
            GL_ARRAY_BUFFER, 
            sizeof(g_vertex_buffer_data),
            g_vertex_buffer_data,
            GL_STATIC_DRAW
        );
    }

    void draw_in_loop() override {
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        // Use our shader
        glUseProgram(programID);
        // 1st attribute buffer : vertices
        glEnableVertexAttribArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer);
        glVertexAttribPointer(
        0,                  // attribute 0. No particular reason for 0, but must match the layout in the shader.
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        0,                  // stride
        (void*)0            // array buffer offset
        );
        // Draw the triangle !
        glDrawArrays(GL_TRIANGLES, 0, 3); // Starting from vertex 0; 3 vertices total -> 1 triangle
        glDisableVertexAttribArray(0);
    }
};

/**
 * Draw n random quads and the reference point inside them 
 */
template<int Pn>
struct DrawQuads : public Example {
    // the number of random quads to generate 
    int nquads;

    GLuint vertex_array_id;
    GLuint domain_shader_id;
    GLuint vertex_buffer;
    GLuint element_buffer;

    std::vector<HypercubeElement<double, int, 2, Pn>> quads;
    NodeArray<double, 2> coords;

    // GL buffer arrays
    std::vector< glm::vec3 > vertices;
    std::vector< unsigned int > vertex_indices;


    // An array of 3 vectors which represents 3 vertices
    inline static const GLfloat g_vertex_buffer_data[] = {
        -1.0f, -1.0f, 0.0f,
        1.0f, -1.0f, 0.0f,
        0.0f,  1.0f, 0.0f,
    };
    
    DrawQuads(int nquads) {

        // generate the gl arrays
        glGenVertexArrays(1, &vertex_array_id);
        glBindVertexArray(vertex_array_id);

        // randomness init 
        random_device rdev{};
        default_random_engine engine{rdev()};
        uniform_real_distribution<double> centroid_dist{-0.5, 0.5};
        uniform_real_distribution<double> size_dist{0.3, 0.6};

        // load shaders and compile 
        domain_shader_id = LoadShaders("./shaders/domn_shader.vert", "./shaders/domn_shader.frag" );

        // create the quads
        for(int i = 0; i < nquads; ++i){
            double centroid[2] = {centroid_dist(engine), centroid_dist(engine)};
            double sizes[2] = {size_dist(engine), size_dist(engine)};
            auto quad = create_hypercube<double, int, 2, Pn>(centroid, sizes, coords);
            quads.push_back(quad);
            std::cout << "centroid: [ " << centroid[0] 
                << " " << centroid[1] << " ]" << std::endl;
        }

        // TODO:
        // coords.random_perturb(-0.05, 0.05);

        // decompose each quad into triangles 
        for(auto &quad : quads){
            triangle_decompose_quad(quad, coords, 10 * Pn, vertex_indices, vertices);
        }

        for(int i = 0; i < nquads * Pn * Pn * 2; ++i){
            std::cout << "triangle " << i << ": [ ";
            for(int ipoin = 0; ipoin < 3; ++ipoin){
                std::cout << vertex_indices[i * 3 + ipoin] << " ";
            }
            std::cout << "]" << std::endl;
        }

        std::cout << std::endl;
        for(glm::vec3 vert : vertices){
            std::cout << "vert: [ " << vert.x 
                << " " << vert.y << " " 
                << vert.z << " ]" << std::endl;
        }

        // Dark blue background
        glClearColor(0.0f, 0.0f, 0.4f, 0.0f);

        // bind the vertex buffer 
        glGenBuffers(1, &vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            vertices.size() * sizeof(glm::vec3),
            &vertices[0],
            GL_STATIC_DRAW
        );


        // bind the element_buffer for indexing
        glGenBuffers(1, &element_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, element_buffer);
        glBufferData(
            GL_ARRAY_BUFFER,
            vertex_indices.size() * sizeof(unsigned int),
            &vertex_indices[0],
            GL_STATIC_DRAW
        );

    }

    void draw_in_loop() override {
        // Clear the screen
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // use the domain shader 
        glUseProgram(domain_shader_id);

        // 1rst attribute buffer : vertices
		glEnableVertexAttribArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
		glVertexAttribPointer(
			0,                  // attribute
			3,                  // size
			GL_FLOAT,           // type
			GL_FALSE,           // normalized?
			0,                  // stride
			(void*)0            // array buffer offset
		);

        // Index buffer
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, element_buffer);

        // Draw the triangles !
		glDrawElements(
			GL_TRIANGLES,      // mode
			vertex_indices.size(),    // count
			GL_UNSIGNED_INT,   // type
			(void*)0           // element array buffer offset
		);

//        glDrawArrays(GL_TRIANGLES, 0, 3);
        glDisableVertexAttribArray(0);

        // swap buffers 
//        glfwSwapBuffers(window);
 //       glfwPollEvents();
    }

    ~DrawQuads(){
        glDeleteBuffers(1, &vertex_buffer);
        glDeleteBuffers(1, &element_buffer);
    }
};

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
    bool show_another_window = false;
    Example0 ex0{};
    DrawQuads<2> draw_quads{1};

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

            ImGui::Begin("Hello, world!");                          // Create a window called "Hello, world!" and append into it.

            ImGui::Text("This is some useful text.");               // Display some text (you can use a format strings too)
            ImGui::Checkbox("Another Window", &show_another_window);

            ImGui::SliderFloat("float", &f, 0.0f, 1.0f);            // Edit 1 float using a slider from 0.0f to 1.0f
            ImGui::ColorEdit3("clear color", (float*)&clear_color); // Edit 3 floats representing a color

            if (ImGui::Button("Button"))                            // Buttons return true when clicked (most widgets return true when edited/activated)
                counter++;
            ImGui::SameLine();
            ImGui::Text("counter = %d", counter);

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
            ImGui::End();
        }
        // 3. Show another simple window.
        if (show_another_window)
        {
            ImGui::Begin("Another Window", &show_another_window);   // Pass a pointer to our bool variable (the window will have a closing button that will clear the bool when clicked)
            ImGui::Text("Hello from another window!");
            if (ImGui::Button("Close Me"))
                show_another_window = false;
            ImGui::End();
        }

        // Draw our triangle 
        {
            ImGui::Begin("Triangle!");
            ImGui::BeginChild("Render");
            ImVec2 wSize = ImGui::GetContentRegionAvail();
            ImVec2 wpos = ImGui::GetWindowPos();

            // resize viewport and framebuffer 
            // TODO: use a callback
            GLfloat aspect_ratio = 1.0;
            gl::set_viewport_maintain_aspect_ratio(
                    wSize.x, wSize.y, aspect_ratio);
            fbo1.rescale_frame_buffer(wSize.x, wSize.y);

            // bind the framebuffer and draw
            fbo1.bind();
            draw_quads.draw_in_loop();
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
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
