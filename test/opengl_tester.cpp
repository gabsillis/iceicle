
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iceicle/opengl_drawer.hpp>
#include <iceicle/opengl_memory.hpp>

using namespace glm;
using namespace std;
using namespace iceicle::gl;


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "GLFW Error %d: %s\n", error, description);
}

int main(int argc, char** argv){

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
    window = glfwCreateWindow( 1024, 768, "Test OpenGL", NULL, NULL);

    if(window == nullptr) return 1;
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    glViewport(0, 0, 1024, 768);

    
    float xmin = -1.0;
    float ymin = -2.0;
    float xmax = 1.0;
    float ymax = 2.0;
    Triangle tri1 = {
        .pt1 = {xmin, ymin, 0.0},
        .pt2 = {xmax, ymin, 0.0},
        .pt3 = {xmax, ymax, 0.0}
    };


    Triangle tri2 = {
        .pt1 = {xmin, ymin, 0.0},
        .pt2 = {xmax, ymax, 0.0},
        .pt3 = {xmin, ymax, 0.0}
    };

//    ShapeDrawer<Triangle> triangle_drawer;
//    triangle_drawer.shader.load();
//    triangle_drawer.shader.set3Float("xmin", xmin, ymin, 0.0);
//    triangle_drawer.shader.set3Float("xmax", xmax, ymax, 0.0);
//    triangle_drawer.add_shape(tri1);
//    triangle_drawer.add_shape(tri2);
//    triangle_drawer.update();

    BufferedShapeDrawer<Curve> curve_drawer;
    Curve c1 = {{{0.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {1.0, 2.0, 0.0}}};
    curve_drawer.shader.load();
    curve_drawer.shader.set3Float("xmin", xmin, ymin, 0.0);
    curve_drawer.shader.set3Float("xmax", xmax, ymax, 0.0);
    curve_drawer.add_shape(c1);
    curve_drawer.update();


    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        curve_drawer.draw();
//        triangle_drawer.draw();
        glfwPollEvents();    
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
