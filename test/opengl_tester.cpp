
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iceicle/opengl_drawer.hpp>
#include <iceicle/opengl_memory.hpp>

using namespace glm;
using namespace std;
using namespace ICEICLE_GL;


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

    ShapeDrawer<ArrowGenerated> arrow_drawer;
    glm::vec3 anchor = {0.0, 0.0, 0.0};
    glm::vec3 arrow = {-0.5, -0.5, 0.0};

    glm::vec3 anchor2 = {0.5, 0.5, 0.0};
    ArrowGenerated arrow1{anchor, arrow};
    ArrowGenerated arrow2{anchor2, arrow};
    arrow_drawer.add_shape(arrow1);
    arrow_drawer.add_shape(arrow2);

    Triangle tri1 = {
        .pt1 = {0.0, 0.0, 0.0},
        .pt2 = {1.0, 0.0, 0.0},
        .pt3 = {0.5, 1.0, 0.0},
    };


    Triangle tri2 = {
        .pt1 = {-1.0, 0.0, 0.0},
        .pt2 = {0.0, 0.0, 0.0},
        .pt3 = {-0.5, 1.0, 0.0},
    };

    unsigned int VBO;
    ICEICLE_GL::ArrayObject vao;
    glGenBuffers(1, &VBO);

	const char *tri_vert_shader = 
#include "../../shaders/triangle2d.vert"
	;
	const char *tri_frag_shader = 
#include "../../shaders/triangle2d.frag"
	;
    Shader shader{tri_vert_shader, tri_frag_shader, nullptr};

    vao.bind();
    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, 3 * sizeof(glm::vec3), &(tri1.pt1), GL_STATIC_DRAW);
    glVertexAttribPointer(
        0,                  // attribute
        3,                  // size
        GL_FLOAT,           // type
        GL_FALSE,           // normalized?
        3 * sizeof(GLfloat),                  // stride
        (void*)0            // array buffer offset
    );

    ShapeDrawer<Triangle> triangle_drawer;
    triangle_drawer.add_shape(tri2);
    triangle_drawer.update();
    arrow_drawer.update();

    while(!glfwWindowShouldClose(window))
    {
        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        shader.load();
        vao.bind();
        glEnableVertexAttribArray(0);
        //glDrawArrays(GL_TRIANGLES, 0, 3);

        triangle_drawer.draw();
        arrow_drawer.draw();
        glfwPollEvents();    
        glfwSwapBuffers(window);
    }

    glfwTerminate();
    return 0;
}
