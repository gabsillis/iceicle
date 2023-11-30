
#include "iceicle/geometry/hypercube_element.hpp"
#include <iceicle/transformations/HypercubeTransformUtility.hpp>
#include <iceicle/fe_function/opengl_fe_function_utils.hpp>
#include <iceicle/load_shaders.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <iostream>
#include <random>

using namespace glm;
using namespace std;
using namespace ELEMENT;
using namespace ELEMENT::TRANSFORMATIONS;

struct Example {
    virtual void draw_in_loop(GLFWwindow *window) = 0;

    virtual ~Example() = default;
};

struct Example0 : public Example {
    GLuint VertexArrayID;
    GLuint vertexbuffer;
    GLuint programID; // id for shader 

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

    void draw_in_loop(GLFWwindow *window) override {
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
    FE::NodalFEFunction<double, 2> coords;

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

        coords.random_perturb(-0.05, 0.05);

        // decompose each quad into triangles 
        for(auto &quad : quads){
            triangle_decompose_quad(quad, coords, Pn * 10, vertex_indices, vertices);
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

    void draw_in_loop(GLFWwindow *window) override {
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

int main(int argc, char **argv){
    // Initialise GLFW
    glewExperimental = true; // Needed for core profile
    if( !glfwInit() )
    {
        fprintf( stderr, "Failed to initialize GLFW\n" );
        return -1;
    }

    glfwWindowHint(GLFW_SAMPLES, 4); // 4x antialiasing
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3); // We want OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // To make MacOS happy; should not be needed
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE); // We don't want the old OpenGL 

    // Open a window and create its OpenGL context
    GLFWwindow* window; // (In the accompanying source code, this variable is global for simplicity)
    window = glfwCreateWindow( 1024, 768, "Element Visualizer", NULL, NULL);
    if( window == NULL ){
        fprintf( stderr, "Failed to open GLFW window. If you have an Intel GPU, they are not 3.3 compatible. Try the 2.1 version of the tutorials.\n" );
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window); // Initialize GLEW
    glewExperimental=true; // Needed in core profile
    if (glewInit() != GLEW_OK) {
        fprintf(stderr, "Failed to initialize GLEW\n");
        return -1;
    }

    // Ensure we can capture the escape key being pressed below
    glfwSetInputMode(window, GLFW_STICKY_KEYS, GL_TRUE);

    // Get the example we want to run 
    int examplenr = 0;
    if(argc > 1){
        examplenr = atoi(argv[1]);
    }
    Example *ex;
    switch(examplenr){
        case 0:
            ex = new Example0{};
            break;
        case 1:
            int nquads = 1;
            if(argc > 2) nquads = atoi(argv[2]);
            ex = new DrawQuads<3>(nquads);
            break;
    }

    do{
        // Clear the screen. It's not mentioned before Tutorial 02, but it can cause flickering, so it's there nonetheless.
        glClear( GL_COLOR_BUFFER_BIT );

        // execute the draw loop
        ex->draw_in_loop(window);

        // Swap buffers
        glfwSwapBuffers(window);
        glfwPollEvents();

    } // Check if the ESC key was pressed or the window was closed
    while( glfwGetKey(window, GLFW_KEY_ESCAPE ) != GLFW_PRESS &&
        glfwWindowShouldClose(window) == 0 );

    // cleanup
    delete ex;

}
