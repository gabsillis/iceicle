#pragma once
#include <iceicle/load_shaders.hpp>
#include <iceicle/opengl_memory.hpp>
#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

namespace ICEICLE_GL {
    

    /**
     * @brief Abstract class for shapes that can be drawn
     */
    struct Shape {


    };


    /**
     * @brief an Arrow
     * the base is at a point and it forms a vector in the given direction
     */
    struct Arrow final : public Shape {
        glm::vec3 anchor; /// the point at the base of the shaft
        glm::vec3 vec; /// the magnitude and direction of the arrow
    };
    static_assert(sizeof(Arrow) == 2 * sizeof(glm::vec3), "Arrow data not shaped correctly, so copy will fail");

    struct ArrowGenerated final : public Shape { // must be a simple struct to copy
        glm::vec3 pts[9];

        // The ratio of the shaft width to the length vec
        static constexpr float SHAFT_WIDTH_MUL = 0.06;

        // The ratio of the widest point of the head to vec
        static constexpr float HEAD_WIDTH_MUL = 0.13;

        // The ratio of the length of the shaft to vec
        static constexpr float SHAFT_LENGTH_MUL = 0.85;

        ArrowGenerated(glm::vec3 &anchor, glm::vec3 &arrow){
            glm::vec3 normal = {-arrow[1], arrow[0], 0.0};

            glm::vec3 shaft_horiz = normal;
            shaft_horiz *= 0.5 * SHAFT_WIDTH_MUL;

            glm::vec3 shaft_vert = arrow;
            shaft_vert *= SHAFT_LENGTH_MUL;

            // first triangle
            pts[0] = anchor - shaft_horiz;
            pts[1] = anchor - shaft_horiz + shaft_vert;
            pts[2] = anchor + shaft_horiz;

            // second triangle 
            pts[3] = anchor + shaft_horiz;
            pts[4] = anchor + shaft_horiz + shaft_vert;
            pts[5] = anchor - shaft_horiz + shaft_vert;

            glm::vec3 head_horiz = normal;
            head_horiz *= 0.5 * HEAD_WIDTH_MUL;
            // head triangle 
            pts[6] = anchor + head_horiz + shaft_vert;
            pts[7] = anchor - head_horiz + shaft_vert;
            pts[8] = anchor + arrow;
        }
    };
    static_assert(sizeof(ArrowGenerated) == 9 * sizeof(glm::vec3));


    struct Triangle : public Shape {
        glm::vec3 pt1;
        glm::vec3 pt2;
        glm::vec3 pt3;
    };


    /**
     * @brief Draws shapes onto the current frame
     */
    template<class ShapeType>
    class ShapeDrawer {
        ArrayObject vao;

        std::vector<ShapeType> draw_list; /// list of things to draw
        std::vector<GLuint> vertex_attributes; /// list of vertex attributes to enable

        private:

        /** call the draw arrays command */ 
        void draw_arrays();

        /** buffer the data from the draw_list */
        void buffer_data();

        public:

        /// The shader that is used to draw the shapes
        Shader shader;

        /** 
         * @brief initialize the VAO and shader
         * done in a kind of CRTP way
         * by specializing for each ShapeType
         * Must create a shader 
         * Must create the vao 
         * list all the vertex attribute indices used
         * Generate any buffers needed
         */
        ShapeDrawer();
        
        /** 
         * @brief add a shape to the list to get drawn
         * works for shape types that can be directly copied,
         */
        void add_shape(ShapeType shp){
            draw_list.push_back(shp);
        }

        /** @brief clear all items in the draw list */
        void clear_list(){ draw_list.clear(); }

        /** update Device to prepare for a call to draw() */ 
        void update(){
            vao.bind();
            buffer_data();
        }

        /**
         * @brief draw all the shapes
         * WARNING: prerequisite: update()
         */
        void draw(){
            // load the shader program
            shader.load();

            // bind the vertex array 
            vao.bind();

            // enable the vertex attributes
            for(GLuint attr_id: vertex_attributes) glEnableVertexAttribArray(attr_id);

            // draw 
            draw_arrays();

            // disable the vetex attributes
            for(GLuint attr_id: vertex_attributes) glDisableVertexAttribArray(attr_id);
        }
    };

    // ===================
    // = Buffered Shapes =
    // ===================

    struct Curve final {
        std::vector<glm::vec3> pts;
    };

    /** @brief a custom buffer for each shape */
    template<class ShapeType>
    struct ShapeDataBuffer{
        void add_shape(const ShapeType shp){}

        void clear(){}
    };

    template<>
    struct ShapeDataBuffer<Curve> {
        std::vector<glm::vec3> pts;
        std::vector<std::size_t> iterators;

        ShapeDataBuffer() : pts{}, iterators{0} {}

        // TODO: use triangles to make thickness
        void add_shape(const Curve &shp){
            // add all the points
            pts.insert(pts.end(), shp.pts.begin(), shp.pts.end());

            // update the iterator 
            iterators.push_back(pts.size());
        }

        void clear(){
            pts.clear();
            iterators.clear();
            iterators.push_back(0);
        }
    };

    /**
     * @brief draw shapes onto the current frame 
     * for shapes with complex or dynamic data that must be buffered 
     */
    template<class ShapeType>
    class BufferedShapeDrawer{

        ArrayObject vao;
        ShapeDataBuffer<ShapeType> host_buffer;
        std::vector<GLuint> vertex_attributes; /// list of vertex attributes to enable

        /** buffer data to the device */ 
        void buffer_data();

        /** the draw command that draws out the buffered data */ 
        void draw_arrays(); 

        public:
        Shader shader; /// the shader used when drawing
      
        /** @brief Constructor
         *
         * Must create a shader 
         * Must create the vao 
         * list all the vertex attribute indices used
         * Generate any buffers needed
         */
        BufferedShapeDrawer();

        /** add a shape to draw */ 
        void add_shape(const ShapeType &shp) { host_buffer.add_shape(shp); }

        /** clear the data on the host (empty it out) */
        void clear(){ host_buffer.clear(); }


        /** update Device to prepare for a call to draw() */ 
        void update(){
            vao.bind();
            buffer_data();
        }

        /**
         * @brief draw all the shapes
         * WARNING: prerequisite: update()
         */
        void draw(){
            // load the shader program
            shader.load();

            // bind the vertex array 
            vao.bind();

            // enable the vertex attributes
            for(GLuint attr_id: vertex_attributes) glEnableVertexAttribArray(attr_id);

            // draw 
            draw_arrays();

            // disable the vetex attributes
            for(GLuint attr_id: vertex_attributes) glDisableVertexAttribArray(attr_id);
        }
        
    };

}
