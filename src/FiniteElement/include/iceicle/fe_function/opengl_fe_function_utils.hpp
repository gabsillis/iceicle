#pragma once 
#include <iceicle/fe_function/nodal_fe_function.hpp>
#include <iceicle/load_shaders.hpp>
#include <iceicle/geometry/face.hpp>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/opengl_drawer.hpp>
#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

#ifndef NDEBUG
#include <iostream>
#endif // !NDEBUG


namespace ICEICLE_GL{

    struct FrameBuffer {
        GLuint fbo; /// frame buffer id
        GLuint texture; /// texture id
        GLuint rbo; /// render buffer id

        FrameBuffer(GLfloat width, GLfloat height){
            // create framebuffer
            glGenFramebuffers(1, &fbo);
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);

            // create texture 
            glGenTextures(1, &texture);
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, 
                    GL_RGB, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                    GL_TEXTURE_2D, texture, 0);

            // create render buffer
            glGenRenderbuffers(1, &rbo);
            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                    GL_RENDERBUFFER, rbo);

            // check creation of framebuffer 
            if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
                throw std::logic_error("Framebuffer not complete.");

            // unbind everything until we want it bound
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
            glBindTexture(GL_TEXTURE_2D, 0);
            glBindRenderbuffer(GL_RENDERBUFFER, 0); 
        }

        ~FrameBuffer(){
            glDeleteFramebuffers(1, &fbo);
            glDeleteTextures(1, &texture);
            glDeleteRenderbuffers(1, &rbo);
        }

        /** 
         * @brief rescale the frame buffer 
         * @param width the new width of the frame buffer 
         * @param height the new height of the frame buffer 
         */
        void rescale_frame_buffer(GLfloat width, GLfloat height){
            glBindTexture(GL_TEXTURE_2D, texture);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0);

            glBindRenderbuffer(GL_RENDERBUFFER, rbo);
            glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, width, height);
            glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);
        }

        /**
         * @brief bind the frame buffer 
         */
        void bind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, fbo);
        }

        /**
         * @brief unbind the frame buffer 
         */
        void unbind() const {
            glBindFramebuffer(GL_FRAMEBUFFER, 0);
        }
    };

    /**
     * @brief call glViewport to rescale viewport 
     * but maintain an aspect ratio 
     * and center the image in the viewport
     * @param width, the new width of the window 
     * @param height the new height of the window 
     * @param ar the aspect ratio 
     */
    inline void set_viewport_maintain_aspect_ratio(int width, int height, GLfloat ar){

        int width2 = std::min(width, (int) ((GLfloat)height * ar));
        int height2 = std::min(height, (int) ((GLfloat) width / ar));
        int width_diff = width2 - width;
        int height_diff = height2 - height;
        glViewport(-width_diff / 2, -height_diff / 2, width2, height2);
    }


    template<typename T, typename IDX>
    class ElementDrawer2D {
        GLuint vertex_array_id;
        GLuint element_shader_id;
        GLuint vertex_buffer_id;
        GLuint element_buffer_id;

        std::vector< glm::vec3 > vertices;
        std::vector< unsigned int > vertex_indices;

        public:
        ElementDrawer2D(){
            glGenBuffers(1, &vertex_buffer_id);
            glGenBuffers(1, &element_buffer_id);

            glGenVertexArrays(1, &vertex_array_id);

            element_shader_id = LoadShaders("./shaders/domn_shader.vert", "./shaders/domn_shader.frag" );
        }

        // TODO: add generic element drawing from reading mesh

        ~ElementDrawer2D(){
            glDeleteBuffers(1, &vertex_buffer_id);
            glDeleteBuffers(1, &element_buffer_id);
            glDeleteVertexArrays(1, &vertex_array_id);
        }
    };

    /** A bounding box in 3d space */
    struct BoundingBox {
        glm::vec3 xmin;
        glm::vec3 xmax;
    };

    template<typename T, typename IDX, int ndim>
    BoundingBox get_mesh_bounds(MESH::AbstractMesh<T, IDX, ndim> &mesh){
        BoundingBox box = {{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};
   
        for(int idim = 0; idim < ndim; ++idim){
            box.xmin[idim] = mesh.nodes[0][idim];
            box.xmax[idim] = mesh.nodes[0][idim];
        }
        for(int inode = 1; inode < mesh.nodes.n_nodes(); ++inode){
            for(int idim = 0; idim  < ndim; ++idim){
                box.xmin[idim] = std::min(box.xmin[idim], (GLfloat) mesh.nodes[inode][idim]);
                box.xmax[idim] = std::max(box.xmax[idim], (GLfloat) mesh.nodes[inode][idim]);
            }
        }

        return box;
    }

    template<typename T, typename IDX>
    Curve create_curve(ELEMENT::Face<T, IDX, 2> *facptr, FE::NodalFEFunction<T, 2> &coord, int nsegment=4){
        Curve out;

        double dx = 2.0 / nsegment;
        for(int ipt = 0; ipt < nsegment + 1; ++ipt){
            MATH::GEOMETRY::Point<T, 1> s = {dx * ipt - 1.0};
            MATH::GEOMETRY::Point<T, 2> x;
            facptr->transform(s, coord, x.data());
            out.pts.emplace_back(static_cast<GLfloat>(x[0]), static_cast<GLfloat>(x[1]), 0.0);
        }

        return out;
    }

    template<typename T, typename IDX>
    class FaceDrawer2D {
        GLuint vertex_array_id;
        GLuint face_shader_id;
        GLuint vertex_buffer_id;

        std::vector<int> face_vertex_buffer_idxs;
        std::vector<glm::vec3> pts_host;

        public:
        FaceDrawer2D(){

            // Generate the vertex buffer 
            // this can store an array of vertex points on the graphics adaptor memory
            glGenBuffers(1, &vertex_buffer_id);

            // Generate the Vertex Array Object 
            // stores information about the how to interpret vertex buffer objects 
            glGenVertexArrays(1, &vertex_array_id);

            // load the shaders 
            face_shader_id = LoadShaders("./shaders/face_shader_2d.vert", "./shaders/face_shader_2d.frag");

            // push back the first index into the list of indices for face vertices 
            face_vertex_buffer_idxs.push_back(0);
        }


        template<typename T1>
        T1 lerp(T1 v0, T1 v1, T1 t){
            return v0 + t * (v1 - v0);
        }

        template<typename T1>
        T1 inv_lerp(T1 v0, T1 v1, T1 val){
            return (val - v0) / (v1 - v0);
        }

        /**
         * @brief add the given face 
         * This computes the vertices to draw the face onto the screen and stores this 
         * @param facptr the face to draw 
         * @param coord the node coordinates array 
         * @param mesh_bounds the bounding box of the mesh 
         * @param scaled_bounds the bounds in the viewport to scale the mesh bounding box to
         * @param nsegment the number of segments to split each face into
         */
        void add_face(
            ELEMENT::Face<T, IDX, 2> *facptr,
            FE::NodalFEFunction<T, 2> &coord,
            BoundingBox mesh_bounds,
            BoundingBox scaled_bounds,
            int nsegment=4 
        ) {

#ifndef NDEBUG 
            std::cout << "Adding Face:" << std::endl;
#endif
            // generate the points
            double dx = 2.0 / nsegment;
            for(int ipt = 0; ipt < nsegment + 1; ++ipt){
                MATH::GEOMETRY::Point<T, 1> s = {dx * ipt - 1.0};
                MATH::GEOMETRY::Point<T, 2> x;
                facptr->transform(s, coord, x.data());

                // scale by bounding box 
                GLfloat maxlen = 0;
                for(int idim = 0; idim < 2; ++idim) maxlen = std::max(maxlen, mesh_bounds.xmax[idim] - mesh_bounds.xmin[idim]);
                for(int idim = 0; idim < 2; ++idim){
                    GLfloat t = inv_lerp(mesh_bounds.xmin[idim], mesh_bounds.xmax[idim], static_cast<GLfloat>(x[idim]));
                    GLfloat len = mesh_bounds.xmax[idim] - mesh_bounds.xmin[idim];
                    // adjust t to keep aspect ratio 
                    t = (t - 0.5) * len / maxlen + 0.5;
                    x[idim] = lerp(scaled_bounds.xmin[idim], scaled_bounds.xmax[idim], t);
                }

                pts_host.emplace_back(static_cast<GLfloat>(x[0]), static_cast<GLfloat>(x[1]), 0.0);
#ifndef NDEBUG
                std::cout << "s: [ " << s[0] << " ] | x: [ " << x[0] << " " << x[1] << " ]" << std::endl;
#endif // !NDEBUG
            }
            // store the end of the current face
            face_vertex_buffer_idxs.push_back(pts_host.size()); 
        }

        /** @brief clear stored faces */
        void clear_faces(){ pts_host.clear(); }

        /* 
         * @brief upate the gpu buffer after adding faces 
         */
        void update(){
            glBindVertexArray(vertex_array_id);
            glVertexAttribPointer(
                0,                  // attribute
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                3 * sizeof(GLfloat),                  // stride
                (void*)0            // array buffer offset
            );
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            glBufferData(GL_ARRAY_BUFFER, pts_host.size() * sizeof(glm::vec3), pts_host.data(), GL_STATIC_DRAW);
        }

        void draw_faces(){
            // clear the screen 
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            // use the shaders 
            glUseProgram(face_shader_id);

            // attribute buffer for vertices 
            glBindVertexArray(vertex_array_id);
            glVertexAttribPointer(
                0,                  // attribute
                3,                  // size
                GL_FLOAT,           // type
                GL_FALSE,           // normalized?
                3 * sizeof(GLfloat),                  // stride
                (void*)0            // array buffer offset
            );
            glEnableVertexAttribArray(0);

            glLineWidth(1.0f);

            // vertex buffer 
            glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer_id);
            for(int ipt_end = 1; ipt_end < face_vertex_buffer_idxs.size(); ++ipt_end){
                GLsizei count = face_vertex_buffer_idxs[ipt_end] - face_vertex_buffer_idxs[ipt_end - 1]; 
                glDrawArrays(GL_LINE_STRIP, face_vertex_buffer_idxs[ipt_end - 1], count);
            }
            glDisableVertexAttribArray(0);
        }

        ~FaceDrawer2D(){
            glDeleteBuffers(1, &vertex_buffer_id);
            glDeleteVertexArrays(1, &vertex_array_id);
        }
    };

}

namespace FE {
    template<typename T>
    std::vector< glm::vec3 > to_opengl_vertices(const FE::NodalFEFunction<T, 2> &nodes){
        std::vector< glm::vec3 > out;
        out.reserve(nodes.n_nodes());
        for(int i = 0; i < nodes.n_nodes(); ++i){
            out.emplace_back(nodes[i][0], nodes[i][1], 0.0);
        }
        return out;
    }
}
