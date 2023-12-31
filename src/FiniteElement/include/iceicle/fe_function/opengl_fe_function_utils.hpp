#pragma once 
#include <iceicle/fe_function/nodal_fe_function.hpp>
#include <iceicle/geometry/face.hpp>
#include <vector>
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>

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

    /**
     * @brief draw a 2D face into the current frame 
     * TODO: split into a load phase to load a large array of all the faces and send to device 
     * then a draw phase 
     *
     * @param facptr the pointer to the face object
     * @param coord the node coordinates
     * @param nsegment the number of segments to split this into (default = 10)
     */
    template<typename T, typename IDX>
    inline void draw_face_2d(
        ELEMENT::Face<T, IDX, 2> *facptr,
        FE::NodalFEFunction<T, 2> &coord,
        int nsegment=10 
    ){
        // generate the gl arrays
        GLuint vertex_array_id;
        glGenVertexArrays(1, &vertex_array_id);
        glBindVertexArray(vertex_array_id);

        // generate the points
        glm::vec3 *pts_host = new glm::vec3[nsegment + 1];
        double dx = 2.0 / nsegment;
        for(int ipt = 0; ipt < nsegment + 1; ++ipt){
            MATH::GEOMETRY::Point<T, 1> s = {dx * ipt};
            MATH::GEOMETRY::Point<T, 2> x;
            facptr->transform(s, coord, x);
            pts_host[ipt] = {static_cast<GLfloat>(x[0]), static_cast<GLfloat>(x[1]), 0.0};
        }

        // send to device
        GLuint vertex_buffer;
        glGenBuffers(1, &vertex_buffer);
        glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
        glBufferData(
            GL_ARRAY_BUFFER,
            (nsegment + 1) * sizeof(glm::vec3),
            &pts_host[0],
            GL_STATIC_DRAW
        );

        // draw the lines 
        glDrawArrays(GL_LINE_STRIP, 0, nsegment+1);

        //cleanup
        delete[] pts_host;
        glDeleteBuffers(1, &vertex_buffer);
        glEnableVertexAttribArray(0);
        glBindVertexArray(0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

    }

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
