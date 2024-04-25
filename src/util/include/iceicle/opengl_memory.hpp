#pragma once

#include <GL/glew.h>
#include <glm/glm.hpp>
#include <GLFW/glfw3.h>
#include <map>

namespace iceicle::gl {

    /**
     * @brief get the data type enumeration from the built-in type
     * @return the OpenGL enumeration for type T
     */
    template<typename T>
    GLenum get_data_type(){
        if constexpr (std::is_same<T, GLfloat>()){
            return GL_FLOAT;
        }

        if constexpr (std::is_same<T, GLdouble>()){
            return GL_DOUBLE;
        }


        if constexpr (std::is_same<T, GLuint>()){
            return GL_UNSIGNED_INT;
        }

        // TODO: all types
    }

    /**
     * @brief class that represents a vertex buffer object 
     * These are gpu memory resources that get accessed by the shader programs
     */
    struct BufferObject {

        /// the openGL id for the VBO
        GLuint id;

        /// the type of buffer (GL_ARRAY_BUFFER, GL_ELEMENT_BUFFER)
        GLenum type;

        /*
         * @brief create a new BufferObject 
         *
         */
        BufferObject(GLenum type = GL_ARRAY_BUFFER) : type(type) {
            glGenBuffers(1, &id);
        }

        /// bind the buffer object
        void bind() { glBindBuffer(type, id); }


        /**
         * @brief set the vertex attribute pointer
         * @param attr_id the index of this data as it will appear in the shader 
         * @param size the number of components per generic vertex attribute
         * @param stride the offset between generic vertex attributes 
         * the byte length of the type is taken care of (i.e. don't need to multiply by sizeof(GL_FLOAT))
         * @param offset the offfset of the first component of the first genereic vertex attribute
         * (again byte length of the type is taken care of)
         */
        template<typename T>
        void set_attr_pointer(GLuint attr_id, GLuint size, GLuint stride = 0, GLuint offset = 0){
            glVertexAttribPointer(
                attr_id,
                size,
                get_data_type<T>(), 
                GL_FALSE,
                stride * sizeof(T),
                reinterpret_cast<void *>(offset * sizeof(T))
            );
        }

        /**
         * @brief send data to the GPU buffer
         * @tparam T the type of data being buffered
         * @param nelem the number of items of size T to buffer 
         * @param data the data to copy to the gpu 
         * @param usage how this data is expected to be used 
         *  The symbolic constant must be GL_STREAM_DRAW, GL_STREAM_READ, GL_STREAM_COPY, GL_STATIC_DRAW,
         *  GL_STATIC_READ, GL_STATIC_COPY, GL_DYNAMIC_DRAW, GL_DYNAMIC_READ, or GL_DYNAMIC_COPY.
         */
        template<typename T>
        void buffer_data(GLuint nelem, T *data, GLenum usage){
            glBufferData(type, nelem * sizeof(T), data, usage);
        }

        BufferObject(const BufferObject &other) = delete;

        BufferObject &operator= (const BufferObject &other) = delete;

        ~BufferObject(){
            glDeleteBuffers(1, &id);
        }

    };

    /**
     * @brief class that represents a vertex array object 
     * Which specifies the format of the vertex data and the buffer objects
     */
    struct ArrayObject {

        // The openGL id for this VAO
        GLuint id;

        // dictionary of buffers 
        // WARNING: make sure to bind() before using/modifying
        std::map<const char *, BufferObject> buffers;

        ArrayObject(){
            glGenVertexArrays(1, &id);
        }

        ArrayObject(const ArrayObject &other) = delete;

        ArrayObject &operator=(const ArrayObject &other) = delete;

        /** @brief get the given buffer in the dictionary of buffers */
        BufferObject& operator[](const char *key){
            return buffers[key];
        }

        // @brief bind this array object 
        void bind(){
            glBindVertexArray(id);
        }
    };
}
