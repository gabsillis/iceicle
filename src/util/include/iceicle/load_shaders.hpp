#pragma once
#include <GL/glew.h>
#include <glm/glm.hpp>
#include <string>
#include <iostream>
GLuint LoadShaders(const char * vertex_file_path,const char * fragment_file_path);

namespace ICEICLE_GL{
    /**
    * @brief a general purpose shader object that represents a GLSL shader
    */
    class Shader {
        public:

        /// the opengl id for this shader
        GLuint id;

        /**
         * @brief construct a shader object from the shader sources
         * compiles the given sources 
         * @param vertex_source the source code for the vertex shader 
         * @param fragment_source the source code for the fragment shader 
         * @param geometry_source the source code for the geometry shader (optional)
         */
        Shader(
            const char *vertex_source,
            const char *fragment_source,
            const char *geometry_source
        );

        /** 
         * @brief load the current shader onto the GPU for use 
         * @return a reference to the current shader object
         **/
        Shader &load();

        /** @brief delete this shader (frees opengl resources) */
        ~Shader();

        // Delete Copying because this represents a GPU resource 

        Shader(const Shader &other) = delete;

        Shader& operator=(const Shader &other) = delete;

        /** from https://github.com/michaelg29/glmathviz */ 
        /*
            set uniform variables
        */

        /** 
         * @brief set a boolean uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void setBool(const std::string& name, bool value) {
            glUniform1i(glGetUniformLocation(id, name.c_str()), (int)value);
        }

        /** 
         * @brief set a integer uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void setInt(const std::string& name, int value) {
            glUniform1i(glGetUniformLocation(id, name.c_str()), value);
        }

        /** 
         * @brief set a float uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void setFloat(const std::string& name, float value) {
            glUniform1f(glGetUniformLocation(id, name.c_str()), value);
        }

        /** 
         * @brief set a vec3 uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void set3Float(const std::string& name, float v1, float v2, float v3) {
            glUniform3f(glGetUniformLocation(id, name.c_str()), v1, v2, v3);
        }

        /** 
         * @brief set a vec3 uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void set3Float(const std::string& name, glm::vec3 v) {
            glUniform3f(glGetUniformLocation(id, name.c_str()), v.x, v.y, v.z);
        }

        /** 
         * @brief set a vec4 uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void set4Float(const std::string& name, float v1, float v2, float v3, float v4) {
            glUniform4f(glGetUniformLocation(id, name.c_str()), v1, v2, v3, v4);
        }

        /** 
         * @brief set a vec4 uniform variable (a shader constant)
         * @param name the name of the variable
         * @param value the value to set 
         */
        void set4Float(const std::string& name, glm::vec4 v) {
            glUniform4f(glGetUniformLocation(id, name.c_str()), v.x, v.y, v.z, v.w);
        }

        private:

        /**
         * Check for compile errors 
         * reference: https://github.com/JoeyDeVries/LearnOpenGL
         * @param shader_id the id to check 
         * @param type the type of shader being compiled 
         */
        void checkCompileErrors(GLuint shader_id, std::string type){
            int success;
            char infoLog[1024];
            if (type != "PROGRAM")
            {
                glGetShaderiv(shader_id, GL_COMPILE_STATUS, &success);
                if (!success)
                {
                    glGetShaderInfoLog(shader_id, 1024, NULL, infoLog);
                    std::cerr << "| ERROR::SHADER: Compile-time error: Type: " << type << "\n"
                        << infoLog << "\n -- --------------------------------------------------- -- "
                        << std::endl;
                }
            }
            else
            {
                glGetProgramiv(shader_id, GL_LINK_STATUS, &success);
                if (!success)
                {
                    glGetProgramInfoLog(shader_id, 1024, NULL, infoLog);
                    std::cerr << "| ERROR::Shader: Link-time error: Type: " << type << "\n"
                        << infoLog << "\n -- --------------------------------------------------- -- "
                        << std::endl;
                }
            }
        }

    };
}
