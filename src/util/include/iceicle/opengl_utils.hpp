#pragma once

#include <algorithm>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

#include <Numtool/point.hpp>
#include <Numtool/fixed_size_tensor.hpp>

namespace ICEICLE_GL {

    /**
     * @brief convert a Numtool Point to a opengl vec3
     */
    template<typename T, std::size_t ndim>
    glm::vec3 to_vec3(const MATH::GEOMETRY::Point<T, ndim> &pt){
        glm::vec3 out;
        std::fill_n(&out[0], 3, 0.0);
        for(int i = 0; i < std::min(ndim, (std::size_t) 3); ++i){
            out[i] = static_cast<GLfloat>(pt[i]);
        }
        return out;
    }


    /**
     * @brief convert a Numtool Fixed Size Tensor to a opengl vec3
     */
    template<typename T, std::size_t ndim>
    glm::vec3 to_vec3(const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> &tensor){
        glm::vec3 out;
        std::fill_n(&out[0], 3, 0.0);
        for(int i = 0; i < std::min(ndim, (std::size_t) 3); ++i){
            out[i] = static_cast<GLfloat>(tensor[i]);
        }
        return out;
    }

}
