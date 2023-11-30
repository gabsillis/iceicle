#pragma once 
#include <iceicle/fe_function/nodal_fe_function.hpp>
#include <vector>
#include <glm/glm.hpp>
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
