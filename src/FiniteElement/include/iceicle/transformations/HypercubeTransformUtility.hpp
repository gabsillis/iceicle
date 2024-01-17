/**
 * @brief utilities for using HypercubeTransform
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "Numtool/point.hpp"
#include <Numtool/tmp_flow_control.hpp>
#include <iceicle/transformations/HypercubeElementTransformation.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <iceicle/fe_function/nodal_fe_function.hpp>

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>

namespace ELEMENT::TRANSFORMATIONS {
   
    /**
     * @brief append a hypercube element to the coordinate array 
     * and return a HypercubeElement newly created to specified size 
     * @param centroid the centroid of the hypercube 
     * @param lengths the length of the cube in each coordinate direction
     * @param [in/out] coord the global coordinate array
     * @return a HypercubeElement mapped to coord 
     */
    template<typename T, typename IDX, int ndim, int Pn>
    HypercubeElement<T, IDX, ndim, Pn> create_hypercube(
        T centroid[ndim],
        T lengths[ndim], 
        FE::NodalFEFunction<T, ndim> &coord
    ) {
        HypercubeElement<T, IDX, ndim, Pn> el{};
        HypercubeElementTransformation<T, IDX, ndim, Pn> &trans = el.transformation;
        std::size_t nnode = trans.n_nodes();
        std::size_t start = coord.n_nodes(); // start at current coord
        coord.resize(coord.n_nodes() + trans.n_nodes());

        for(int inode = 0; inode < nnode; ++inode){
            const auto &pt = trans.reference_nodes()[inode];
            for(int idim = 0; idim < ndim; ++idim){
                coord[start + inode][idim] = centroid[idim] + lengths[idim] / 2.0 * pt[idim];
            }

            el.setNode(inode, start + inode);
        }

        return el;
    }

    template<typename T, typename IDX, int Pn>
    void triangle_decompose_quad(
        HypercubeElement<T, IDX, 2, Pn> &quad,
        FE::NodalFEFunction<T, 2> &coord,
        int refine_order,
        std::vector< unsigned int > &vertexIndices,
        std::vector< glm::vec3 > &vertices
    ){
        if (Pn == refine_order){
            auto &trans = quad.transformation;
            std::size_t vi_offset = vertices.size();

            // split each subquad into two triangles 
            for(int iquad = 0; iquad < Pn; ++iquad){
                for(int jquad = 0; jquad < Pn; ++jquad){
                    // first triangle 
                    int ijk1[3] = {iquad, jquad};
                    int ijk2[3] = {iquad+1, jquad};
                    int ijk3[3] = {iquad+1, jquad+1};
                    // push back the triangle indices 

                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk1));
                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk2));
                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk3));

                    // second triangle 
                    int ijk4[3] = {iquad, jquad};
                    int ijk5[3] = {iquad+1, jquad+1};
                    int ijk6[3] = {iquad, jquad+1};
                    // push back the triangle indices 
                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk4));
                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk5));
                    vertexIndices.emplace_back(
                        vi_offset + trans.convert_indices_helper(ijk6));
                }
            }

            for(int inode = 0; inode < trans.n_nodes(); ++inode){
                vertices.emplace_back(
                    (GLfloat) coord[quad.nodes()[inode]][0],
                    (GLfloat) coord[quad.nodes()[inode]][1],
                    (GLfloat) 0.0
                );
            }
        } else {
            auto convert_ijk = [&](int ijk[2]) -> int {
                return ijk[0] * refine_order + ijk[1];
            };

            for(int iquad = 0; iquad < refine_order; ++iquad){
                for(int jquad = 0; jquad < refine_order; ++jquad){

                    std::size_t vi_offset = vertices.size();

                    // first triangle 
                    int ijk1[2] = {iquad, jquad};
                    int ijk2[2] = {iquad+1, jquad};
                    int ijk3[2] = {iquad+1, jquad+1};

                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk1));
                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk2));
                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk3));

                    // second triangle 
                    int ijk4[2] = {iquad, jquad};
                    int ijk5[2] = {iquad+1, jquad+1};
                    int ijk6[2] = {iquad, jquad+1};

                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk4));
                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk5));
                    vertexIndices.emplace_back(vi_offset + convert_ijk(ijk6));

                }
            }

            for(int ivert = 0; ivert < refine_order + 1; ++ivert){
                for(int jvert = 0; jvert < refine_order + 1; ++jvert){
                    MATH::GEOMETRY::Point<T, 2> xi = {
                        -1.0 + ivert * 2.0 / refine_order,
                        -1.0 + jvert * 2.0 / refine_order 
                    };

                    MATH::GEOMETRY::Point<T, 2> x;
                    quad.transform(coord, xi, x);
                    vertices.emplace_back(
                        (GLfloat) x[0],
                        (GLfloat) x[1],
                        (GLfloat) 0.0
                    );
                }
            }
        }
    }
}
