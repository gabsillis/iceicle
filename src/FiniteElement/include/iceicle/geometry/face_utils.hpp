/**
 * @brief general utility functions for faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "Numtool/point.hpp"
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/fe_definitions.hpp>
#include <iceicle/geometry/face.hpp>
#include <algorithm>
#include <utility>

namespace iceicle {

    /**
     * @brief get the normal vector at a given point 
     * @param face the face to get the normal vector to
     * @param coord the global node coordinates array
     * @param face_point the point on the face where the normal will be calculated
     * @return the normal vector to the face at the given point 
     */
    template<typename T, typename IDX, int ndim>
    NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> calc_normal(
        const Face<T, IDX, ndim> &face,
        NodeArray<T, ndim> &coord,
        const MATH::GEOMETRY::Point<T, ndim - 1> &face_point
    ){
        auto J = face.Jacobian(coord, face_point);
        return NUMTOOL::TENSOR::FIXED_SIZE::calc_ortho(J);
    }

    /**
     * @brief get the centroid in the reference domain of the face 
     * @param face a const reference to the face
     * @return the centroid of that face in the reference domain 
     */
    template<typename T, typename IDX, int ndim>
    MATH::GEOMETRY::Point<T, ndim - 1> ref_face_centroid(
        const Face<T, IDX, ndim> &face
    ) {
        MATH::GEOMETRY::Point<T, ndim - 1> centroid;
        switch(face.domain_type()){
            case DOMAIN_TYPE::HYPERCUBE:
                std::fill_n(centroid.data(), ndim - 1, 0.0);
                return centroid;
            case DOMAIN_TYPE::SIMPLEX:
                std::fill_n(centroid.data(), ndim - 1, 1.0 / 3.0);
                return centroid;
            default:
                std::unreachable();
        }
        return centroid;
    }


    /**
     * @brief get the centroid in the physical domain of the face 
     * @param face a const reference to the face
     * @param coord the global node coordinates array
     * @return the centroid of that face in the physical domain 
     */
    template<typename T, typename IDX, int ndim>
    MATH::GEOMETRY::Point<T, ndim> face_centroid(
        const Face<T, IDX, ndim>& face,
        NodeArray<T, ndim>& coord
    ) {
        using FacePoint = MATH::GEOMETRY::Point<T, ndim - 1>;
        FacePoint c_ref = ref_face_centroid(face);
        MATH::GEOMETRY::Point<T, ndim> c_phys;
        face.transform(c_ref, coord, c_phys);
        return c_phys;
    }
}
