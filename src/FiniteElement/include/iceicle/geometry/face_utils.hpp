/**
 * @brief general utility functions for faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "Numtool/point.hpp"
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/fe_definitions.hpp>
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_face.hpp>
#include <iceicle/anomaly_log.hpp>
#include <algorithm>
#include <utility>

namespace iceicle {

    /// @brief Make the face that corresponds to the intersection of the two given elements 
    /// @param elemL the index of the left element 
    /// @param elemR the index of the right element 
    /// @param elptrL the pointer to the left element 
    /// @param elptrR the pointer to the right element 
    /// @param faceNrL the face number of the left element 
    /// @param faceNrR the face number of the right element
    /// @param bctype the boundary condition type 
    /// @param bcflag the integer falg for the boundary condition
    template<class T, class IDX, int ndim>
    auto make_face(
        IDX elemL,
        IDX elemR,
        GeometricElement<T, IDX, ndim> *elptrL,
        GeometricElement<T, IDX, ndim> *elptrR,
        int faceNrL,
        int faceNrR,
        BOUNDARY_CONDITIONS bctype,
        int bcflag
    ) -> Face<T, IDX, ndim>* {
        using namespace NUMTOOL::TMP;
        if(elptrL->domain_type() == DOMAIN_TYPE::HYPERCUBE && elptrR->domain_type() == DOMAIN_TYPE::HYPERCUBE) {
            static constexpr int geo_pn_last = build_config::FESPACE_BUILD_GEO_PN + 1;
            return invoke_at_index<int, 0, geo_pn_last>(
                std::max(elptrL->geometry_order(), elptrR->geometry_order()),
                [&]<int geo_pn>{
                    return new HypercubeFace<T, IDX, ndim, geo_pn>(
                        elemL, elemR, elptrL, elptrR, faceNrL, faceNrR,
                        bctype, bcflag
                    );
                }
            );
        } else {
            util::AnomalyLog::log_anomaly(
                    util::Anomaly{"No matching face constructor for given domain types",
                    util::general_anomaly_tag{}});
        }
    }
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
