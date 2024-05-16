/**
 * @brief general utility functions for faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "Numtool/point.hpp"
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/build_config.hpp"
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/fe_definitions.hpp>
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_face.hpp>
#include <iceicle/anomaly_log.hpp>
#include <iceicle/algo.hpp>
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

    /// @brief compute the face identifiers (domain type, left face number, right face number, right orientation)
    /// or std::nullopt if there is no intersection of the given elements
    /// @param elptrL pointer to the left geometric element 
    /// @param elptrR pointer to the right geometric element 
    /// @return a tuple with respectively:
    ///     the domain type of the face
    ///     the left face number 
    ///     the right face number 
    ///     the orientation of the right element
    template<class T, class IDX, int ndim>
    [[nodiscard]] inline constexpr 
    auto intersect_face_info(
        GeometricElement<T, IDX, ndim> *elptrL,
        GeometricElement<T, IDX, ndim> *elptrR
    ) noexcept -> std::optional<std::tuple<DOMAIN_TYPE, int, int, int>> {
        for(int iface_l = 0; iface_l < elptrL->n_faces(); ++iface_l){
            // get the vertices for the left element
            std::vector<IDX> face_vert_l(elptrL->n_face_vert(iface_l));
            elptrL->get_face_vert(iface_l, face_vert_l.data());
            for(int iface_r = 0; iface_r < elptrR->n_faces(); ++iface_r){
                // get the vertices for the right element
                std::vector<IDX> face_vert_r(elptrR->n_face_vert(iface_r));
                elptrR->get_face_vert(iface_r, face_vert_r.data());
                if(util::eqset(face_vert_l, face_vert_r)){
                    // found a vertex match
                    DOMAIN_TYPE domn_type = elptrL->face_domain_type(iface_l);
                    if(domn_type != elptrR->face_domain_type(iface_r)){
                        util::AnomalyLog::log_anomaly(util::Anomaly{"All vertices match, but face domain types differ", util::general_anomaly_tag{}});
                        return std::nullopt;
                    }

                    // get the orientation and return all information found
                    switch(domn_type){
                        case DOMAIN_TYPE::HYPERCUBE:
                        {
                            int orient_r = hypercube_orient_trans<T, IDX, ndim>.getOrientation(face_vert_l.data(), face_vert_r.data());
                            return std::tuple{domn_type, iface_l, iface_r, orient_r};
                        }

                        default:
                        {
                            util::AnomalyLog::log_anomaly(util::Anomaly{"orientation deduction not supported for the given domain type", util::general_anomaly_tag{}});
                            return std::nullopt;
                        }
                    }
                } 
            }
        }
        return std::nullopt;
    }

    /// @brief Make the face that corresponds to the intersection of the two given elements 
    /// or std::nullopt if there is no intersection of the given elements
    /// @param elemL the index of the left element 
    /// @param elemR the index of the right element 
    /// @param elptrL the pointer to the left element 
    /// @param elptrR the pointer to the right element 
    /// @param bctype the boundary condition type 
    /// @param bcflag the integer falg for the boundary condition
    /// @return a pointer to the created face if applicable
    template<class T, class IDX, int ndim>
    [[nodiscard]] inline constexpr 
    auto make_face(
        IDX elemL,
        IDX elemR,
        GeometricElement<T, IDX, ndim> *elptrL,
        GeometricElement<T, IDX, ndim> *elptrR,
        BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::INTERIOR,
        int bcflag = 0
    ) noexcept -> std::optional<Face<T, IDX, ndim>*> {

        // look for matching vertices 
        auto face_info_match = intersect_face_info(elptrL, elptrR);
        if(face_info_match){
            auto [fac_domn, face_nr_l, face_nr_r, orient_r] = face_info_match.value();
            switch(fac_domn){
                case DOMAIN_TYPE::HYPERCUBE:
                {
                    // NOTE: we use the minimum geometry order for the face so that the intersection of 
                    // different order elements results in well defined mappings from both sides.
                    // (The higher order element will always have a well-posed representation of lower order geometry)
                    // This does, however, require the higher order element to conform to the lower order geometry
                    int min_geo_order;
                    GeometricElement<T, IDX, ndim>* min_order_elptr;
                    if(elptrL->geometry_order() <= elptrR->geometry_order()){
                        min_geo_order = elptrL->geometry_order();
                        min_order_elptr = elptrL;
                    } else {
                        min_geo_order = elptrR->geometry_order();
                        min_order_elptr = elptrR;
                    }
                    
                    auto create_face = [&]<int Pn> -> std::optional<Face<T, IDX, ndim> *>{
                        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
                        using FaceType = HypercubeFace<T, IDX, ndim, Pn>;
                        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, FaceType::trans.n_nodes> nodes{};
                        min_order_elptr->get_face_nodes(face_nr_l, nodes.data());
                        return std::optional{new FaceType(elemL, elemR, nodes, face_nr_l, face_nr_r, orient_r, bctype, bcflag) };
                    };

                    return NUMTOOL::TMP::invoke_at_index<int, 0, build_config::FESPACE_BUILD_GEO_PN>(
                        min_geo_order, create_face);
                }
                default:
                {
                    util::AnomalyLog::log_anomaly(util::Anomaly{"unsupported face domain type", util::general_anomaly_tag{}});
                    return std::nullopt;
                }
            }
        } else {
            return std::nullopt;
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
