#pragma once

#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "hypercube_face.hpp"
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

}
