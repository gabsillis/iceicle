/**
 * @file hypercube_face.hpp
 * @brief Hypercube faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iceicle/transformations/HypercubeElementTransformation.hpp>
#include <iceicle/geometry/face.hpp>

#include "iceicle/transformations/HypercubeElementTransformation.hpp"
namespace ELEMENT {
    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeFace final : public Face<T, IDX, ndim> {

        public:
        inline static TRANSFORMATIONS::HypercubeTraceOrientTransformation<T, IDX, ndim> orient_trans{};
        inline static TRANSFORMATIONS::HypercubeTraceTransformation<T, IDX, ndim, Pn> trans{};

        HypercubeFace(
            IDX elemL, IDX elemR, 
            int faceNrL, int faceNrR,
            int orientR, 
            BOUNDARY_CONDITIONS bctype,
            int bcflag
        ) : Face<T, IDX, ndim>(elemL, elemR,
            faceNrL * FACE_INFO_MOD,
            faceNrR * FACE_INFO_MOD + orientR,
            bctype, bcflag)
        {}


    };
}
