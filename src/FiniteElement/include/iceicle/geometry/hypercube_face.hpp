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
        template<int size>
        using IndexArrayType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, size>;

        public:
        inline static TRANSFORMATIONS::HypercubeTraceOrientTransformation<T, IDX, ndim> orient_trans{};
        inline static TRANSFORMATIONS::HypercubeTraceTransformation<T, IDX, ndim, Pn> trans{};

        /// The global node coordinates
        IndexArrayType<trans.n_nodes> nodes;

        HypercubeFace(
            IDX elemL, 
            IDX elemR,
            IndexArrayType<trans.n_nodes> &nodes,
            int faceNrL, int faceNrR,
            int orientR,
            BOUNDARY_CONDITIONS bctype,
            int bcflag
        ) : Face<T, IDX, ndim>(elemL, elemR,
            faceNrL * FACE_INFO_MOD,
            faceNrR * FACE_INFO_MOD + orientR,
            bctype, bcflag), nodes(nodes) {}


        /**
         * @brief create a hypercube face 
         * @param elemL the left element 
         * @param elemR the right element 
         * @param faceNrL the face number for the left element 
         * @param faceNrR the face number for the right element
         * @param orientR the orientation of the right face 
         * @param bctype the boundary condition type 
         * @param bcflag the integer flag to define additional information for the boundary condition 
         */
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
