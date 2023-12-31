/**
 * @file hypercube_face.hpp
 * @brief Hypercube faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iceicle/transformations/HypercubeElementTransformation.hpp>
#include <iceicle/geometry/face.hpp>

#include "iceicle/fe_function/nodal_fe_function.hpp"
#include "iceicle/transformations/HypercubeElementTransformation.hpp"
namespace ELEMENT {
    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeFace final : public Face<T, IDX, ndim> {
        template<int size>
        using IndexArrayType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, size>;
        using FaceBase = Face<T, IDX, ndim>;
        using FacePoint = FaceBase::FacePoint;
        using JacobianType = FaceBase::JacobianType;

        public:
        inline static TRANSFORMATIONS::HypercubeTraceOrientTransformation<T, IDX, ndim> orient_trans{};
        inline static TRANSFORMATIONS::HypercubeTraceTransformation<T, IDX, ndim, Pn> trans{};

        /// The global node coordinates
        IndexArrayType<trans.n_nodes> _nodes;

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
            bctype, bcflag), _nodes(nodes) {}


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

        void transform(
            const FacePoint &s,
            FE::NodalFEFunction<T, ndim> &coord,
            T *result
        ) const override {
            MATH::GEOMETRY::PointView<T, ndim> result_view{result};
            trans.transform_physical(
                _nodes.data(),
                this->face_infoL / ELEMENT::FACE_INFO_MOD,
                s, coord, result_view
            );
        }

        void transform_xiL(
            const FacePoint &s,
            T *result
        ) const override {
            MATH::GEOMETRY::PointView<T, ndim> result_view{result};
            trans.transform(
                _nodes.data(),
                this->face_infoL / ELEMENT::FACE_INFO_MOD,
                s.data(), result_view);
        }

        void transform_xiR(
            const FacePoint &s,
            T *result
        ) const override {
            FacePoint sR;
            orient_trans.transform(
                this->face_infoR % ELEMENT::FACE_INFO_MOD,
                s, sR);
            MATH::GEOMETRY::PointView<T, ndim> result_view{result};
            trans.transform(
                _nodes.data(),
                this->face_infoR / ELEMENT::FACE_INFO_MOD,
                sR.data(), result_view);
        }


        JacobianType Jacobian(
            FE::NodalFEFunction<T, ndim> &node_coords,
            const FacePoint &s
        ) const override {
            return trans.Jacobian(
                node_coords,
                _nodes.data(),
                this->face_infoL / ELEMENT::FACE_INFO_MOD,
                s
            );
        }

        int n_nodes() const override { return trans.n_nodes; }

        IDX *nodes() override { return _nodes.data(); }

    };
}
