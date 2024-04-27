/**
 * @file hypercube_face.hpp
 * @brief Hypercube faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/geometry/geo_element.hpp"
#include <iceicle/transformations/HypercubeTransformations.hpp>
#include <iceicle/geometry/face.hpp>
#include <valarray>
namespace iceicle {
    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeFace final : public Face<T, IDX, ndim> {
        template<int size>
        using IndexArrayType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, size>;
        using FaceBase = Face<T, IDX, ndim>;
        using Point = FaceBase::Point;
        using FacePoint = FaceBase::FacePoint;
        using JacobianType = FaceBase::JacobianType;

        public:
        inline static transformations::HypercubeTraceOrientTransformation<T, IDX, ndim> orient_trans{};
        inline static transformations::HypercubeTraceTransformation<T, IDX, ndim, Pn> trans{};

        /// The global node coordinates
        IndexArrayType<trans.n_nodes> _nodes;

        /// @brief construct a hypercube face
        /// @param elemL the index of the left element
        /// @param elemR the index of the right element 
        /// @param nodes the node indices of the face
        /// @param faceNrL the face number for the left element 
        /// @param faceNrR the face number for the right element
        /// @param bctype the boundary condition type 
        /// @param bcflag the integer flag for the boundary condition
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

        /// @brief construct a hypercube face 
        /// using Given Geometric elements to help calculate orientation
        /// @param elemL the index of the left element 
        /// @param elemR the index of the right element
        HypercubeFace(
            IDX elemL, IDX elemR,
            GeometricElement<T, IDX, ndim> *elptrL,
            GeometricElement<T, IDX, ndim> *elptrR,
            int faceNrL, int faceNrR,
            BOUNDARY_CONDITIONS bctype,
            int bcflag
        ): Face<T, IDX, ndim>(elemL, elemR,
            faceNrL * FACE_INFO_MOD,
            faceNrR * FACE_INFO_MOD + [elptrL, elptrR, faceNrL, faceNrR]() -> int {
                IndexArrayType<orient_trans.nvert_tr> vert_l{}, vert_r{};
                elptrL->get_face_vert(faceNrL, vert_l.data());
                elptrR->get_face_vert(faceNrR, vert_r.data());
                return orient_trans.getOrientation(vert_l.data(), vert_r.data());
            }(),
            bctype, bcflag), _nodes{[elptrL, faceNrL](){
                IndexArrayType<trans.n_nodes> nodes{};
                elptrL->get_face_nodes(faceNrL, nodes.data());
                return nodes;
            }()}
        {}

//        /**
//         * @brief create a hypercube face 
//         * @param elemL the left element 
//         * @param elemR the right element 
//         * @param faceNrL the face number for the left element 
//         * @param faceNrR the face number for the right element
//         * @param orientR the orientation of the right face 
//         * @param bctype the boundary condition type 
//         * @param bcflag the integer flag to define additional information for the boundary condition 
//         */
//        HypercubeFace(
//            IDX elemL, IDX elemR, 
//            int faceNrL, int faceNrR,
//            int orientR, 
//            BOUNDARY_CONDITIONS bctype,
//            int bcflag
//        ) : Face<T, IDX, ndim>(elemL, elemR,
//            faceNrL * FACE_INFO_MOD,
//            faceNrR * FACE_INFO_MOD + orientR,
//            bctype, bcflag)
//        {}


        constexpr DOMAIN_TYPE domain_type() const override { return DOMAIN_TYPE::HYPERCUBE; }

        constexpr int geometry_order() const noexcept override { return Pn; }
        void transform(
            const FacePoint &s,
            NodeArray<T, ndim> &coord,
            Point& result
        ) const override {
            trans.transform_physical(
                _nodes.data(),
                this->face_infoL / FACE_INFO_MOD,
                s, coord, result
            );
        }

        void transform_xiL(
            const FacePoint &s,
            Point& result
        ) const override {
            trans.transform(
                _nodes.data(),
                this->face_infoL / FACE_INFO_MOD,
                s.data(), result);
        }

        void transform_xiR(
            const FacePoint &s,
            Point& result
        ) const override {
            FacePoint sR;
            orient_trans.transform(
                this->face_infoR % FACE_INFO_MOD,
                s, sR);
            trans.transform(
                _nodes.data(),
                this->face_infoR / FACE_INFO_MOD,
                sR.data(), result);
        }


        JacobianType Jacobian(
            NodeArray<T, ndim> &node_coords,
            const FacePoint &s
        ) const override {
            return trans.Jacobian(
                node_coords,
                _nodes.data(),
                this->face_infoL / FACE_INFO_MOD,
                s
            );
        }

        int n_nodes() const override { return trans.n_nodes; }

        const IDX *nodes() const override { return _nodes.data(); }
    };
}
