/**
 * @file hypercube_face.hpp
 * @brief Hypercube faces
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include <iceicle/transformations/HypercubeTransformations.hpp>
#include <iceicle/geometry/face.hpp>
namespace iceicle {


    template<class T, class IDX, int ndim>
    inline static transformations::HypercubeTraceOrientTransformation<T, IDX, ndim> hypercube_orient_trans{};

    template<class T, class IDX, int ndim, int Pn>
    inline static transformations::HypercubeTraceTransformation<T, IDX, ndim, Pn> hypercube_trans{};

    template<class T, class IDX, DOMAIN_TYPE domain_left, DOMAIN_TYPE domain_right>
    class SegmentFace final : public Face<T, IDX, 2> {
        public:
        static constexpr int ndim = 2;
        static constexpr int nnode = 2;
        using FaceBase = Face<T, IDX, ndim>;
        using Point = typename FaceBase::Point;
        using FacePoint = typename FaceBase::FacePoint;
        using JacobianType = typename FaceBase::JacobianType;

        std::array<IDX, nnode> _nodes;

        /// @brief construct a segment face 
        /// @param elemL the index of the left element 
        /// @param elemR the index of the right element 
        /// @param node_idxs the indices of the nodes on the face 
        /// @param faceNrL the face number for the left element
        /// @param faceNrR the face number for the right element
        /// @param bctype the boundary condition type 
        /// @param bcflag the integer flag for the boundary condition
        template<std::size_t span_size>
        SegmentFace(
            IDX elemL, IDX elemR,
            std::span<const IDX, span_size> node_idxs,
            int faceNrL, int faceNrR, int orientR,
            BOUNDARY_CONDITIONS bctype,
            int bcflag 
        ) : Face<T, IDX, ndim>(elemL, elemR, faceNrL * FACE_INFO_MOD, 
                faceNrR * FACE_INFO_MOD + orientR, bctype, bcflag),
            _nodes{}
        { std::ranges::copy(node_idxs, _nodes.begin()); }

        constexpr
        auto domain_type() const -> DOMAIN_TYPE override {
            return DOMAIN_TYPE::HYPERCUBE;
        }


        constexpr
        auto geometry_order() const noexcept -> int override {
            return 1;
        }

        void transform(
            const FacePoint &s,
            NodeArray<T, ndim> &coord, 
            Point& result
        ) const override {
            T s_normalized = (1.0 + s[0]) / 2.0;
            T lambda0 = 1.0 - s_normalized;
            T lambda1 = s_normalized;

            result[0] = lambda0 * coord[_nodes[0]][0] + lambda1 * coord[_nodes[1]][0];
            result[1] = lambda0 * coord[_nodes[0]][1] + lambda1 * coord[_nodes[1]][1];
        }

        void transform_xiL(
            const FacePoint &s,
            Point& result
        ) const override {
            if constexpr (domain_left == DOMAIN_TYPE::HYPERCUBE){
                hypercube_trans<T, IDX, ndim, 1>.transform(
                    _nodes.data(),
                    this->face_infoL / FACE_INFO_MOD,
                    s.data(), result);
            } else {
                int face_nr = this->face_nr_l();
                T s_normalized = (1.0 + s[0]) / 2.0;
                switch(face_nr){
                    case 0:
                        result[0] = 0.0;
                        result[1] = 1.0 - s_normalized;
                        break;
                    case 1:
                        result[0] = s_normalized;
                        result[1] = 0;
                        break;
                    case 2:
                        result[0] = 1.0 - s_normalized;
                        result[1] = s_normalized;
                        break;
                }
            }
        }

        void transform_xiR(
            const FacePoint &s,
            Point& result
        ) const override {
            if constexpr (domain_right == DOMAIN_TYPE::HYPERCUBE){
                FacePoint sR;
                hypercube_orient_trans<T, IDX, ndim>.transform(
                    this->face_infoR % FACE_INFO_MOD,
                    s, sR);
                hypercube_trans<T, IDX, ndim, 1>.transform(
                    _nodes.data(),
                    this->face_infoR / FACE_INFO_MOD,
                    sR.data(), result);
            } else {
                int face_nr = this->face_nr_r();
                int orient = this->orientation_r();
                T s_normalized = (1.0 + s[0]) / 2.0;

                if(orient == 0){
                    switch(face_nr){
                        case 0:
                            result[0] = 0.0;
                            result[1] = 1.0 - s_normalized;
                            break;
                        case 1:
                            result[0] = s_normalized;
                            result[1] = 0;
                            break;
                        case 2:
                            result[0] = 1.0 - s_normalized;
                            result[1] = s_normalized;
                            break;
                    }
                } else {
                    switch(face_nr){
                        case 0:
                            result[0] = 0.0;
                            result[1] = s_normalized;
                            break;
                        case 1:
                            result[0] = 1.0 - s_normalized;
                            result[1] = 0.0;
                            break;
                        case 2:
                            result[0] = s_normalized;
                            result[1] = 1.0 - s_normalized;
                            break;
                    }
                }
            }
        }


        JacobianType Jacobian(
            NodeArray<T, ndim> &node_coords,
            const FacePoint &s
        ) const override {
            return hypercube_trans<T, IDX, ndim, 1>.Jacobian(
                node_coords,
                _nodes.data(),
                this->face_infoL / FACE_INFO_MOD,
                s
            );
        }

        int n_nodes() const override { return nnode; }

        const IDX *nodes() const override { return _nodes.data(); }

        auto clone() const -> std::unique_ptr<Face<T, IDX, ndim>> override {
            return std::make_unique<SegmentFace<T, IDX, domain_left, domain_right>>(*this);
        }
    };

    template<typename T, typename IDX, int ndim, int Pn>
    class HypercubeFace final : public Face<T, IDX, ndim> {
        public:
        using FaceBase = Face<T, IDX, ndim>;
        using Point = typename FaceBase::Point;
        using FacePoint = typename FaceBase::FacePoint;
        using JacobianType = typename FaceBase::JacobianType;

        inline static auto orient_trans = hypercube_orient_trans<T, IDX, ndim>;
        inline static auto trans = hypercube_trans<T, IDX, ndim, Pn>;

        /// The global node coordinates
        std::array<IDX, trans.n_nodes> _nodes;

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
            std::span<const IDX> nodes,
            int faceNrL, int faceNrR,
            int orientR,
            BOUNDARY_CONDITIONS bctype,
            int bcflag
        ) : Face<T, IDX, ndim>(elemL, elemR,
            faceNrL * FACE_INFO_MOD,
            faceNrR * FACE_INFO_MOD + orientR,
            bctype, bcflag), _nodes{} 
        { std::ranges::copy(nodes, _nodes.begin()) ;}

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

        auto clone() const -> std::unique_ptr<Face<T, IDX, ndim>> override {
            return std::make_unique<HypercubeFace<T, IDX, ndim, Pn>>(*this);
        }
    };
}
