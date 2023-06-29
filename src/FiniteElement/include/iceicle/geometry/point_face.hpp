/**
 * @file point_face.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief A 1D face type
 *
 */
#pragma once
#include <iceicle/geometry/face.hpp>

namespace ELEMENT {
    
    template<typename T, typename IDX>
    class PointFace final : public Face<T, IDX, 1> {
        static constexpr int ndim = 1;
        using Point = GEOMETRY::Point<T, ndim>;

        // === Index data ===
        /// the node corresponding to this face
        IDX node;

        // === Precomputation ===
        /// the normal vector
        T normal;
        /// the face area
        T area = 1.0;
        /// the centroid in the physical domain
        Point centroid;

        public:
        // === Constructors ===
        PointFace(
            IDX elemL,
            IDX elemR,
            int faceNrL,
            int faceNrR,
            IDX nodeIdx,
            bool positiveNormal,
            BOUNDARY_CONDITIONS bctype = INTERIOR,
            int bcflag = 0
        ) : Face<T, IDX, 1>(elemL, elemR, faceNrL, faceNrR, 0, 0, bctype, bcflag),
            node(node)
        {
           if(positiveNormal) normal = 1.0;
           else normal = -1.0;
        }

        // === Overriden Face methods ===
        void updateGeometry(std::vector< Point > &nodeCoords) override {
            centroid = nodeCoords[node];
        }

        inline void getNormal(
            std::vector< Point > &nodeCoords,
            const GEOMETRY::Point<T, ndim - 1> &s,
            T *n
        ) override {
            n[0] = normal;
        }

        inline T getArea() override { return area; }

        inline void getUnitNormal(std::vector< Point > &nodeCoords,
            const GEOMETRY::Point<T, ndim - 1> &s,
            T *n
        ) override {
            n[0] = normal;
        }

        void convertRefToAct(
            std::vector< Point > &nodeCoords,
            const GEOMETRY::Point<T, ndim - 1> &s,
            T *result
        ) override {
            result[0] = nodeCoords[node][0];
        }

        inline const Point &getCentroid() override { return centroid; }

        T rootRiemannMetric(
            std::vector<GEOMETRY::Point<T, ndim> > &nodeCoords,
            const GEOMETRY::Point<T, ndim - 1> &s
        ) override {
            return 1.0;
        }

        int n_nodes() override { return 1; }

        IDX *nodes() override { return &node; }
    };
}
