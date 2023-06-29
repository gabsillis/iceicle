/**
 * @file segment_mesh.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief 1D mesh consisting of Segment elements
 * @date 2023-06-27
 */

#pragma once
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/geometry/segment.hpp>
#include <iceicle/geometry/face.hpp>

namespace MESH {
    
    template<typename T, typename IDX>
    class SegmentMesh : public AbstractMesh<T, IDX, 1> {
        private:
        static constexpr int ndim = 1;
        using BMesh = AbstractMesh<T, IDX, ndim>;

        public:
        /**
         * @brief construct a Uniformly spaced segment mesh
         * 
         * @param nelem the number of elements
         * @param xstart the x coordinate of the left side of the first element
         * @param dx the cell size
         * @param bcleft the left boundary condition, periodic by default
         * @param bcright the right boundary condition, periodic by default
         * @param bcflagL the left boundary condition flag leave default for periodic
         * as this gets automatically calculated
         * @param bcflagR the right boundary condition flag see bcflagL
         */
        SegmentMesh(
            IDX nelem,
            T xstart,
            T dx,
            ELEMENT::BOUNDARY_CONDITIONS bcleft = ELEMENT::PERIODIC,
            ELEMENT::BOUNDARY_CONDITIONS bcright = ELEMENT::PERIODIC,
            int bcflagL = 0,
            int bcflagR = 0
        );
    };

    template<typename T, typename IDX>
    class ParSegmentMesh : public AbstractMesh<T, IDX, 1> {
        private:
        static constexpr int ndim = 1;
        using BMesh = AbstractMesh<T, IDX, ndim>;

        public:
        /**
         * @brief construct a Uniformly spaced parallel segment mesh
         * 
         * @param nelem the number of elements
         * @param xstart the x coordinate of the left side of the first element
         * @param dx the cell size
         * @param bcleft the left boundary condition, periodic by default
         * @param bcright the right boundary condition, periodic by default
         * @param bcflagL the left boundary condition flag leave default for periodic
         * as this gets automatically calculated
         * @param bcflagR the right boundary condition flag see bcflagL
         */
        ParSegmentMesh(
            IDX nelem,
            T xstart,
            T dx,
            ELEMENT::BOUNDARY_CONDITIONS bcleft = ELEMENT::PERIODIC,
            ELEMENT::BOUNDARY_CONDITIONS bcright = ELEMENT::PERIODIC,
            int bcflagL = 0,
            int bcflagR = 0
        );
    };
}
