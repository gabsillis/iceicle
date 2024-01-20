/**
 * @file segment_mesh.cpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Segment Mesh Implementation
 */
#ifdef ICEICLE_USE_MPI
#include <mpi.h>
#endif
#include <iceicle/mesh/segment_mesh.hpp>
#include <iceicle/geometry/point_face.hpp>
#include <iceicle/build_config.hpp>

namespace MESH {
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;

    template<>
    SegmentMesh<T, IDX>::SegmentMesh(
        IDX nelem,
        T xstart,
        T dx,
        ELEMENT::BOUNDARY_CONDITIONS bcleft,
        ELEMENT::BOUNDARY_CONDITIONS bcright,
        int bcflagL,
        int bcflagR
    ) : AbstractMesh<T, IDX, 1>(nelem + 1) {
        // namespaces used and type aliases
        using namespace ELEMENT;
        using Point = MATH::GEOMETRY::Point<T, 1>;
        static constexpr int ndim = 1;

        // Generate the nodes and elements
        nodes[0][0] = xstart;
        T x_i = xstart + dx;
        for(IDX i = 0; i < nelem; ++i) {
            elements.push_back(new Segment<T, IDX>(i, i + 1));
            nodes[i][0] = x_i;
            x_i += dx;
        }

        // Generate the interior faces
        for(IDX ifac = 0; ifac < nelem - 1; ++ifac){
            int faceNrL = 1; // this face is to the right of elemL
            int faceNrR = 0;
            IDX elemL = ifac;
            IDX elemR = ifac + 1;
            IDX facenode = ifac;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, 
                faceNrL, faceNrR,
                facenode, true
            ));
        }
        BMesh::interiorFaceStart = 0;
        BMesh::interiorFaceEnd = faces.size();

        // Generate boundary faces
        // if periodic, make a single periodic face
        if(bcleft == bcright && bcleft == PERIODIC){
            int faceNrL = 1;
            int faceNrR = 0;
            IDX elemL = nelem - 1;
            IDX elemR = 0;
            IDX facenode = 0; // note this could be considered first or last node
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR,
                faceNrL, faceNrR,
                facenode, true, ELEMENT::PERIODIC
            ));
        } else {
            // left boundary face
            IDX elemL = 0; // real element is left
            IDX elemR = -1; // ghost right element
            int faceNrL = 0; // this face is to the left of elemL
            int faceNrR = 1;
            IDX facenode = 0;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, faceNrL, faceNrR,
                facenode, false, bcleft, bcflagR
            ));
            // right boundary face
            elemL = nelem - 1; // real element is left
            elemR = -1; // ghost right element
            faceNrL = 1; // this face is to the right of elemL
            faceNrR = 1;
            facenode = nelem;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, faceNrL, faceNrR,
                facenode, true, bcright, bcflagR
            ));
        }
        BMesh::bdyFaceStart = interiorFaceEnd;
        BMesh::bdyFaceEnd = faces.size();
    }

#ifdef ICEICLE_USE_MPI
    template<>
    ParSegmentMesh<T, IDX>::ParSegmentMesh(
        IDX nelem,
        T xstart,
        T dx,
        ELEMENT::BOUNDARY_CONDITIONS bcleft,
        ELEMENT::BOUNDARY_CONDITIONS bcright,
        int bcflagL,
        int bcflagR
    ) {
        int myid, nproc;
        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        if(myid == 0){
            bcright = ELEMENT::PARALLEL_COM;
            bcflagR = 1;
        } else if (myid == nproc-1){
            bcleft = ELEMENT::PARALLEL_COM;
            bcflagL = nproc - 2;
        } else {
            bcleft = ELEMENT::PARALLEL_COM;
            bcflagL = myid - 1;
            bcright = ELEMENT::PARALLEL_COM;
            bcflagR = myid + 1;
        }

        int nelem_split = nelem / nproc;
        // pick up remainder in last processor
        nelem = (myid == nproc - 1) ? nelem - (nproc - 1) * nelem_split : nelem_split;
        xstart = myid * dx * nelem_split;

        // namespaces used and type aliases
        using namespace ELEMENT;
        using Point = MATH::GEOMETRY::Point<T, 1>;
        static constexpr int ndim = 1;

        // Generate the nodes and elements
        nodes[0][0] = xstart;
        T x_i = xstart + dx;
        for(IDX i = 0; i < nelem; ++i) {
            elements.push_back(new Segment<T, IDX>(i, i + 1));
            nodes[i][0] = x_i;
            x_i += dx;
        }

        // Generate the interior faces
        for(IDX ifac = 0; ifac < nelem - 1; ++ifac){
            int faceNrL = 1; // this face is to the right of elemL
            int faceNrR = 0;
            IDX elemL = ifac;
            IDX elemR = ifac + 1;
            IDX facenode = ifac;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, 
                faceNrL, faceNrR,
                facenode, true
            ));
        }
        BMesh::interiorFaceStart = 0;
        BMesh::interiorFaceEnd = faces.size();

        // Generate boundary faces
        // if periodic, make a single periodic face
        if(bcleft == bcright && bcleft == PERIODIC){
            int faceNrL = 1;
            int faceNrR = 0;
            IDX elemL = nelem - 1;
            IDX elemR = 0;
            IDX facenode = 0; // note this could be considered first or last node
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR,
                faceNrL, faceNrR,
                facenode, true, ELEMENT::PERIODIC
            ));
        } else {
            // left boundary face
            IDX elemL = 0; // real element is left
            IDX elemR = -1; // ghost right element
            int faceNrL = 0; // this face is to the left of elemL
            int faceNrR = 1;
            IDX facenode = 0;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, faceNrL, faceNrR,
                facenode, false, bcleft, bcflagR
            ));
            // right boundary face
            elemL = nelem - 1; // real element is left
            elemR = -1; // ghost right element
            faceNrL = 1; // this face is to the right of elemL
            faceNrR = 1;
            facenode = nelem;
            faces.push_back(new PointFace<T, IDX>(
                elemL, elemR, faceNrL, faceNrR,
                facenode, true, bcright, bcflagR
            ));
        }
        BMesh::bdyFaceStart = interiorFaceEnd;
        BMesh::bdyFaceEnd = faces.size();

    }
#endif
}
