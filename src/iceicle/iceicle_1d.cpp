#include <iceicle/geometry/segment.hpp>
#include <iceicle/mesh/segment_mesh.hpp>
#include <mpi.h>
#include <iostream>
int main(int argc, char *argv[]) {

    MPI_Init(&argc, &argv);
    using namespace ELEMENT;
    // construct a segment
    Segment<double, int> seg1(0, 1);

    // construct a parallel segment mesh with periodic bcs
    MESH::ParSegmentMesh<double, int> mesh(10, 0.0, 0.1);
    mesh.printElements(std::cout);
    mesh.printFaces(std::cout);
    MPI_Finalize();
}
