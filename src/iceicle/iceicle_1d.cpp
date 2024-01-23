#include <iceicle/geometry/segment.hpp>
#include <iceicle/mesh/segment_mesh.hpp>
#ifdef ICEICLE_USE_MPI
#include <mpi.h>
#endif
#include <iostream>
int main(int argc, char *argv[]) {

#ifdef ICEICLE_USE_MPI
    MPI_Init(&argc, &argv);
    using namespace ELEMENT;
    // construct a segment
    Segment<double, int> seg1(0, 1);

    // construct a parallel segment mesh with periodic bcs
    MESH::ParSegmentMesh<double, int> mesh(10, 0.0, 0.1);
    int nproc, myid;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &myid);
    for(int i = 0; i < nproc; ++i){
        MPI_Barrier(MPI_COMM_WORLD);
        if(i == myid){
            std::cout << "Processor " << myid << ":\n";
            mesh.printElements(std::cout);
            mesh.printFaces(std::cout);
        }
    }
    MPI_Finalize();
#endif
}
