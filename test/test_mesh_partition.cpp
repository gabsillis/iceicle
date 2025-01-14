#include <iceicle/mesh/mesh.hpp>
#include <iceicle/mesh/mesh_utils.hpp>
#include <iceicle/mesh/mesh_partition.hpp>
#include <fmt/core.h>
#include <iceicle/vtk_writer.hpp>

using namespace NUMTOOL::TENSOR::FIXED_SIZE;
using namespace iceicle;
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    // get mpi information
    int nrank, myrank;
    MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
    MPI_Comm_size(MPI_COMM_WORLD, &nrank);

    AbstractMesh<double, int, 2> mesh(
        Tensor<double, 2>{0.0, 0.0},
        Tensor<double, 2>{1.0, 1.0},
        Tensor<int, 2>{4, 3},
        1);

    auto elsuel = to_elsuel<double, int, 2>(mesh.nelem(), mesh.faces);


    auto [el_part, renumbering] = partition_elements(elsuel, EL_PARTITION_ALGORITHM::METIS);

    std::cout << el_part;

    // NOTE: we will need some sort of renumbering or petsc matrices will be a mess
    // try to get all global indices in contiguous groups of same rank

    AbstractMesh<double, int, 2> pmesh{partition_mesh(mesh)};

    io::write_mesh_vtk(pmesh);

    MPI_Finalize();
}
