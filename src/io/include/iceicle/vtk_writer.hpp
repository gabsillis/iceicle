#pragma once
#include "iceicle/iceicle_mpi_utils.hpp"
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkMPI.h>
#include <iceicle/mesh/mesh.hpp>
namespace iceicle::io {

    template<class T, class IDX, int ndim>
    auto write_mesh_vtk(const AbstractMesh<T, IDX, ndim>& mesh) {

        // setup unstructured grid
        auto vtk_grid = vtkSmartPointer<vtkUnstructuredGrid>::New();
        vtkSmartPointer<vtkXMLPUnstructuredGridWriter> writer =
            vtkSmartPointer<vtkXMLPUnstructuredGridWriter>::New();
#ifdef ICEICLE_USE_MPI
        // set VTK to use our MPI communicator
        vtkSmartPointer<vtkMPICommunicator> vtk_comm = vtkSmartPointer<vtkMPICommunicator>::New();
        MPI_Comm mpi_comm = MPI_COMM_WORLD;
        vtkMPICommunicatorOpaqueComm vtk_opaque_comm(&mpi_comm);
        vtk_comm->InitializeExternal(&vtk_opaque_comm);
        vtkSmartPointer<vtkMPIController> vtk_mpi_ctrl = vtkSmartPointer<vtkMPIController>::New();
        vtk_mpi_ctrl->SetCommunicator(vtk_comm);
        writer->SetController(vtk_mpi_ctrl);
#endif

        vtk_grid->Allocate(mesh.nelem());
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            std::vector<vtkIdType> pts{ 
                mesh.conn_el[ielem, 0],
                mesh.conn_el[ielem, 2],
                mesh.conn_el[ielem, 3],
                mesh.conn_el[ielem, 1]
            };
            vtk_grid->InsertNextCell(VTK_QUAD, 4, pts.data());
        }
        vtkNew<vtkPoints> vtk_coord;
        for(int inode = 0; inode < mesh.n_nodes(); ++inode){
            double point[3];
            for(int idim = 0; idim < std::min(3, ndim); ++idim)
                point[idim] = mesh.coord[inode][idim];
            for(int idim = ndim; idim < 3; ++idim)
                point[idim] = 0;
            vtk_coord->InsertPoint(inode, point);
        }
        vtk_grid->SetPoints(vtk_coord);

        writer->SetNumberOfPieces(mpi::mpi_world_size());
        writer->SetStartPiece(mpi::mpi_world_rank());
        writer->SetEndPiece(mpi::mpi_world_rank());

        writer->SetInputData(vtk_grid);
        writer->SetFileName("test.pvtu");
        writer->SetDataModeToAscii();
        writer->Write();
    }
}
