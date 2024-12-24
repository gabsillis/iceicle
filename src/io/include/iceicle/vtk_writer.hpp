#pragma once
#include "iceicle/build_config.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include <vtkXMLPUnstructuredGridWriter.h>
#include <vtkUnstructuredGrid.h>
#include <vtkMPICommunicator.h>
#include <vtkMPIController.h>
#include <vtkMPI.h>
#include <vtkLagrangeQuadrilateral.h>
#include <vtkLagrangeHexahedron.h>
#include <vtkLagrangeTriangle.h>
#include <vtkLagrangeTetra.h>
#include <iceicle/mesh/mesh.hpp>
namespace iceicle::io {

    template<class T, class IDX, int ndim>
    auto write_mesh_vtk(const AbstractMesh<T, IDX, ndim>& mesh) {

        int max_pn = build_config::FESPACE_BUILD_GEO_PN + 1;
        std::vector< vtkSmartPointer< vtkCell > >  reference_hypercubes(max_pn);
        std::vector< vtkSmartPointer< vtkCell > >  reference_simplices(max_pn);
        for(int geo_order = 1; geo_order < max_pn; ++geo_order) {
            switch(ndim){
                case 2:
                    {
                        auto quad = vtkSmartPointer<vtkLagrangeQuadrilateral>::New();
                        quad->SetOrder(geo_order, geo_order);
                        quad->Initialize();
                        reference_hypercubes[geo_order] = quad;
                        auto tri = vtkSmartPointer<vtkLagrangeTriangle>::New();
                        int npoin = (geo_order + 1) * (geo_order + 2) / 2;
                        tri->GetPointIds()->SetNumberOfIds(npoin);
                        tri->GetPoints()->SetNumberOfPoints(npoin);
                        tri->Initialize();
                        reference_simplices[geo_order] = tri;
                    }
                    break;
                case 3:
                    {
                        auto hex = vtkSmartPointer<vtkLagrangeHexahedron>::New();
                        hex->SetOrder(geo_order, geo_order, geo_order);
                        hex->Initialize();
                        reference_hypercubes[geo_order] = hex;
                        auto tetr = vtkSmartPointer<vtkLagrangeTetra>::New();
                        int npoin = (geo_order + 1) * (geo_order + 2) * (geo_order + 3) / 6;
                        tetr->GetPointIds()->SetNumberOfIds(npoin);
                        tetr->GetPoints()->SetNumberOfPoints(npoin);
                        tetr->Initialize();
                        reference_simplices[geo_order] = tetr;
                    }
                    break;
            }
        }

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

        IDX ivtk_pt = 0;
        vtkNew<vtkPoints> vtk_coord;
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            switch(mesh.el_transformations[ielem]->domain_type){
                case DOMAIN_TYPE::HYPERCUBE:
                    auto ref_cell = reference_hypercubes[mesh.el_transformations[ielem]->order];
                    double* coords = ref_cell->GetParametricCoords();
                    std::vector<vtkIdType> pts{};
                    for(int ipoin = 0; ipoin < ref_cell->GetNumberOfPoints(); ++ipoin){
                        double* coords_start = coords + 3 * ipoin; // always 3d 
                        MATH::GEOMETRY::Point<T, ndim> refpt{};
                        for(int idim = 0; idim < ndim; ++idim){
                            // shift from [0, 1] to [-1, 1]
                            refpt[idim] = coords_start[idim] * 2.0 - 1.0;
                        }
                        std::vector<MATH::GEOMETRY::Point<T, ndim>> el_coord(
                                mesh.coord_els.rowsize(ielem));
                        std::ranges::copy(mesh.coord_els.rowspan(ielem), el_coord.begin());
                        auto act_pt = mesh.el_transformations[ielem]->transform(
                            std::span{el_coord}, refpt);
                        double vtk_point[3];
                        for(int idim = 0; idim < std::min(3, ndim); ++idim)
                            vtk_point[idim] = act_pt[idim];
                        for(int idim = ndim; idim < 3; ++idim)
                            vtk_point[idim] = 0.0;
                        vtk_coord->InsertPoint(ivtk_pt, vtk_point);
                        pts.push_back(ivtk_pt);
                    }
                    vtk_grid->InsertNextCell(ref_cell->GetCellType(),
                            ref_cell->GetNumberOfPoints(), pts.data());
                break;
            }
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
