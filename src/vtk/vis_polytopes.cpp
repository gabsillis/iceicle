/*=========================================================================

  Copyright (c) Ken Martin, Will Schroeder, Bill Lorensen
  All rights reserved.
  See Copyright.txt or http://www.kitware.com/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

// First include the required header files for the VTK classes we are using.
#include <algorithm>
#include <vtkActor.h>
#include <vtkCamera.h>
#include <vtkCubeAxesActor.h>
#include <vtkConeSource.h>
#include <vtkNamedColors.h>
#include <vtkNew.h>
#include <vtkPoints.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkProperty.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkRenderer.h>
#include <vtkTextProperty.h>
#include <vtkVertex.h>

#include <iceicle/transformations/polytope_transformations.hpp>

using namespace iceicle;
using namespace polytope;

//
// Next we create an instance of vtkNamedColors and we will use
// this to select colors for the object and background.
//
vtkNew<vtkNamedColors> colors;

/// @brief factory method for the actor that represents the vertices of a toplology
/// in the reference domain
template<geo_code auto t>
auto tcode_vertices_vtk() -> vtkNew<vtkActor> {
  vtkNew<vtkPoints> points;
  auto vertices = gen_vert<t>();
  vtkNew<vtkCellArray> vert_vtk_arr;
  std::vector<vtkIdType> point_ids;
  for(int ipoin = 0; ipoin < vertices.size(); ++ipoin){
    std::array<double, 3> pt_3d_coord;
    std::ranges::fill(pt_3d_coord, 0);
    for(int idim = 0; idim < std::min((std::size_t) 3, vertices[ipoin].size()); ++idim){
      if(vertices[ipoin][idim] == 0){
        pt_3d_coord[idim] = 0.0;
      } else {
        pt_3d_coord[idim] = 1.0;
      }
    }
    point_ids.push_back(points->InsertNextPoint(pt_3d_coord.data()));
    vert_vtk_arr->InsertNextCell(1, &point_ids[ipoin]);
  }

  vtkNew<vtkPolyData> pts_polydata;
  pts_polydata->SetPoints(points);
  pts_polydata->SetVerts(vert_vtk_arr);
  vtkNew<vtkPolyDataMapper> pts_mapper;
  pts_mapper->SetInputData(pts_polydata);
  vtkNew<vtkActor> actor;
  actor->SetMapper(pts_mapper);
  actor->GetProperty()->SetColor(colors->GetColor3d("Tomato").GetData());
  actor->GetProperty()->SetPointSize(20);
  return actor;
}

int main(int, char*[])
{

  vtkColor3d backgroundColor = colors->GetColor3d("DarkSlateGray");
  vtkColor3d axis1Color = colors->GetColor3d("Salmon");
  vtkColor3d axis2Color = colors->GetColor3d("PaleGreen");
  vtkColor3d axis3Color = colors->GetColor3d("LightSkyBlue");

  static constexpr tcode<3> t{"101"};
  //
  // Create the Renderer and assign actors to it. A renderer is like a
  // viewport. It is part or all of a window on the screen and it is
  // responsible for drawing the actors it has.  We also set the background
  // color here.
  //
  vtkNew<vtkRenderer> renderer;
  renderer->SetBackground(colors->GetColor3d("MidnightBlue").GetData());


  vtkNew<vtkActor> shape_vert_actor = tcode_vertices_vtk<t>();
  renderer->AddActor(shape_vert_actor);

  // Finally we create the render window which will show up on the screen.
  // We put our renderer into the render window using AddRenderer. We also
  // set the size to be 300 pixels by 300.
  //
  vtkNew<vtkRenderWindow> renWin;
  renWin->AddRenderer(renderer);
  renWin->SetSize(1920, 1080);
  renWin->SetWindowName("Polytope Visualization");

  vtkNew<vtkCubeAxesActor> cubeAxesActor;
  cubeAxesActor->SetUseTextActor3D(1);
  cubeAxesActor->SetBounds(shape_vert_actor->GetBounds());
  cubeAxesActor->SetCamera(renderer->GetActiveCamera());
  cubeAxesActor->GetTitleTextProperty(0)->SetColor(axis1Color.GetData());
  cubeAxesActor->GetTitleTextProperty(0)->SetFontSize(48);
  cubeAxesActor->GetLabelTextProperty(0)->SetColor(axis1Color.GetData());

  cubeAxesActor->GetTitleTextProperty(1)->SetColor(axis2Color.GetData());
  cubeAxesActor->GetLabelTextProperty(1)->SetColor(axis2Color.GetData());

  cubeAxesActor->GetTitleTextProperty(2)->SetColor(axis3Color.GetData());
  cubeAxesActor->GetLabelTextProperty(2)->SetColor(axis3Color.GetData());

  cubeAxesActor->DrawXGridlinesOn();
  cubeAxesActor->DrawYGridlinesOn();
  cubeAxesActor->DrawZGridlinesOn();
#if VTK_MAJOR_VERSION == 6
  cubeAxesActor->SetGridLineLocation(VTK_GRID_LINES_FURTHEST);
#endif
#if VTK_MAJOR_VERSION > 6
  cubeAxesActor->SetGridLineLocation(cubeAxesActor->VTK_GRID_LINES_FURTHEST);
#endif
  cubeAxesActor->XAxisMinorTickVisibilityOff();
  cubeAxesActor->YAxisMinorTickVisibilityOff();
  cubeAxesActor->ZAxisMinorTickVisibilityOff();
  renderer->AddActor(cubeAxesActor);

  vtkNew<vtkRenderWindowInteractor> renderWindowInteractor;
  renderWindowInteractor->SetRenderWindow(renWin);

  vtkNew<vtkInteractorStyleTrackballCamera> style;
  renderWindowInteractor->SetInteractorStyle(style);

  renWin->Render();
  renderer->GetActiveCamera()->Zoom(0.3);
  renderWindowInteractor->Start();

  return EXIT_SUCCESS;
}
