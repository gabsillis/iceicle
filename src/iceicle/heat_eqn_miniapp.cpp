/**
 * @brief miniapp to solve the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/element/reference_element.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/explicit_euler.hpp>
#include <iceicle/build_config.hpp>
#include <type_traits>

int main(int argc, char *argv[]){
    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // Get the floating point and index types from 
    // cmake configuration
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;

    // 2 dimensional simulation
    static constexpr int ndim = 2;

    // =========================
    // = create a uniform mesh =
    // =========================

    IDX nx=2, ny=2;
    const IDX nelem_arr[ndim] = {nx, ny};
    // bottom left corner
    T xmin[ndim] = {-1.0, -1.0};
    // top right corner
    T xmax[ndim] = {1.0, 1.0};
    // boundary conditions
    Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes = {{
        ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET, // left side 
        ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET, // right side 
        ELEMENT::BOUNDARY_CONDITIONS::NEUMANN,   // bottom side 
        ELEMENT::BOUNDARY_CONDITIONS::NEUMANN    // top side
    }};
    int bcflags[2 * ndim] = {
        0, // left side 
        1, // right side
        0, // bottom side 
        0  // top side
    };
    int geometry_order = 1;

    MESH::AbstractMesh<T, IDX, ndim> mesh{xmin, xmax, 
        nelem_arr, geometry_order, bctypes, bcflags};

    // ===================================
    // = create the finite element space =
    // ===================================

    static constexpr int basis_order = 1;

    FE::FESpace<T, IDX, ndim> fespace{
        &mesh, 
        FE::FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE, 
        FE::FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE, 
        std::integral_constant<int, basis_order>{}
    };

    return 0;
}


