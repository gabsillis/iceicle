/**
 * @brief miniapp to solve the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/explicit_euler.hpp>
#include <iceicle/solvers/element_linear_solve.hpp>
#include <iceicle/build_config.hpp>
#include <iceicle/pvd_writer.hpp>
#include <type_traits>
#include <fenv.h>

int main(int argc, char *argv[]){
    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

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

    IDX nx=10, ny=10;
    const IDX nelem_arr[ndim] = {nx, ny};
    // bottom left corner
    T xmin[ndim] = {-1.0, -1.0};
    // top right corner
    T xmax[ndim] = {1.0, 1.0};
    // boundary conditions
    Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes = {{
        ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET, // left side 
        ELEMENT::BOUNDARY_CONDITIONS::NEUMANN,   // bottom side 
        ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET, // right side 
        ELEMENT::BOUNDARY_CONDITIONS::NEUMANN    // top side
    }};
    int bcflags[2 * ndim] = {
        0, // left side 
        0, // bottom side 
        1, // right side
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

    // =============================
    // = set up the discretization =
    // =============================

    DISC::HeatEquation<T, IDX, ndim> heat_equation{}; 
    // Dirichlet BC: Left side = 0
    heat_equation.dirichlet_values.push_back(1.0); 
    // Dirichlet BC: Rigght side = 1
    heat_equation.dirichlet_values.push_back(2.0); 
    // Neumann BC: Top and bottom have 0 normal gradient 
    heat_equation.neumann_values.push_back(0.0);

    // ============================
    // = set up a solution vector =
    // ============================
    constexpr int neq = decltype(heat_equation)::nv_comp;
    std::vector<T> u_data(fespace.dg_offsets.calculate_size_requirement(neq));
    FE::dg_layout<T, neq> u_layout{fespace.dg_offsets};
    FE::fespan u{u_data.data(), u_layout};

    // ===========================
    // = initialize the solution =
    // ===========================
    auto ic = [](const double *xarr, double *out) -> void{
        double x = xarr[0];
        double y = xarr[1];

        //out[0] = x;
        out[0] = std::sin(x) * std::cos(y);
    };
    
    // Manufactured solution BC 
    heat_equation.dirichlet_callbacks.resize(2);
    heat_equation.dirichlet_callbacks[1] = ic;


    DISC::Projection<T, IDX, ndim, neq> projection{ic};
    // TODO: extract into LinearFormSolver
    std::vector<T> u_local_data(fespace.dg_offsets.max_el_size_reqirement(neq));
    std::vector<T> res_local_data(fespace.dg_offsets.max_el_size_reqirement(neq));
    std::for_each(fespace.elements.begin(), fespace.elements.end(), 
        [&](const ELEMENT::FiniteElement<T, IDX, ndim> &el){
            // form the element local views
            // TODO: maybe instead of scatter from local view 
            // we can directly create the view on the subset of u 
            // for CG this might require a different compact Layout 
            FE::elspan u_local{u_local_data.data(), u.create_element_layout(el.elidx)};
            u_local = 0;

            FE::elspan res_local{res_local_data.data(), u.create_element_layout(el.elidx)};
            res_local = 0;

            // project
            projection.domainIntegral(el, fespace.meshptr->nodes, res_local);

            // solve 
            SOLVERS::ElementLinearSolver<T, IDX, ndim, neq> solver{el, fespace.meshptr->nodes};
            solver.solve(u_local, res_local);

            // scatter to global array 
            // (note we use 0 as multiplier for current values in global array)
            FE::scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);
        }
    );

    // =============================
    // = Solve with Explicit Euler =
    // =============================
    using namespace ICEICLE::SOLVERS;
    ExplicitEuler explicit_euler{fespace, heat_equation};
    explicit_euler.ivis = 10;
    explicit_euler.tfinal = 2000;
    ICEICLE::IO::PVDWriter<T, IDX, ndim> pvd_writer;
    pvd_writer.register_fespace(fespace);
    pvd_writer.register_fields(u, "u");
    explicit_euler.vis_callback = [&](ExplicitEuler<T, IDX> &disc){
        T sum = 0.0;
        for(int i = 0; i < disc.res_data.size(); ++i){
            sum += SQUARED(disc.res_data[i]);
        }
        std::cout << std::setprecision(8);
        std::cout << "itime: " << std::setw(6) << disc.itime 
            << " | t: " << std::setw(14) << disc.time
            << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
            << std::endl;

        pvd_writer.write_vtu(disc.itime, disc.time);
    };
    explicit_euler.solve(fespace, heat_equation, u);
    return 0;
}


