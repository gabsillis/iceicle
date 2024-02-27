/**
 * @brief miniapp to solve the heat equation
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include "iceicle/disc/l2_error.hpp"
#include "iceicle/fespace/fespace_lua_interface.hpp"
#include "iceicle/lua_utils.hpp"
#include <iomanip>
#ifdef ICEICLE_USE_PETSC 
#include "iceicle/petsc_newton.hpp"
#endif
#ifdef ICEICLE_USE_MPI
#include "mpi.h"
#endif

#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/lua_utils.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/explicit_euler.hpp>
#include <iceicle/solvers/element_linear_solve.hpp>
#include <iceicle/build_config.hpp>
#include <iceicle/pvd_writer.hpp>
#include <iceicle/mesh/mesh_lua_interface.hpp>
#include <type_traits>
#include <fenv.h>

#include <sol/sol.hpp>

int main(int argc, char *argv[]){
#ifdef ICEICLE_USE_PETSC
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#elifdef ICEICLE_USE_MPI
   /* Initialize MPI */
   MPI_Init(&argc, &argv);
#endif

//    feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);

    // The default file to read as input deck
    // TODO: add command line arg parsing for specifying other file names as input deck
    const char *default_input_deck_filename = "iceicle.lua";

    // Parse the input deck 
    sol::state lua_state;
    lua_state.open_libraries(sol::lib::base);
    lua_state.open_libraries(sol::lib::math);
    lua_state.script_file(default_input_deck_filename);


    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace ICEICLE::UTIL;

    // Get the floating point and index types from 
    // cmake configuration
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;

    // 2 dimensional simulation
    static constexpr int ndim = 2;

    // =========================
    // = create a uniform mesh =
    // =========================

    MESH::AbstractMesh<T, IDX, ndim> mesh = MESH::lua_uniform_mesh<T, IDX, ndim>(lua_state);

    // ===================================
    // = create the finite element space =
    // ===================================

    auto fespace = FE::lua_fespace(&mesh, lua_state);

    // =============================
    // = set up the discretization =
    // =============================

    DISC::HeatEquation<T, IDX, ndim> heat_equation{}; 

    // Get boundary condition values from input deck 
    static constexpr int MAX_BC_ENTRIES = 100; // protect from infinite loops

    sol::optional<sol::table> dirichlet_spec_opt = lua_state["boundary_conditions"]["dirichlet"];
    if(dirichlet_spec_opt){
        sol::table dirichlet_spec = dirichlet_spec_opt.value();

        // Value arguments
        // To make indexing consistently 1-indexed 
        // push a 0 valued boundary condition at index 0
        // Now values start at 1 and callbacks start at -1 with no overlap on index 0
        heat_equation.dirichlet_values.push_back(0); 
        for(int ibc = 1; ibc < MAX_BC_ENTRIES; ++ibc){
            sol::optional<T> lua_value = dirichlet_spec["values"][ibc];
            if(lua_value) heat_equation.dirichlet_values.push_back(lua_value.value());
            else break;
        }

        // Callbacks
        // same index 0 trick
        // actually use the table iterator here
        heat_equation.dirichlet_callbacks.push_back([](const double*, double *out){ out[0] = 0.0; });
        sol::table callback_tbl = dirichlet_spec["callbacks"];
        sol::function testf = callback_tbl[2];
        for(const auto& key_value : callback_tbl){
            int index = key_value.first.as<int>();
            auto dirichlet_func = key_value.second.as<std::function<double(double, double)>>();

            // indexes could be in non-sequential order so manually work with vector capacity
            if(index >= heat_equation.dirichlet_callbacks.size())
                heat_equation.dirichlet_callbacks.resize(2 * heat_equation.dirichlet_callbacks.size());

            // add the callback function at the given index
//            heat_equation.dirichlet_callbacks[index] =
//                [dirichlet_func](const double *xarr, double *out){
//                    out[0] = dirichlet_func(xarr[0], xarr[1]);
//                };
            heat_equation.dirichlet_callbacks[index] =
                [&lua_state, index](const double *xarr, double *out){
                    sol::optional<sol::table> ftbl = lua_state["boundary_conditions"]["dirichlet"]["callbacks"];
                    sol::function f = ftbl.value()[index];
                    out[0] = f(xarr[0], xarr[1]);
                };
        }
    }


    sol::optional<sol::table> neumann_spec_opt = lua_state["boundary_conditions"]["neumann"];
    if(neumann_spec_opt){
        sol::table neumann_spec = neumann_spec_opt.value();

        // Value arguments
        // To make indexing consistently 1-indexed 
        // push a 0 valued boundary condition at index 0
        // Now values start at 1 and callbacks start at -1 with no overlap on index 0
        heat_equation.neumann_values.push_back(0); 
        int ilua_value = 1;
        sol::optional<T> lua_value;
        while((lua_value = neumann_spec["values"][ilua_value++])){
            heat_equation.neumann_values.push_back(lua_value.value());
        }

    }

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

    std::cout << "Initializing Solution..." << std::endl;

    std::function<void(const double*, double *)> ic;

    // first check if the initial condiiton is identified by string
    sol::optional<std::string> ic_string = lua_state["initial_condition"];
    if(ic_string){
        if(eq_icase(ic_string.value(), "zero")) {
            ic = [](const double *, double *out){ out[0] = 0.0; };
        }
    } else {
        // else we take ic as a lua function with
        // 2 inputs (ndim) and one output (neq)
        // and wrap in a lambda
        ic = [&lua_state](const double *xarr, double *out){
            out[0] = lua_state["initial_condition"](xarr[0], xarr[1]);
        };
    }


    auto analytic_sol = [](const double *xarr, double *out) -> void{
        double x = xarr[0];
        double y = xarr[1];

        out[0] = 0.1 * std::sinh(M_PI * x) / std::sinh(M_PI) * std::sin(M_PI * y) + 1.0;
    };

    auto dirichlet_func = [](const double *xarr, double *out) ->void{
        double x = xarr[0];
        double y = xarr[1];

        out[0] = 1 + 0.1 * std::sin(M_PI * y);
    };
    
    // Manufactured solution BC 
    heat_equation.dirichlet_callbacks.resize(2);
    heat_equation.dirichlet_callbacks[1] = dirichlet_func;


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

    bool use_explicit = false;

    if(use_explicit){
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
    explicit_euler.vis_callback = [&](ExplicitEuler<T, IDX> &solver){
        T sum = 0.0;
        for(int i = 0; i < solver.res_data.size(); ++i){
            sum += SQUARED(solver.res_data[i]);
        }
        std::cout << std::setprecision(8);
        std::cout << "itime: " << std::setw(6) << solver.itime 
            << " | t: " << std::setw(14) << solver.time
            << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
            << std::endl;

        pvd_writer.write_vtu(solver.itime, solver.time);

    };
    explicit_euler.solve(fespace, heat_equation, u);

    } else {
#ifdef ICEICLE_USE_PETSC
    // ==============================
    // = Solve with Newton's Method =
    // ==============================
    using namespace ICEICLE::SOLVERS;
    ConvergenceCriteria<T, IDX> conv_criteria{
        .tau_abs = std::numeric_limits<T>::epsilon(),
        .tau_rel = std::numeric_limits<T>::epsilon(),
        .kmax = 5
    };
    PetscNewton solver{fespace, heat_equation, conv_criteria};
    solver.idiag = -1;
    solver.ivis = 1;
    ICEICLE::IO::PVDWriter<T, IDX, ndim> pvd_writer;
    pvd_writer.register_fespace(fespace);
    pvd_writer.register_fields(u, "u");
    pvd_writer.write_vtu(0, 0.0); // print the initial solution
    solver.vis_callback = [&](decltype(solver) &solver, IDX k, Vec res_data, Vec du_data){
        T res_norm;
        PetscCallAbort(solver.comm, VecNorm(res_data, NORM_2, &res_norm));
        std::cout << std::setprecision(8);
        std::cout << "itime: " << std::setw(6) << k
            << " | residual l2: " << std::setw(14) << res_norm
            << std::endl;
        // offset by initial solution iteration
        pvd_writer.write_vtu(k + 1, (T) k + 1);
    };
    solver.solve(u);

    // ========================
    // = Compute the L2 Error =
    // =   (if applicable)    =
    // ========================
    sol::optional<sol::function> exact_sol = lua_state["exact_sol"];
    if(exact_sol){
        std::function<void(T*, T*)> exactfunc = 
            [&lua_state](T *x, T *out) -> void {
                sol::function fexact = lua_state["exact_sol"];
                out[0] = fexact(x[0], x[1]);
            };
        T l2_error = DISC::l2_error(exactfunc, fespace, u);
        std::cout << "L2 error: " << std::setprecision(9) << l2_error << std::endl;
    }

    //cleanup
    PetscFinalize();
#elifdef ICEICLE_USE_MPI
    // cleanup
    MPI_Finalize();
#endif
    }
    return 0;
}


