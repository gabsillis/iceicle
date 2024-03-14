#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/dat_writer.hpp"
#include "iceicle/disc/heat_eqn.hpp"
#include "iceicle/explicit_utils.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/solvers/element_linear_solve.hpp"
#include "iceicle/explicit_euler.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/disc/l2_error.hpp"
#include "iceicle/ssp_rk3.hpp"
#include <cmath>
#include <fenv.h>

int main(int argc, char *argv[]){

    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;

    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace ICEICLE::UTIL;

    static constexpr int ndim = 1;
    static constexpr int neq = 1;
    
    // ===============================
    // = Command line argument setup =
    // ===============================
    using namespace ICEICLE::UTIL::PROGRAM_ARGS;
    cli_parser cli_args{argc, argv};
    
    cli_args.add_options(
        cli_flag{"help", "print the help text and quit."},
        cli_flag{"enable_fp_except", "enable floating point exceptions (ignoring FE_INEXACT)"},
        cli_option{"mu", "The diffusion coefficient (positive)", parse_type<T>{}}, 
        cli_option{"nelem", "The number of elements", parse_type<IDX>{}},
        cli_option{"order", "The polynomial order of the basis functions", parse_type<int>{}},
        cli_option{"ivis", "the number of timesteps between outputs", parse_type<IDX>{}},
        cli_option{"dt", "the timestep", parse_type<T>{}}, 
        cli_option{"ddgic_mult", "multiplier for ddgic", parse_type<T>{}}, 
        cli_option{"fo", "fourier number", parse_type<T>{}},
        cli_flag{"interior_penalty", "enable interior penalty instead of ddg"}
    );
    if(cli_args["help"]){
        cli_args.print_options(std::cout);
        return 0;
    }
    if(cli_args["enable_fp_except"]){
        feenableexcept(FE_ALL_EXCEPT & ~FE_INEXACT);
    }
    int runtime_order = (cli_args["order"].has_value()) ? cli_args["order"].as<int>() : 1;
    IDX nelem = (cli_args["nelem"].has_value()) ? cli_args["nelem"].as<IDX>() : 8;

    auto mainfunc = [&]<int order>->int{
        MESH::AbstractMesh<T, IDX, ndim> mesh{Tensor<T, 1>{0}, Tensor<T, 1>{2 * M_PI}, Tensor<IDX, 1>{nelem}, 1, 
            Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2>{ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET, ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET},
            Tensor<int, 2>{0, 0}};

        FE::FESpace<T, IDX, ndim> fespace{&mesh, FE::FESPACE_ENUMS::LAGRANGE, FE::FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, order>{}};

        DISC::HeatEquation<T, IDX, ndim> disc{};
        disc.mu = (cli_args["mu"]) ? cli_args["mu"].as<T>() : 1.0;
        disc.sigma_ic = (cli_args["ddgic_mult"]) ? cli_args["ddgic_mult"].as<T>() : 0.0;
        disc.dirichlet_values.push_back(0.0);
        disc.interior_penalty = cli_args["interior_penalty"];

        std::vector<T> u_data(fespace.dg_offsets.calculate_size_requirement(neq));
        FE::dg_layout<T, neq> u_layout{fespace.dg_offsets};
        FE::fespan u{u_data.data(), u_layout};

        // ===========================
        // = initialize the solution =
        // ===========================

        std::function<void(const double*, double *)> ic = [](const double*x, double *out){
            out[0] = std::sin(x[0]);
        };

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

        ICEICLE::SOLVERS::CFLTimestep<T, IDX> cfl{0.1};
        ICEICLE::SOLVERS::FixedTimestep<T, IDX> dt{(cli_args["dt"]) ? cli_args["dt"].as<T>() : 1e-5};
        // fourier number as an option 
        if(cli_args["fo"]) dt.dt = cli_args["fo"].as<T>() * SQUARED(2 * M_PI / (T) nelem);
        ICEICLE::SOLVERS::TfinalTermination<T, IDX> stop_condition{1.0};

        ICEICLE::SOLVERS::RK3SSP solver{fespace, disc, dt, stop_condition};
        solver.ivis = (cli_args["ivis"].has_value()) ? cli_args["ivis"].as<IDX>() : 100;
        ICEICLE::IO::DatWriter<T, IDX, ndim> writer{fespace};
        writer.register_fields(u, "u");
        solver.vis_callback = [&](decltype(solver) &solver){
            T sum = 0.0;
            for(int i = 0; i < solver.res_data.size(); ++i){
                sum += SQUARED(solver.res_data[i]);
            }
            std::cout << std::setprecision(8);
            std::cout << "itime: " << std::setw(6) << solver.itime 
                << " | t: " << std::setw(14) << solver.time
                << " | residual l2: " << std::setw(14) << std::sqrt(sum) 
                << std::endl;

            writer.write_dat(solver.itime, solver.time);

        };

        solver.solve(fespace, disc, u);

        // compute L2 error
        std::function<void(T*, T*)> exactfunc = [mu = disc.mu](T *x, T *out) -> void {
            out[0] = std::exp(-mu) * std::sin(x[0]);
        };
        T l2_error = DISC::l2_error(exactfunc, fespace, u);
        std::cout << "L2 error: " << std::setprecision(9) << l2_error << std::endl;

        AnomalyLog::handle_anomalies();
        return 0;
    };

    return NUMTOOL::TMP::invoke_at_index(NUMTOOL::TMP::make_range_sequence<int, 1, ELEMENT::MAX_DYNAMIC_ORDER>{}, 
            runtime_order, mainfunc);
}

