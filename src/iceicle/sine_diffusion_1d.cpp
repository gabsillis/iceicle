#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/dat_writer.hpp"
#include "iceicle/disc/burgers.hpp"
#include "iceicle/disc/conservation_law.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/explicit_utils.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/element_linear_solve.hpp"
#include "iceicle/explicit_euler.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/disc/l2_error.hpp"
#include "iceicle/tvd_rk3.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include <cmath>
#include <fenv.h>
#include <type_traits>

int main(int argc, char *argv[]){

    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace iceicle;
    using namespace iceicle::solvers;
    using namespace iceicle::util;
    using namespace iceicle::util::program_args;

    using T = build_config::T;
    using IDX = build_config::IDX;


    static constexpr int ndim = 1;
    static constexpr int neq = 1;
    
    // ===============================
    // = Command line argument setup =
    // ===============================
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
        AbstractMesh<T, IDX, ndim> mesh{Tensor<T, 1>{0}, Tensor<T, 1>{2 * M_PI}, Tensor<IDX, 1>{nelem}, 1, 
            Tensor<BOUNDARY_CONDITIONS, 2>{BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET},
            Tensor<int, 2>{0, 0}};

        FESpace<T, IDX, ndim> fespace{&mesh, FESPACE_ENUMS::LEGENDRE, FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, order>{}};

        BurgersCoefficients<T, ndim> burgers_coeffs{};
        BurgersFlux physical_flux{burgers_coeffs};
        BurgersUpwind convective_flux{burgers_coeffs};
        BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
        T mu = (cli_args["mu"]) ? cli_args["mu"].as<T>() : 1.0;
        burgers_coeffs.mu = mu;
        ConservationLawDDG disc{std::move(physical_flux),
                              std::move(convective_flux),
                              std::move(diffusive_flux)};
        disc.field_names = std::vector<std::string>{"u"};
        disc.sigma_ic = (cli_args["ddgic_mult"]) ? cli_args["ddgic_mult"].as<T>() : 0.0;
        disc.dirichlet_callbacks.push_back( 
            [](const T *x, T *out){
                out[0] = 0.0;
        });
        disc.interior_penalty = cli_args["interior_penalty"];

        fe_layout_right u_layout{fespace, std::integral_constant<std::size_t, neq>{}, std::true_type{}};
        std::vector<T> u_data(u_layout.size());
        fespan u{u_data.data(), u_layout};

        // ===========================
        // = initialize the solution =
        // ===========================

        std::function<void(const double*, double *)> ic = [](const double*x, double *out){
            out[0] = std::sin(x[0]);
        };

        Projection<T, IDX, ndim, neq> projection{ic};
        // TODO: extract into LinearFormSolver
        std::vector<T> u_local_data(fespace.dofs.max_el_size_reqirement(neq));
        std::vector<T> res_local_data(fespace.dofs.max_el_size_reqirement(neq));
        std::for_each(fespace.elements.begin(), fespace.elements.end(), 
            [&](const FiniteElement<T, IDX, ndim> &el){
                // form the element local views
                // TODO: maybe instead of scatter from local view 
                // we can directly create the view on the subset of u 
                // for CG this might require a different compact Layout 
                dofspan u_local{u_local_data.data(), u.create_element_layout(el.elidx)};
                u_local = 0;

                dofspan res_local{res_local_data.data(), u.create_element_layout(el.elidx)};
                res_local = 0;

                // project
                projection.domain_integral(el, res_local);

                // solve 
                ElementLinearSolver<T, IDX, ndim, neq> solver{el};
                solver.solve(u_local, res_local);

                // scatter to global array 
                // (note we use 0 as multiplier for current values in global array)
                scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);
            }
        );

        CFLTimestep<T, IDX> cfl{0.1};
        FixedTimestep<T, IDX> dt{(cli_args["dt"]) ? cli_args["dt"].as<T>() : 1e-5};
        // fourier number as an option 
        if(cli_args["fo"]) dt.dt = cli_args["fo"].as<T>() * SQUARED(2 * M_PI / (T) nelem);
        TfinalTermination<T, IDX> stop_condition{1.0};

        RK3TVD solver{fespace, disc, dt, stop_condition};
        solver.ivis = (cli_args["ivis"].has_value()) ? cli_args["ivis"].as<IDX>() : 100;
        io::DatWriter<T, IDX, ndim> writer{fespace};
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
        std::function<void(T*, T*)> exactfunc = [mu](T *x, T *out) -> void {
            out[0] = std::exp(-mu) * std::sin(x[0]);
        };
        T error = l2_error(exactfunc, fespace, u);
        std::cout << "L2 error: " << std::setprecision(9) << error << std::endl;

        // print the exact solution in dat format
        std::ofstream outexact{"iceicle_data/exact.dat"};
        int npoin = 1000;
        constexpr int field_width = 18;
        constexpr int precision = 10;
        for(int ipoin = 0; ipoin < npoin; ++ipoin){
            T dx = 2 * M_PI / (npoin - 1);
            T x = ipoin * dx;
            outexact << fmt::format("{:>{}.{}e}", x, field_width, precision);
            T f;
            exactfunc(&x, &f);
            outexact << " " << fmt::format("{:>{}.{}e}", f, field_width, precision) << std::endl;
        }
        AnomalyLog::handle_anomalies();
        return 0;
    };

    return NUMTOOL::TMP::invoke_at_index(NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{}, 
            runtime_order, mainfunc);
}

