#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/dat_writer.hpp"
#include "iceicle/disc/burgers.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/disc/projection.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/element_linear_solve.hpp"
#include "iceicle/explicit_euler.hpp"
#include "iceicle/program_args.hpp"
#include "iceicle/disc/l2_error.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/disc/conservation_law.hpp>
#include <cmath>
#include <fenv.h>
#include <petscsys.h>
#include <type_traits>

#ifdef ICEICLE_USE_PETSC 
#include "iceicle/petsc_newton.hpp"
#endif
int main(int argc, char *argv[]){
#ifdef ICEICLE_USE_PETSC
    PetscInitialize(&argc, &argv, nullptr, nullptr);
#endif

    // using declarations
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    using namespace iceicle;
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
        cli_option{"a-adv", "The linear advection coefficient", parse_type<T>{}}, 
        cli_option{"b-adv", "The nonlinear advection coefficient", parse_type<T>{}}, 
//        cli_option{"nelem", "The number of elements", parse_type<IDX>{}},
        cli_option{"order", "The polynomial order of the basis functions", parse_type<int>{}},
        cli_option{"ivis", "the number of timesteps between outputs", parse_type<IDX>{}},
        cli_option{"ddgic_mult", "multiplier for ddgic", parse_type<T>{}}, 
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
    IDX nelem = 2;

    auto mainfunc = [&]<int order>->int{

        // uniform mesh from 0 to 1 
        // Dirchlet BCs 
        // flag 0 on the left, flag 1 on the right
        AbstractMesh<T, IDX, ndim> mesh{Tensor<T, 1>{0}, Tensor<T, 1>{1}, Tensor<IDX, 1>{nelem}, 1, 
            Tensor<BOUNDARY_CONDITIONS, 2>{BOUNDARY_CONDITIONS::DIRICHLET, BOUNDARY_CONDITIONS::DIRICHLET},
            Tensor<int, 2>{0, 1}};

        FESpace<T, IDX, ndim> fespace{&mesh, FESPACE_ENUMS::LAGRANGE, FESPACE_ENUMS::GAUSS_LEGENDRE, std::integral_constant<int, order>{}};

        // ========================
        // = Setup Discretization =
        // ========================
        BurgersCoefficients<T, ndim> burgers_coeffs{};

        /// set discretization parameters (default to Peclet Nr = 100)
        burgers_coeffs.mu = (cli_args["mu"]) ? cli_args["mu"].as<T>() : 0.01;
        burgers_coeffs.a[0] = (cli_args["a-adv"]) ? cli_args["a-adv"].as<T>() : 1.0;
        burgers_coeffs.b[0] = (cli_args["b-adv"]) ? cli_args["b-adv"].as<T>() : 0.0;


        BurgersFlux physical_flux{burgers_coeffs};
        BurgersUpwind convective_flux{burgers_coeffs};
        BurgersDiffusionFlux diffusive_flux{burgers_coeffs};
        ConservationLawDDG disc{std::move(physical_flux), std::move(convective_flux), std::move(diffusive_flux)};
        disc.sigma_ic = (cli_args["ddgic_mult"]) ? cli_args["ddgic_mult"].as<T>() : 0.0;
        disc.interior_penalty = cli_args["interior_penalty"];

        // left bc u = 0, right bc u = 1
        disc.dirichlet_callbacks.push_back([](const T*x, T*out){ out[0] = 0.0; });
        disc.dirichlet_callbacks.push_back([](const T*x, T*out){ out[0] = 1.0; });

        fe_layout_right u_layout{fespace, std::integral_constant<std::size_t, neq>{},
            std::true_type{}};
        std::vector<T> u_data(u_layout.size());
        fespan u{u_data.data(), u_layout};

        // ===========================
        // = initialize the solution =
        // ===========================

        std::function<void(const double*, double *)> ic = [](const double*x, double *out){
            out[0] = x[0];
        };

        Projection<T, IDX, ndim, neq> projection{ic};
        // TODO: extract into LinearFormSolver
        std::vector<T> u_local_data(u_layout.size());
        std::vector<T> res_local_data(u_layout.size());
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
                solvers::ElementLinearSolver<T, IDX, ndim, neq> solver{el};
                solver.solve(u_local, res_local);

                // scatter to global array 
                // (note we use 0 as multiplier for current values in global array)
                scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);

            }
        );

#ifdef ICEICLE_USE_PETSC

        std::ofstream ic_out{"ic_residuals.dat"};


        for(int i = 1; i <= 999; ++i){
            mesh.coord[1][0] = 0.001 * i;
            using namespace iceicle::solvers;
            ConvergenceCriteria<T, IDX> conv_criteria{
                .tau_abs = std::numeric_limits<T>::epsilon(),
                .tau_rel = 1e-9,
                .kmax = 5
            };
            PetscNewton solver{fespace, disc, conv_criteria, mpi::comm_world};
            solver.ivis = 1;
            io::DatWriter<T, IDX, ndim> writer{fespace};
            writer.register_fields(u, "u");
            solver.vis_callback = [&](IDX k, Vec res_data, Vec du_data){
                T res_norm;
                PetscCallAbort(PETSC_COMM_WORLD, VecNorm(res_data, NORM_2, &res_norm));
                std::cout << std::setprecision(8);
                std::cout << "itime: " << std::setw(6) << k
                << " | residual l2: " << std::setw(14) << res_norm
                << std::endl;

                writer.write_dat(k, (T) k);
            };

            solver.solve(u);

            std::vector<IDX> selected_traces = {0};
            geo_dof_map geo_map{selected_traces, fespace};
            std::vector<T> r_mdg_data{0.0};
            geo_data_layout mdg_layout{geo_map};
            ic_residual_layout<T, IDX, ndim, neq> ice_layout{geo_map};
            dofspan r_mdg{r_mdg_data, ice_layout};
            form_mdg_residual(fespace, disc, u, geo_map, r_mdg);
            ic_out << mesh.coord[1][0] << " " << r_mdg[0, 0] << std::endl;;
        }

#endif
        // compute L2 error
        T Pe = burgers_coeffs.a[0] / burgers_coeffs.mu;
        std::function<void(T*, T*)> exactfunc = [Pe](T *x, T *out) -> void {
            out[0] = ( 1 - std::exp(x[0] * Pe) ) / (1 - std::exp(Pe));
        };
        T error = l2_error(exactfunc, fespace, u);
        std::cout << "L2 error: " << std::setprecision(9) << error << std::endl;

        // print the exact solution in dat format
        std::ofstream outexact{"iceicle_data/exact.dat"};
        int npoin = 1000;
        constexpr int field_width = 18;
        constexpr int precision = 10;
        for(int ipoin = 0; ipoin < npoin; ++ipoin){
            T dx = 1.0 / (npoin - 1);
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

