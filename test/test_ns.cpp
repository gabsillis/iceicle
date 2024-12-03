#include <gtest/gtest.h>
#include <iceicle/disc/navier_stokes.hpp>

using namespace iceicle;
using namespace navier_stokes;
template<class T2, std::size_t... sizes>
using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T2, sizes...>;

TEST(test_ns, test_reference) {
    ReferenceParameters<double> unit_ref{};
    ASSERT_DOUBLE_EQ(1.0, unit_ref.t);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.l);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.rho);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.u);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.e);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.p);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.T);
    ASSERT_DOUBLE_EQ(1.0, unit_ref.mu);

    Nondimensionalization<double> unit_nondim{create_nondim<double>(unit_ref)};
    ASSERT_DOUBLE_EQ(1.0, unit_nondim.Re);
    ASSERT_DOUBLE_EQ(1.0, unit_nondim.Eu);
    ASSERT_DOUBLE_EQ(1.0, unit_nondim.Sr);
    ASSERT_DOUBLE_EQ(1.0, unit_nondim.e_coeff);
}

TEST(test_ns, test_conversion) {
    // sanity check for conversion between varible sets

    ReferenceParameters<double> ref{
        .t = 1, // s
        .l = 1, // m
        .rho = 1.225, // kg / m^3
        .u = 1, // m / s
        .e = 1, // 
        .p = 10, // for non-unit Euler number
        .T = 273.15, // K
        .mu = 1
    };

    Nondimensionalization<double> nondim{create_nondim(ref)};

    std::array<double, 3> rho_u_T{1.0, 0.5, 1.0};

    CaloricallyPerfectEoS<double, 1> eos;

    std::array<double, 3> rho_u_T_2 = 
        eos.template convert_varset<VARSET::RHO_U_T, VARSET::CONSERVATIVE>(
            eos.template convert_varset<VARSET::CONSERVATIVE, VARSET::RHO_U_T>(rho_u_T, ref, nondim), ref, nondim);
    for(int i = 0; i < 3; ++i) ASSERT_DOUBLE_EQ(rho_u_T_2[i], rho_u_T[i]);
    std::array<double, 3> rho_u_T_3 = 
        eos.template convert_varset<VARSET::RHO_U_T, VARSET::RHO_U_P>(
            eos.template convert_varset<VARSET::RHO_U_P, VARSET::RHO_U_T>(rho_u_T, ref, nondim), ref, nondim);
    for(int i = 0; i < 3; ++i) ASSERT_DOUBLE_EQ(rho_u_T_3[i], rho_u_T[i]);

    std::array<double, 3> cons{1.0, 1.0, 1.0};
    std::array<double, 3> cons2 = 
        eos.template convert_varset<VARSET::CONSERVATIVE, VARSET::RHO_U_P>(
            eos.template convert_varset<VARSET::RHO_U_P, VARSET::CONSERVATIVE>(cons, ref, nondim), ref, nondim);
    for(int i = 0; i < 3; ++i) ASSERT_DOUBLE_EQ(cons2[i], cons[i]);
    std::array<double, 3> cons3 = 
        eos.template convert_varset<VARSET::CONSERVATIVE, VARSET::RHO_U_T>(
            eos.template convert_varset<VARSET::RHO_U_T, VARSET::CONSERVATIVE>(cons, ref, nondim), ref, nondim);
    for(int i = 0; i < 3; ++i) ASSERT_DOUBLE_EQ(cons3[i], cons[i]);
}

TEST(test_ns, test_flux_consistency_vanleer){
    static constexpr int ndim = 3;
    ReferenceParameters<double> ref{};
    CaloricallyPerfectEoS<double, ndim> eos{};
    Physics physics{ref, eos};

    VanLeer numflux{physics};
    Flux pflux{physics};

    std::array u_L1 = {-1.0, -0.001, 0.0, 0.001, 1.0};

    for(double uval : u_L1) {
        std::array<double, 5> u{ 1.0, uval, uval, uval, 1.4 };

        std::array<double, 15> ugrad_data;
        std::ranges::fill(ugrad_data, 0.0);
        std::mdspan ugrad{ugrad_data.data(), std::extents{5, 3}};

        Tensor<double, 5, 3> f_phys = pflux(u, ugrad);

        {
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, 3> normal{1.0, 0.0, 0.0};
            auto f_num = numflux(u, u, normal);
            for(int ieq = 0; ieq < 5; ++ieq)
                ASSERT_NEAR(f_phys[ieq][0], f_num[ieq], 1e-10);
        }
        {
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, 3> normal{0.0, 1.0, 0.0};
            auto f_num = numflux(u, u, normal);
            for(int ieq = 0; ieq < 5; ++ieq)
                ASSERT_NEAR(f_phys[ieq][1], f_num[ieq], 1e-10);
        }
        {
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, 3> normal{0.0, 0.0, 1.0};
            auto f_num = numflux(u, u, normal);
            for(int ieq = 0; ieq < 5; ++ieq)
                ASSERT_NEAR(f_phys[ieq][2], f_num[ieq], 1e-10);
        }
    }
}
