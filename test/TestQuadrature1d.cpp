#include <iceicle/quadrature/quadrules_1d.hpp>
#include "gtest/gtest.h"

#include <iostream>
#include <iomanip>

TEST(test_quadrature_1d, print_gauss_legendre){

    NUMTOOL::TMP::constexpr_for_range<1,10>([]<int npoin>(){
        std::cout << "===================================================" << std::endl;
        std::cout << npoin << " Point Quadrature Rule" << std::endl;

        QUADRATURE::GaussLegendreQuadrature<double, int, npoin> quadrule{};

        for(int ipoin = 0; ipoin < quadrule.npoints(); ++ipoin){
            auto pt = quadrule.getPoint(ipoin);
            std::cout << std::setprecision(10);
            std::cout << std::setw(6) << ipoin << " | "
            << std::setw(16) << pt.abscisse[0] << " | "
            << std::setw(16) << pt.weight
            << std::endl;;
        }

        std::cout << "===================================================" << std::endl << std::endl;
    });
}
