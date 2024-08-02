
#include "iceicle/linalg/qr.hpp"
#include <iostream>
using namespace iceicle::linalg;

int main(){
    std::array A{
        3.0, 4.0, 4.0,
        4.0, 5.0, 1.0,
        3.0, 2.0, 2.0,
        9.0, 2.0, 5.0
    };

    std::mdspan Amat{A.data(), std::extents{4, 3}};
    std::array<double, 3> taudata;
    std::mdspan tau{taudata.data(), std::extents{3}};
    householder_qr(Amat, tau);

    std::cout << "A: " << std::endl;
    std::cout << Amat;
    std::cout << std::endl;

    std::array<double, 12> Q;
    std::mdspan Qmat{Q.data(), std::extents{4, 3}};
    form_Q(Amat, tau, Qmat);
    
    std::cout << "Q: " << std::endl;
    std::cout << Qmat;
    std::cout << std::endl;

    std::array x{4.0, 2.0, 2.0, 1.0};
    std::array<double, 4> y;
    std::mdspan yvec{y.data(), std::extents{4}};
    apply_QT(Amat, tau, std::mdspan{x.data(), std::extents{4}}, yvec);
    std::cout << yvec;
}
