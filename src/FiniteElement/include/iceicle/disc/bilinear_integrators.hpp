#pragma once

#include "iceicle/element/finite_element.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
#include "iceicle/fe_function/fespan.hpp"
namespace iceicle {


    /// @brief Integrates \f $(\lambda \nabla u, \nabla v)$ \f
    template<class T, class IDX, int ndim>
    struct DiffusionIntegrator {
        T lambda; /// @brief diffusion coefficient

        constexpr
        auto form_operator(
            const FiniteElement<T, IDX, ndim>& el,
            linalg::out_matrix auto K
        ) const noexcept -> void {
            std::size_t nbasis = el.nbasis();
            std::vector<T> gradx_basis_data(nbasis * ndim);

            // zero out the matrix 
            for(int i = 0; i < nbasis; ++i) 
                for(int j = 0; j < nbasis; ++j)
                    K[i, j] = 0;


            for(int iqp = 0; iqp < el.nQP(); ++iqp){
                const QuadraturePoint<T, ndim> &quadpt = el.getQP(iqp);

                // calculate the jacobian determinant 
                auto J = el.jacobian(quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // get the basis values and gradients in the physical domain
                auto bi = el.eval_basis_qp(iqp);
                auto gradx_basis = el.eval_phys_grad_basis(quadpt.abscisse, J,
                        el.eval_grad_basis_qp(iqp), gradx_basis_data.data());

                // form the matrix
                for(std::size_t ibasis = 0; ibasis < nbasis; ++ibasis) {
                    for(std::size_t itest = 0; itest < nbasis; ++itest) {
                        T gradu_dot_gradv = 0;
                        for(int idim = 0; idim < ndim; ++idim)
                            gradu_dot_gradv += gradx_basis[ibasis, idim] * gradx_basis[itest, idim];
                        K[ibasis, itest] += lambda * gradu_dot_gradv * detJ * quadpt.weight;
                    }
                }
            }
        }

    };
}
