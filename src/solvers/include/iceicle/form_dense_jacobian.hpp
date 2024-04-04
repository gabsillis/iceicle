#pragma once
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
#include "mdspan/mdspan.hpp"
#include <span>

namespace ICEICLE::SOLVERS{

    template<
        class T,
        class IDX,
        int ndim,
        class disc_class,
        class uLayoutPolicy,
        class uAccessorPolicy,
        class resLayoutPolicy
    >
    auto form_dense_jacobian_fd(
        FE::FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        FE::fespan<T, resLayoutPolicy> res,
        FE::node_selection_span auto mdg_residual,
        LINALG::out_matrix auto jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon())
    ) {

        IDX total_neq = res.size() + mdg_residual.size();
        FE::nodeset_dof_map<IDX>& nodeset = mdg_residual.get_layout().nodeset;
        IDX total_nvar = u.size() + ndim * nodeset.selected_nodes.size();
    }
}
