#pragma once
#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/mesh/mesh.hpp"

namespace iceicle {

    template<class T, class IDX, int ndim>
    struct FiniteDifference {
        using Trace_t = TraceSpace<T, IDX, ndim>;

        template<class Operator, class uL_span_t, class uR_span_t, class res_span_t>
        auto eval_res_and_jac(
            Operator&& boundary_integral,
            const Trace_t& trace,
            AbstractMesh<T, IDX, ndim>& mesh,
            uL_span_t uL,
            uR_span_t uR,
            res_span_t res
        ) -> void 
        {
            // zero out the residual
            res = 0;

            // get the unperturbed residual
            boundary_integral(trace, mesh.coord, uL, uR, res);

            std::vector<T> scratch_resp(res.size());

        }
    };

}
