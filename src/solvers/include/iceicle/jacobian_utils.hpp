#pragma once
#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fd_utils.hpp"
#include <ranges>

#ifdef ICEICLE_USE_PETSC 
#include "petscmat.h"
#endif

namespace iceicle {

    /// @brief utility function to compute the storage requirement for a given jacobian computation 
    ///
    /// Jacobian is df / du where f is the rows and u is the columns
    ///
    /// @param input the input data view (variables u jacobian is wrt)
    /// @param output the output data view (values f jacobian is taking derivatives of)
    /// @return the storage requirement
    template<class input_span, class output_span>
    [[nodiscard]] inline constexpr
    auto compute_jacobian_storage_requirement(input_span input, output_span output)
    -> std::size_t
    { return output.size() * input.size(); }

    /// @brief utility function to generate an mdspan view over data 
    /// to represent the jacobian given an input and output span 
    /// bounds checked in debug builds
    ///
    /// Jacobian is df / du where f is the rows and u is the columns
    ///
    /// @param input the input data view (variables u jacobian is wrt)
    /// @param output the output data view (values f jacobian is taking derivatives of)
    /// @return the mdspan to represent the jacobian over the data
    template<class input_span, class output_span, std::ranges::contiguous_range R>
    [[nodiscard]] inline constexpr
    auto create_jacobian_mdspan(input_span input, output_span output, R&& data)
    {
#ifndef NDEBUG
        // bounds check
        if(data.size() < compute_jacobian_storage_requirement(input, output))
            util::AnomalyLog::log_anomaly(util::Anomaly{"Bounds error when creating jacobian mdspan.", util::general_anomaly_tag{}});
#endif
        return std::mdspan{data.data(), output.size(), input.size()};
    }

    /// @brief handles finite differences of senstivities 
    /// for finite element computations
    template<class T, class IDX, int ndim>
    struct FiniteDifference {
        using Trace_t = TraceSpace<T, IDX, ndim>;

        /// @brief finite difference options
        /// TODO: implement more options
        enum FD_TYPE {
            FIRST_ORDER,
        };

        // @brief compute the residual and sensitivities for computations over a trace space with a single output span
        // Any sensitivities you do not want to compute can be ignored by passing linalg::empty_matrix{}
        // otherwise, jacobians must be set up to be the right size before passing 
        // check utility functions 
        // - compute_jacobian_storage_requirement 
        // - create_jacobian_mdspan
        // 
        // @param res_op the operation the compute the residual 
        // @param trace the trace space to calculate over
        // @param mesh the mesh
        // @param uL the data for the element on the canonical "left" side of the trace 
        // @param uR the data for the element on the canonical "right" side of the trace
        // @param epsilon the finite difference perturbation (gets scaled by residual norm)
        // @param res the output span for the residual
        // @param jac_wrt_uL the sensitivities of res wrt uL
        // @param jac_wrt_uR the sensitivities of res wrt uR
        // @param jac_wrt_x the sensitivities of x wrt mesh coordinates
        template<class Operator, class uL_span_t, class uR_span_t, class res_span_t>
        auto eval_res_and_jac(
            Operator&& res_op,
            const Trace_t& trace,
            AbstractMesh<T, IDX, ndim>& mesh,
            uL_span_t uL,
            uR_span_t uR,
            T epsilon,
            res_span_t res,
            linalg::out_matrix auto jac_wrt_uL,
            linalg::out_matrix auto jac_wrt_uR,
            linalg::out_matrix auto jac_wrt_x
        ) -> void 
        requires(elspan<uL_span_t> && elspan<uR_span_t>)
        {
            // index types
            using uL_idx_t = uL_span_t::index_type;
            using uR_idx_t = uR_span_t::index_type;
            using res_idx_t = res_span_t::index_type;

            // zero out the residual and jacobians
            res = 0;
            linalg::fill(jac_wrt_uL, 0.0);
            linalg::fill(jac_wrt_uR, 0.0);
            linalg::fill(jac_wrt_x, 0.0);

            // get the unperturbed residual
            res_op(trace, mesh.coord, uL, uR, res);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

            // set up data for perturbed residuals
            std::vector<T> scratch_resp(res.size());
            dofspan resp{scratch_resp, res.get_layout()};

            // perturb uL
            for(uL_idx_t idofu = 0; idofu < uL.ndof(); ++idofu) {
                for(uL_idx_t iequ = 0; iequ < uL.nv(); ++iequ) {

                    // column index in the matrix
                    uL_idx_t jcol = uL.index_1d(idofu, iequ);

                    // perturb 
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resp = 0;

                    // get the perturbed residual
                    res_op(trace, mesh.coord, uL, uR, resp);

                    // fill the jacobian 
                    for(res_idx_t idoff = 0; idoff < res.ndof(); ++idoff) {
                        for(res_idx_t ieqf = 0; ieqf < res.nv(); ++ieqf) {
                            IDX irow = res.index_1d(idoff, ieqf);
                            T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                            jac_wrt_uL[irow, jcol] += fd_val;
                        }
                    }

                    // undo the perturbation
                    uL[idofu, iequ] = old_val;
                }
            }
            std::cout << "test val: " << jac_wrt_uL[1, 0] << std::endl;

            // stop if the jacobian with respect to uR is not requested
            if(jac_wrt_uR.empty()) return;

            // perturb uR
            for(uR_idx_t idofu = 0; idofu < uR.ndof(); ++idofu) {
                for(uR_idx_t iequ = 0; iequ < uR.nv(); ++iequ) {

                    // column index in the matrix
                    uR_idx_t jcol = uR.index_1d(idofu, iequ);

                    // perturb 
                    T old_val = uR[idofu, iequ];
                    uR[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resp = 0;

                    // get the perturbed residual
                    res_op(trace, mesh.coord, uL, uR, resp);

                    // fill the jacobian 
                    for(res_idx_t idoff = 0; idoff < res.ndof(); ++idoff) {
                        for(res_idx_t ieqf = 0; ieqf < res.nv(); ++ieqf) {
                            IDX irow = res.index_1d(idoff, ieqf);
                            jac_wrt_uR[irow, jcol] += (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                        }
                    }

                    // undo the perturbation
                    uR[idofu, iequ] = old_val;
                }
            }

            // stop if jacobian with respect to geometry is not requested
            if(jac_wrt_x.empty()) return;


        }
    };

#ifdef ICEICLE_USE_PETSC

    /// @brief scatter a jacobian representing dres/du to a global petsc matrix 
    /// @param jac_local the jacobian data to scatter 
    /// @param iel_u the index of the element representing the intput 
    ///              which sensitivities are wrt 
    /// @param iel_res the index of the element representing the output 
    ///                that we are getting sensitivities of 
    /// @param u the span over input data for the finite element space 
    ///          (determines the layout for input dofs)
    /// @param res the span over output data for the finite element space 
    ///            (determines the layout for output dofs)
    /// @param jac_global the jacobian to scatter to as a petsc matrix 
    /// @paramm comm the mpi communicator
    template<class T, class IDX, class ul_pol, class ua_pol, class resl_pol, class resa_pol>
    auto scatter_jacobian(
        linalg::in_matrix auto jac_local,
        IDX iel_u,
        IDX iel_res,
        fespan<T, ul_pol, ua_pol> u,
        fespan<T, resl_pol, resa_pol> res,
        Mat jac_global,
        MPI_Comm comm
    ) -> void 
    {
        std::size_t dimprod = jac_local.extent(0) * jac_local.extent(1);
        std::vector<PetscInt> idxm(jac_local.extent(0));
        std::vector<PetscInt> idxn(jac_local.extent(1));

        // indices for residual 
        if constexpr(resl_pol::local_dof_contiguous()) {
            PetscInt rowstart = res.get_layout()[iel_res, 0, 0];
            std::iota(idxm.begin(), idxm.end(), rowstart);
        } else {
            int iidxm = 0;
            for(int idof = 0; idof < res.ndof(iel_res); ++idof){
                for(int iv = 0; iv < res.nv(); ++iv, ++iidxm) {
                    idxm[iidxm] = res.get_layout()[iel_res, idof, iv];
                }
            }
        }

        // indices for input
        if constexpr(ul_pol::local_dof_contiguous()) {
            PetscInt rowstart = u.get_layout()[iel_u, 0, 0];
            std::iota(idxn.begin(), idxn.end(), rowstart);
        } else {
            int iidxn = 0;
            for(int idof = 0; idof < u.ndof(iel_u); ++idof){
                for(int iv = 0; iv < u.nv(); ++iv, ++iidxn) {
                    idxn[iidxn] = u.get_layout()[iel_u, idof, iv];
                }
            }
        }

        if constexpr (linalg::contiguous_mdspan<decltype(jac_local)>){
            // add values to the petsc matrix
            PetscCallAbort(comm, MatSetValues(jac_global, jac_local.extent(0),
                idxm.data(), jac_local.extent(1),
                idxn.data(), jac_local.data_handle(), ADD_VALUES));

        } else {
            // get the values
            std::vector<T> values{};
            values.reserve(dimprod);
            for(std::size_t i = 0; i < jac_local.extent(0); ++i){
                for(std::size_t j = 0; j < jac_local.extent(1); ++j){
                    values.push_back(jac_local[i, j]);
                }
            }

            // add values to the petsc matrix
            PetscCallAbort(comm, MatSetValues(jac_global, jac_local.extent(0),
                idxm.data(), jac_local.extent(1),
                idxn.data(), values.data(), ADD_VALUES));
        }
    }
#endif

}
