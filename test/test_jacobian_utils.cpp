#include <gtest/gtest.h>
#include <iceicle/jacobian_utils.hpp>
#include <iceicle/fe_function/el_layout.hpp>
using namespace iceicle;

template<int neq>
struct neq_struct{
    static constexpr int nv_comp = neq;
};

TEST(test_jacobian_utils, test_fd_trace_operator){

    using T = double;
    using IDX = int;
    static constexpr int ndim = 2;
    static constexpr int pn_geo = 2;
    static constexpr int pn_basis = 3;
    static constexpr int neq = 2;
   
    
    // create a uniform mesh
    AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {2, 1}, pn_geo);

    FESpace<T, IDX, ndim> fespace{
        &mesh, FESPACE_ENUMS::LAGRANGE,
        FESPACE_ENUMS::GAUSS_LEGENDRE, 
        tmp::compile_int<pn_basis>()
    };

    const TraceSpace<T, IDX, ndim>& trace = fespace.get_interior_traces()[0];

    
    compact_layout_right<IDX, neq> uL_layout{trace.elL};
    compact_layout_right<IDX, neq> uR_layout{trace.elR};
    trace_layout_right<IDX, neq> res_layout{trace, neq_struct<neq>{}};

    std::vector<T> uL_data(uL_layout.size());
    std::vector<T> uR_data(uR_layout.size());
    // fill with 1, 2, ...
    std::iota(uL_data.begin(), uL_data.end(), 0); 
    std::iota(uR_data.begin(), uR_data.end(), 0);

    std::vector<T> res_data(res_layout.size());

    dofspan uL{uL_data, uL_layout};
    dofspan uR{uR_data, uR_layout};
    dofspan res{res_data, res_layout};

    /// setup jacobians 
    std::vector<T> jdatauL(compute_jacobian_storage_requirement(uL, res));
    auto jac_wrt_uL{create_jacobian_mdspan(uL, res, jdatauL)};
    std::vector<T> jdatauR(compute_jacobian_storage_requirement(uR, res));
    auto jac_wrt_uR{create_jacobian_mdspan(uR, res, jdatauR)};

    /// fill it with nonsense to test clearing out
    std::ranges::fill(jdatauL, 1.23423402343290);
    std::ranges::fill(jdatauR, 1.23423402343290);
    
    // the function we are taking derivatives of
    auto func = [](const TraceSpace<T, IDX, ndim> &trace, NodeArray<T, ndim>& coord, elspan auto uL, elspan auto uR, facspan auto res)
    {
        // equation 1 depends only on uL 
        for(int ibasis_face = 0; ibasis_face < trace.nbasis_trace(); ++ibasis_face){

            // each basis function index depends on one less index of uL
            for(int ibasis_uL = ibasis_face; ibasis_uL < trace.elL.nbasis(); ++ibasis_uL){
                res[ibasis_face, 0] += uL[ibasis_uL, 0] + uL[ibasis_uL, 1];
            }

            for(int ibasis_uR = ibasis_face; ibasis_uR < trace.elR.nbasis(); ++ibasis_uR){
                res[ibasis_face, 1] += uR[ibasis_uR, 0] + uR[ibasis_uR, 1];
            }
        }
    };

    FiniteDifference<T, IDX, ndim> fd;
    fd.eval_res_and_jac(func, trace, mesh, uL, uR, 1e-8, res, jac_wrt_uL, jac_wrt_uR, linalg::empty_matrix{});

    for(int ibasis_face = 0; ibasis_face < trace.nbasis_trace(); ++ibasis_face){
        for(int ibasis_uL = 0; ibasis_uL < trace.elL.nbasis(); ++ibasis_uL){
            T expected_val = ibasis_uL >= ibasis_face ? 1.0 : 0.0;
            auto residual_index = res.index_1d(ibasis_face, 0);
            auto u_index0 = uL.index_1d(ibasis_uL, 0);
            auto u_index1 = uL.index_1d(ibasis_uL, 1);
            ASSERT_NEAR((jac_wrt_uL[residual_index, u_index0]), expected_val, 1e-7) 
                << "residual_index: " << residual_index << ", uL index: " << u_index0;
            ASSERT_NEAR((jac_wrt_uL[residual_index, u_index1]), expected_val, 1e-7)
                << "residual_index: " << residual_index << ", uL index: " << u_index1;
            
            // test the other entries are empty
            residual_index = res.index_1d(ibasis_face, 1);
            ASSERT_NEAR((jac_wrt_uL[residual_index, u_index0]), 0.0, 1e-7)
                << "residual_index: " << residual_index << ", uL index: " << u_index0;
            ASSERT_NEAR((jac_wrt_uL[residual_index, u_index1]), 0.0, 1e-7)
                << "residual_index: " << residual_index << ", uL index: " << u_index1;
        }
        for(int ibasis_uR = 0; ibasis_uR < trace.elR.nbasis(); ++ibasis_uR){
            T expected_val = ibasis_uR >= ibasis_face ? 1.0 : 0.0;
            auto residual_index = res.index_1d(ibasis_face, 1);
            auto u_index0 = uR.index_1d(ibasis_uR, 0);
            auto u_index1 = uR.index_1d(ibasis_uR, 1);
            ASSERT_NEAR((jac_wrt_uR[residual_index, u_index0]), expected_val, 1e-7)
                << "residual_index: " << residual_index << ", uR index: " << u_index0;
            ASSERT_NEAR((jac_wrt_uR[residual_index, u_index1]), expected_val, 1e-7)
                << "residual_index: " << residual_index << ", uR index: " << u_index1;

            // test the other entries are empty
            residual_index = res.index_1d(ibasis_face, 0);
            ASSERT_NEAR((jac_wrt_uR[residual_index, u_index0]), 0.0, 1e-7)
                << "residual_index: " << residual_index << ", uR index: " << u_index0;
            ASSERT_NEAR((jac_wrt_uR[residual_index, u_index1]), 0.0, 1e-7)
                << "residual_index: " << residual_index << ", uR index: " << u_index1;
        }
    }
}

TEST(test_jacobian_utils, test_scatter){
    using T = double;
    using IDX = int;
    static constexpr int ndim = 2;
    static constexpr int pn_geo = 2;
    static constexpr int neq = 1;

    
    AbstractMesh<T, IDX, ndim> mesh({-1.0, 1.0}, {1.0, 1.0}, {2, 2}, pn_geo);

}
    
