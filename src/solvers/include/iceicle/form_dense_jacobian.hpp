#pragma once
#include "iceicle/anomaly_log.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/node_set_layout.hpp"
#include "iceicle/form_residual.hpp"
#include "iceicle/linalg/linalg_utils.hpp"

namespace iceicle::solvers{

    template<
        class T,
        class IDX,
        int ndim,
        class disc_class,
        int neq_mdg
    >
    auto form_dense_jacobian_fd(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        nodeset_dof_map<IDX>& nodeset,
        std::span<T> u,
        std::span<T> res,
        linalg::out_matrix auto jac,
        std::integral_constant<int, neq_mdg> neq_mdg_arg,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon())
    ) -> void {
        using namespace util;
        // check sizes
        if(jac.extent(0) < res.size()){
            AnomalyLog::log_anomaly(Anomaly{"jacobian rows is less than number of equations", general_anomaly_tag{}});
        }
        if(jac.extent(1) < u.size()){
            AnomalyLog::log_anomaly(Anomaly{"jacobian cols is less than number of unknowns", general_anomaly_tag{}});
        }
       
        // form the unperturbed residual
        form_residual(fespace, disc, nodeset, u, res, neq_mdg_arg);

        // do the perturbed residuals
        std::vector<T> resp(res.size());
        for(IDX jdof = 0; jdof < u.size(); ++jdof){
            T uold = u[jdof];
            u[jdof] += epsilon;
            form_residual(fespace, disc, nodeset, u, std::span{resp}, neq_mdg_arg);
            for(IDX ieq = 0; ieq < res.size(); ++ieq){
                jac[ieq, jdof] = (resp[ieq] - res[ieq]) / epsilon;
            }
            u[jdof] = uold;
        }

    }


    template<
        class T,
        class IDX,
        int ndim,
        class disc_class
    >
    auto form_dense_jacobian_fd(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        geo_dof_map<T, IDX, ndim>& geo_map,
        std::span<T> u,
        std::span<T> res,
        linalg::out_matrix auto jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon())
    ) -> void {
        using namespace util;
        // check sizes
        if(jac.extent(0) < res.size()){
            AnomalyLog::log_anomaly(Anomaly{"jacobian rows is less than number of equations", general_anomaly_tag{}});
        }
        if(jac.extent(1) < u.size()){
            AnomalyLog::log_anomaly(Anomaly{"jacobian cols is less than number of unknowns", general_anomaly_tag{}});
        }
       
        // form the unperturbed residual
        form_residual(fespace, disc, geo_map, u, res);

        // do the perturbed residuals
        std::vector<T> resp(res.size());
        for(IDX jdof = 0; jdof < u.size(); ++jdof){
            T uold = u[jdof];
            u[jdof] += epsilon;
            form_residual(fespace, disc, geo_map, u, std::span{resp});
            for(IDX ieq = 0; ieq < res.size(); ++ieq){
                jac[ieq, jdof] = (resp[ieq] - res[ieq]) / epsilon;
            }
            u[jdof] = uold;
        }

    }
}
