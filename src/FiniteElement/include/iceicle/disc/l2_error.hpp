/**
 * @brief calculate the l2 error norm of a solution
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/point.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include "iceicle/quadrature/SimplexQuadrature.hpp"
#include <algorithm>
#include <cmath>
#include <iceicle/fespace/fespace.hpp>
#include <functional>

namespace iceicle {

    /**
     * @tparam T the floating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam uLayoutPolicy the layout policy for the finite element soution 
     * @tparam uAccessorPolicy the accessor policy for the finite element solution 
     *
     * @param exact_sol the exact solution to compare to
     *   f(x, out)
     *   where:
     *   x - the physical domain coordinate [size = ndim]
     *   out - the value at x [size = ncomp]
     *
     * @param fespace the finite element space
     * @param coord the node coordinate array 
     * @param fedata the finite element solution coefficients
     */
    template<
        class T,
        class IDX,
        int ndim,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    T l2_error(
        std::function<void(T*, T*)> exact_sol,
        FESpace<T, IDX, ndim> &fespace,
        fespan<T, uLayoutPolicy, uAccessorPolicy> fedata
    ) {
        using Element = FiniteElement<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        std::vector<T> l2_eq(fedata.nv(), 0.0);
        // reserve data
        std::vector<T> feval(fedata.nv());
        std::vector<T> u(fedata.nv());

        // make high accuracy quadrature rules
        auto quadrule_hypercube = HypercubeGaussLegendre<T, IDX, ndim, 
             2 * (build_config::FESPACE_BUILD_PN + build_config::FESPACE_BUILD_GEO_PN + 1)>{};
        auto quadrule_simplex= GrundmannMollerSimplexQuadrature<T, IDX, ndim, 
             2 * (build_config::FESPACE_BUILD_PN + build_config::FESPACE_BUILD_GEO_PN + 1)>{};

        // loop over quadrature points
        for(const Element &el : fespace.elements) {
            QuadratureRule<T, IDX, ndim>* quadrule;
            switch(el.trans->domain_type){
                case DOMAIN_TYPE::HYPERCUBE:
                    quadrule = &quadrule_hypercube;
                    break;
                case DOMAIN_TYPE::SIMPLEX:
                    quadrule = &quadrule_simplex;
                    break;
            }
            for(int iqp = 0; iqp < quadrule->npoints(); ++iqp) {
                // convert the quadrature point to the physical domain
                const QuadraturePoint<T, ndim> quadpt = quadrule->getPoint(iqp);
                Point phys_pt = el.transform(quadpt.abscisse);

                // calculate the jacobian determinant
                auto J = el.jacobian(quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                exact_sol(phys_pt.data(), feval.data());

                // evaluate the basis functions
                std::vector<T> bi(el.nbasis());
                el.eval_basis(quadpt.abscisse, bi.data());

                // construct the solution
                std::fill(u.begin(), u.end(), 0.0);
                for(IDX ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    for(IDX iv = 0; iv < fedata.nv(); ++iv){
                        u[iv] += bi[ibasis] * fedata[el.elidx, ibasis, iv];
                    }
                }

                // add the contribution of the squared error
                // NOTE: use a safegaurded jacobian for inverted elements
                for(IDX ieq = 0; ieq < fedata.nv(); ieq++){
                    l2_eq[ieq] += std::pow(u[ieq] - feval[ieq], 2) * quadpt.weight 
                        * std::abs(detJ);
                }
            }
        }

        T l2_sum = 0;
        for(int ieq = 0; ieq < fedata.nv(); ++ieq){ l2_sum += l2_eq[ieq]; }
        return std::sqrt(l2_sum);
    }

    /**
     * @tparam T the floating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam uLayoutPolicy the layout policy for the finite element soution 
     * @tparam uAccessorPolicy the accessor policy for the finite element solution 
     *
     * @param exact_sol the exact solution to compare to
     *   f(x, out)
     *   where:
     *   x - the physical domain coordinate [size = ndim]
     *   out - the value at x [size = ncomp]
     *
     * @param fespace the finite element space
     * @param coord the node coordinate array 
     * @param fedata the finite element solution coefficients
     */
    template<
        class T,
        class IDX,
        int ndim,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    T l1_error(
        std::function<void(T*, T*)> exact_sol,
        FESpace<T, IDX, ndim> &fespace,
        fespan<T, uLayoutPolicy, uAccessorPolicy> fedata
    ) {
        using Element = FiniteElement<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;
       
        auto coord = fespace.meshptr->coord;
        std::vector<T> l1_eq(fedata.nv(), 0.0);
        // reserve data
        std::vector<T> feval(fedata.nv());
        std::vector<T> u(fedata.nv());

        // make high accuracy quadrature rules
        auto quadrule_hypercube = HypercubeGaussLegendre<T, IDX, ndim, 
             2 * (build_config::FESPACE_BUILD_PN + build_config::FESPACE_BUILD_GEO_PN + 1)>{};
        auto quadrule_simplex= GrundmannMollerSimplexQuadrature<T, IDX, ndim, 
             2 * (build_config::FESPACE_BUILD_PN + build_config::FESPACE_BUILD_GEO_PN + 1)>{};

        // loop over quadrature points
        for(const Element &el : fespace.elements) {
            QuadratureRule<T, IDX, ndim>* quadrule;
            switch(el.trans->domain_type){
                case DOMAIN_TYPE::HYPERCUBE:
                    quadrule = &quadrule_hypercube;
                    break;
                case DOMAIN_TYPE::SIMPLEX:
                    quadrule = &quadrule_simplex;
                    break;
            }
            for(int iqp = 0; iqp < quadrule->npoints(); ++iqp) {
                // convert the quadrature point to the physical domain
                const QuadraturePoint<T, ndim> quadpt = quadrule->getPoint(iqp);
                Point phys_pt = el.transform(quadpt.abscisse);

                // calculate the jacobian determinant
                auto J = el.jacobian(quadpt.abscisse);
                T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                // evaluate the function at the point in the physical domain
                exact_sol(phys_pt.data(), feval.data());

                // evaluate the basis functions
                std::vector<T> bi(el.nbasis());
                el.eval_basis(quadpt.abscisse, bi.data());

                // construct the solution
                std::fill(u.begin(), u.end(), 0.0);
                for(IDX ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                    for(IDX iv = 0; iv < fedata.nv(); ++iv){
                        u[iv] += bi[ibasis] * fedata[el.elidx, ibasis, iv];
                    }
                }

                // add the contribution of the squared error
                // NOTE: use a safegaurded jacobian for inverted elements
                for(IDX ieq = 0; ieq < fedata.nv(); ieq++){
                    l1_eq[ieq] += std::abs(u[ieq] - feval[ieq]) * quadpt.weight
                        * std::abs(detJ);
                }
            }
        }

        T l1_sum = 0;
        for(int ieq = 0; ieq < fedata.nv(); ++ieq){ l1_sum += l1_eq[ieq]; }
        return l1_sum;
    }



    /**
     * @tparam T the floating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam uLayoutPolicy the layout policy for the finite element soution 
     * @tparam uAccessorPolicy the accessor policy for the finite element solution 
     *
     * @param exact_sol the exact solution to compare to
     *   f(x, out)
     *   where:
     *   x - the physical domain coordinate [size = ndim]
     *   out - the value at x [size = ncomp]
     *
     * @param fespace the finite element space
     * @param coord the node coordinate array 
     * @param fedata the finite element solution coefficients
     */
    template<
        class T,
        class IDX,
        int ndim,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    std::vector<T> linf_error(
        std::function<void(T*, T*)> exact_sol,
        FESpace<T, IDX, ndim> &fespace,
        fespan<T, uLayoutPolicy, uAccessorPolicy> fedata
    ) {
        using Element = FiniteElement<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        auto coord = fespace.meshptr->coord;
        std::vector<T> l2_eq(fedata.nv(), 0.0);
        // reserve data
        std::vector<T> feval(fedata.nv());
        std::vector<T> u(fedata.nv());
        std::vector<T> bi_data(fespace.dg_map.max_el_size_reqirement(1));

        std::vector<T> max_error(fedata.nv());

        std::ranges::fill(max_error, 0);
        if constexpr (ndim == 1){

            // sample the space 
            for(const Element &el : fespace.elements) {
                for(int isample = 0; isample <= 100; ++isample){
                    const Point pt = {-1.0 + 0.02 * isample };
                    Point phys_pt = el.transform(pt);

                    // calculate the jacobian determinant
                    auto J = el.jacobian(pt);
                    T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

                    // evaluate the function at the point in the physical domain
                    exact_sol(phys_pt.data(), feval.data());

                    // evaluate the basis functions
                    el.eval_basis(pt, bi_data.data());

                    // construct the solution
                    std::fill(u.begin(), u.end(), 0.0);
                    for(IDX ibasis = 0; ibasis < el.nbasis(); ++ibasis){
                        for(IDX iv = 0; iv < fedata.nv(); ++iv){
                            u[iv] += bi_data[ibasis] * fedata[el.elidx, ibasis, iv];
                        }
                    }

                    // get the max abs val of error 
                    for(int iv = 0; iv < fedata.nv(); ++iv)
                        max_error[iv] = std::max(max_error[iv], std::abs(u[iv] - feval[iv]));
                }
            }
        }

        return max_error;
    }


    /**
     * @tparam T the floating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam uLayoutPolicy the layout policy for the finite element soution 
     * @tparam uAccessorPolicy the accessor policy for the finite element solution 
     *
     * @param exact_sol the exact solution to compare to
     *   f(x, out)
     *   where:
     *   x - the physical domain coordinate [size = ndim]
     *   out - the value at x [size = ncomp]
     *
     * @param fespace the finite element space
     * @param coord the node coordinate array 
     * @param fedata the finite element solution coefficients
     */
    template<
        class T,
        class IDX,
        int ndim,
        class DiscType,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    T ic_error(
        FESpace<T, IDX, ndim> &fespace,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        DiscType disc
    ) {
        using Trace = TraceSpace<T, IDX, ndim>;

        // preallocate storage for compact views of u 
        const std::size_t max_local_size =
            fespace.dg_map.max_el_size_reqirement(DiscType::nv_comp);
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);
        std::vector<T> res_storage{};

        std::vector<T> total_residual(DiscType::nv_comp);
        std::ranges::fill(total_residual, 0.0);

        if(iceicle::mpi::mpi_world_size() > 1){
            std::cout << "parallel not implemented for interface error" << std::endl;
        }

        for(const Trace& trace : fespace.get_interior_traces()) {
            
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan unkelL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan unkelR{uR_storage, uR_layout};


            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, unkelL);
            extract_elspan(trace.elR.elidx, u, unkelR);

            // interface residual calculation
            static constexpr int neq = DiscType::nv_comp;
            using namespace MATH::MATRIX_T;
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            using FiniteElement = FiniteElement<T, IDX, ndim>;

            const FiniteElement &elL = trace.elL;
            const FiniteElement &elR = trace.elR;

            // Basis function scratch space 
            std::vector<T> bitrace(trace.nbasis_trace());
            std::vector<T> gradbL_data(elL.nbasis() * ndim);
            std::vector<T> gradbR_data(elR.nbasis() * ndim);

            // solution scratch space 
            std::array<T, neq> uL;
            std::array<T, neq> uR;
            std::array<T, neq * ndim> graduL_data;
            std::array<T, neq * ndim> graduR_data;
            std::array<T, neq * ndim> grad_ddg_data;

            for(int iqp = 0; iqp < trace.nQP(); ++iqp){
                const QuadraturePoint<T, ndim - 1> &quadpt = trace.getQP(iqp);

                // calculate the riemannian metric tensor root
                auto Jfac = trace.face->Jacobian(fespace.meshptr->coord, quadpt.abscisse);
                T sqrtg = trace.face->rootRiemannMetric(Jfac, quadpt.abscisse);

                // calculate the normal vector 
                auto normal = calc_ortho(Jfac);
                auto unit_normal = normalize(normal);

                // get the basis functions, derivatives, and hessians
                // (derivatives are wrt the physical domain)
                auto biL = trace.eval_basis_l_qp(iqp);
                auto biR = trace.eval_basis_r_qp(iqp);
                trace.eval_trace_basis_qp(iqp, bitrace.data());
                auto gradBiL = trace.eval_phys_grad_basis_l_qp(iqp, gradbL_data.data());
                auto gradBiR = trace.eval_phys_grad_basis_r_qp(iqp, gradbR_data.data());

                // construct the solution on the left and right
                std::ranges::fill(uL, 0.0);
                std::ranges::fill(uR, 0.0);
                for(int ieq = 0; ieq < neq; ++ieq){
                    for(int ibasis = 0; ibasis < elL.nbasis(); ++ibasis)
                        { uL[ieq] += unkelL[ibasis, ieq] * biL[ibasis]; }
                    for(int ibasis = 0; ibasis < elR.nbasis(); ++ibasis)
                        { uR[ieq] += unkelR[ibasis, ieq] * biR[ibasis]; }
                }

                // get the solution gradient and hessians
                auto graduL = unkelL.contract_mdspan(gradBiL, graduL_data.data());
                auto graduR = unkelR.contract_mdspan(gradBiR, graduR_data.data());

                if(trace.face->bctype != BOUNDARY_CONDITIONS::INTERIOR){
                    switch(trace.face->bctype){
                        case BOUNDARY_CONDITIONS::DIRICHLET:
                            {
                                if(trace.elL.elidx != trace.elR.elidx)
                                    std::cout << "warning: elements do not match" << std::endl;

                                // calculate the physical domain position
                                MATH::GEOMETRY::Point<T, ndim> phys_pt;
                                trace.face->transform(quadpt.abscisse, fespace.meshptr->coord, phys_pt);

                                // Get the values at the boundary 
                                disc.dirichlet_callbacks[trace.face->bcflag](phys_pt.data(), uR.data());

                            } break;
                        default:
                            for(int ieq = 0; ieq < neq; ++ieq)
                                total_residual[ieq] += 0;
                            continue;
                    }
                }

                // get the physical flux on the left and right
                Tensor<T, neq, ndim> fluxL = disc.phys_flux(uL, graduL);
                Tensor<T, neq, ndim> fluxR = disc.phys_flux(uR, graduR);

                // calculate the jump in normal fluxes
                Tensor<T, neq> jumpflux{};
                for(int ieq = 0; ieq < neq; ++ieq) 
                    jumpflux[ieq] = dot(fluxR[ieq], unit_normal) - dot(fluxL[ieq], unit_normal);

                // scatter unit normal times interface conservation to residual
                for(int ieq = 0; ieq < neq; ++ieq){
                    T ic_res = std::abs(jumpflux[ieq]) * sqrtg * quadpt.weight;
                    total_residual[ieq] += ic_res; 
                }
            }
        }

        T sum = 0;
        for(int ieq = 0; ieq < DiscType::nv_comp; ++ieq){
            sum += total_residual[ieq]; 
        }
        return sum;

    }
}
