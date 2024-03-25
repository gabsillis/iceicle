/**
 * @brief procedures for forming residuals from arbitrary discretizations
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once 
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"

namespace ICEICLE::SOLVERS {

    /**
     * @brief requires that the discretization specifies the number of 
     * vector components the solutions have 
     * both compile time and dynamic versions
     *
     * The compile time version is named nv_comp and can be FE::dynamic_ncomp 
     * the dynamic version is named dnv_comp and is either equal to nv_comp 
     * if nv_comp is specified at compile time or the dynamic value
     */
    template< class disc_class >
    concept specifies_ncomp = 
        std::convertible_to<decltype(disc_class::dnv_comp), int>
        && std::convertible_to<decltype(disc_class::nv_comp), int>;



    /**
     * @brief form the residual based on the fespace and discretization 
     * The residual is the function over the vector components for each degree of freedom
     * where getting a residual of 0 solves the discretized portion of the equation 
     * (i.e. for semi-discrete forms M du/dt = residual)
     *
     * @tparam T the floaating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions 
     * @tparam  disc_class the discretization type 
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param u_data the 
     */
    template<
        class T, 
        class IDX,
        int ndim,
        class disc_class,
        class uLayoutPolicy,
        class uAccessorPolicy,
        class resLayoutPolicy
    >
    void form_residual(
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        FE::fespan<T, resLayoutPolicy> res
    )
    requires specifies_ncomp<disc_class>
    {
        using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Trace = ELEMENT::TraceSpace<T, IDX, ndim>;

        // zero out the residual
        res = 0;

        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dg_map.max_el_size_reqirement(disc_class::dnv_comp);
        T *uL_data = new T[max_local_size];
        T *uR_data = new T[max_local_size];
        T *resL_data = new T[max_local_size];
        T *resR_data = new T[max_local_size];


        // boundary faces 
        for(const Trace &trace : fespace.get_boundary_traces()){
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            FE::dofspan uL{uL_data, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            FE::dofspan uR{uR_data, uR_layout};

            auto resL_layout = res.create_element_layout(trace.elL.elidx);
            FE::dofspan resL{resL_data, resL_layout};

            // extract the compact values from the global u view
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residual
            resL = 0;

            disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resL);

            FE::scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
        }

        // interior faces 
        for(const Trace &trace : fespace.get_interior_traces()){
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            FE::dofspan uL{uL_data, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            FE::dofspan uR{uR_data, uR_layout};

            auto resL_layout = res.create_element_layout(trace.elL.elidx);
            FE::dofspan resL{resL_data, resL_layout};
            auto resR_layout = res.create_element_layout(trace.elR.elidx);
            FE::dofspan resR{resR_data, resR_layout};

            // extract the compact values from the global u view
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residual
            resL = 0;
            resR = 0;

           disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resL, resR); 

           FE::scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
           FE::scatter_elspan(trace.elR.elidx, 1.0, resR, 1.0, res);
        }

        // domain integral
        for(const Element &el : fespace.elements){
            // set up compact data views (reuse the storage defined for traces)
            auto uel_layout = u.create_element_layout(el.elidx);
            FE::dofspan u_el{uL_data, uel_layout};

            auto ures_layout = res.create_element_layout(el.elidx);
            FE::dofspan res_el{resL_data, ures_layout};

            // extract the compact values from the global u view 
            FE::extract_elspan(el.elidx, u, u_el);

            // zero out the residual 
            res_el = 0;

            disc.domainIntegral(el, fespace.meshptr->nodes, u_el, res_el);

            FE::scatter_elspan(el.elidx, 1.0, res_el, 1.0, res);
        }

        delete[] uL_data;
        delete[] uR_data;
        delete[] resL_data;
        delete[] resR_data;
    }
}
