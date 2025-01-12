/**
 * @brief procedures for forming residuals from arbitrary discretizations
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once 
#include "iceicle/anomaly_log.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/geo_layouts.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/fe_function/trace_layout.hpp"
#include "iceicle/fe_function/component_span.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include <type_traits>

#ifdef ICEICLE_USE_MPI
#include "iceicle/mpi_type.hpp"
#include <mpi.h>
#endif

namespace iceicle::solvers {

    /**
     * @brief requires that the discretization specifies the number of 
     * vector components the solutions have 
     * both compile time and dynamic versions
     *
     * The compile time version is named nv_comp and can be dynamic_ncomp 
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
     * @param u the current solution 
     * @param res the residual 
     * @param comm the multi-process communicator
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
        FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        fespan<T, resLayoutPolicy> res,
        mpi::communicator_type comm
    )
    requires specifies_ncomp<disc_class>
    {
        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;

        // zero out the residual
        res = 0;

        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dofs.max_el_size_reqirement(disc_class::nv_comp);
        T *uL_data = new T[max_local_size];
        T *uR_data = new T[max_local_size];
        T *resL_data = new T[max_local_size];
        T *resR_data = new T[max_local_size];

        // get the ghost element information 
        u.sync_mpi(comm);

        // boundary faces 
        for(const Trace &trace : fespace.get_boundary_traces()){

            if(trace.face->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM) {

#ifdef ICEICLE_USE_MPI
                 auto [jrank, imleft] = decode_mpi_bcflag(trace.face->bcflag);
                 // set up compact data layouts
                 auto uL_layout = u.create_element_layout(trace.elL.elidx);
                 dofspan uL{uL_data, uL_layout};
                 // translate the parallel index to a local one
                 compact_layout_right<IDX, disc_class::nv_comp> uR_layout{trace.elR};
                 dofspan uR{uR_data, uR_layout};

                 auto resL_layout = res.create_element_layout(trace.elL.elidx);
                 dofspan resL{resL_data, resL_layout};
                 compact_layout_right<IDX, disc_class::nv_comp> resR_layout{trace.elR};
                 dofspan resR{resR_data, resR_layout};

                 // extract the compact values from the global u view
                 extract_elspan(trace.elL.elidx, u, uL);
                 extract_elspan(trace.elR.elidx, u, uL);

                 // zero out residual 
                 resL = 0;

                 disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR);
                 if(imleft){
                     // scatter only the left
                     scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
                 } else {
                     // scatter only the right
                     scatter_elspan(trace.elR.elidx, 1.0, resR, 1.0, res);
                 }
#else 
            util::AnomalyLog::log_anomaly(util::Anomaly{"Built without mpi, parallel communication boundary condition will not work", util::general_anomaly_tag{}});
#endif

            } else {
                // set up compact data views
                auto uL_layout = u.create_element_layout(trace.elL.elidx);
                dofspan uL{uL_data, uL_layout};
                auto uR_layout = u.create_element_layout(trace.elR.elidx);
                dofspan uR{uR_data, uR_layout};

                auto resL_layout = res.create_element_layout(trace.elL.elidx);
                dofspan resL{resL_data, resL_layout};

                // extract the compact values from the global u view
                extract_elspan(trace.elL.elidx, u, uL);
                extract_elspan(trace.elR.elidx, u, uR);

                // zero out the residual
                resL = 0;

                disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resL);

                scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
            }

        }

        // interior faces 
        for(const Trace &trace : fespace.get_interior_traces()){
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan uL{uL_data, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan uR{uR_data, uR_layout};

            auto resL_layout = res.create_element_layout(trace.elL.elidx);
            dofspan resL{resL_data, resL_layout};
            auto resR_layout = res.create_element_layout(trace.elR.elidx);
            dofspan resR{resR_data, resR_layout};

            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residual
            resL = 0;
            resR = 0;

           disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR); 

           scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
           scatter_elspan(trace.elR.elidx, 1.0, resR, 1.0, res);
        }

        // domain integral
        for(const Element &el : fespace.elements){
            // set up compact data views (reuse the storage defined for traces)
            auto uel_layout = u.create_element_layout(el.elidx);
            dofspan u_el{uL_data, uel_layout};

            auto ures_layout = res.create_element_layout(el.elidx);
            dofspan res_el{resL_data, ures_layout};

            // extract the compact values from the global u view 
            extract_elspan(el.elidx, u, u_el);

            // zero out the residual 
            res_el = 0;

            disc.domain_integral(el, u_el, res_el);

            scatter_elspan(el.elidx, 1.0, res_el, 1.0, res);
        }

        delete[] uL_data;
        delete[] uR_data;
        delete[] resL_data;
        delete[] resR_data;
    }

    template<
        class T, 
        class IDX,
        int ndim,
        class disc_class,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    auto form_mdg_residual(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        node_selection_span auto mdg_residual
    ) -> void {
        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;
        using index_type = IDX;

        // zero out the residual 
        mdg_residual = 0;

        // preallocate storage for compact views of u 
        const std::size_t max_local_size =
            fespace.dofs.max_el_size_reqirement(disc_class::dnv_comp);
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);
        std::vector<T> res_storage{};

        const nodeset_dof_map<index_type>& nodeset = mdg_residual.get_layout().nodeset; 

        // loop over the boundary faces in the selection 
        for(index_type itrace : nodeset.selected_traces){
            Trace& trace = fespace.traces[itrace];
            
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan uL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan uR{uR_storage, uR_layout};

            trace_layout_right<IDX, decltype(mdg_residual)::static_extent()> res_layout{trace};
            res_storage.resize(res_layout.size());
            dofspan res{res_storage, res_layout};

            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // zero out then get interface conservation residual 
            res = 0;
            disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);

            scatter_facspan(trace, 1.0, res, 1.0, mdg_residual);
        }
    }


    template<
        class T,
        class IDX,
        int ndim,
        class disc_class,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    auto form_mdg_residual(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        const geo_dof_map<T, IDX, ndim>& geo_map,
        icespan auto mdg_residual
    ) -> void {
        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;
        using index_type = IDX;

        // zero out the residual 
        mdg_residual = 0;


        // preallocate storage for compact views of u 
        const std::size_t max_local_size =
            fespace.dofs.max_el_size_reqirement(disc_class::dnv_comp);
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);
        std::vector<T> res_storage{};

        // loop over the boundary faces in the selection 
        for(index_type itrace : geo_map.selected_traces){
            Trace& trace = fespace.traces[itrace];
            
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan uL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan uR{uR_storage, uR_layout};

            trace_layout_right<IDX, disc_class::nv_comp> res_layout{trace};
            res_storage.resize(res_layout.size());
            dofspan res{res_storage, res_layout};

            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // zero out then get interface conservation residual 
            res = 0;
            disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);

            scatter_facspan(trace, 1.0, res, 1.0, mdg_residual);
        }
    }

    /**
     * calculate the contiguous storage requirement to represent the dg residual and 
     * selected mdg dofs in a single residual array 
     */
    template<class T, class IDX, int ndim, class disc_class, int neq_mdg>
    auto calculate_residual_size_square_mdg(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        nodeset_dof_map<IDX>& nodeset,
        std::integral_constant<int, neq_mdg>& neq_mdg_arg
    ) -> IDX {
        return fespace.dofs.calculate_size_requirement(disc_class::nv_comp())
            + nodeset.size() * neq_mdg;
    }

    template<class T, class IDX, int ndim, class disc_class, int neq_mdg>
    auto form_residual(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        nodeset_dof_map<IDX>& nodeset,
        std::span<T> u,
        std::span<T> res,
        std::integral_constant<int, neq_mdg> neq_mdg_arg
    ) -> void {

        // create all the layouts
        fe_layout_right dg_layout{fespace.dofs, tmp::to_size<disc_class::nv_comp>()};
        node_selection_layout<IDX, ndim> node_layout{nodeset};
        node_selection_layout<IDX, neq_mdg> mdg_layout{nodeset};

        // create views over the u and res arrays
        fespan u_dg{u.data(), dg_layout};
        dofspan u_nodes{std::span{u.begin() + dg_layout.size(), u.end()}, node_layout};
        fespan res_dg{res.data(), dg_layout};
        dofspan res_mdg{std::span{res.begin() + dg_layout.size(), res.end()}, mdg_layout};

        // set the mesh from u_nodes
        scatter_node_selection_span(1.0, u_nodes, 0.0, fespace.meshptr->coord);

        form_residual(fespace, disc, u_dg, res_dg);
        form_mdg_residual(fespace, disc, u_dg, res_mdg);
    }

    template<class T, class IDX, int ndim, class disc_class>
    auto form_residual(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        const geo_dof_map<T, IDX, ndim>& geo_map,
        std::span<T> u,
        std::span<T> res
    ) -> void {

        // create all the layouts
        fe_layout_right dg_layout{fespace.dofs, tmp::to_size<disc_class::nv_comp>()};
        geo_data_layout x_layout{geo_map};
        ic_residual_layout<T, IDX, ndim, disc_class::nv_comp> ic_layout{geo_map};


        // create views over the u and res arrays
        fespan u_dg{u.data(), dg_layout};
        fespan res_dg{res.data(), dg_layout};
        component_span x{std::span{u.begin() + dg_layout.size(), u.end()}, x_layout};
        dofspan res_mdg{std::span{res.begin() + dg_layout.size(), res.end()}, ic_layout};

        // apply the geometric parameterization to the mesh
        update_mesh(x, *(fespace.meshptr));

        form_residual(fespace, disc, u_dg, res_dg);
        form_mdg_residual(fespace, disc, u_dg, geo_map, res_mdg);
    }
}
