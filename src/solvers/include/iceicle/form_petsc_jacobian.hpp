/**
 * @brief form the jacobian as a petsc matrix
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/component_span.hpp"
#include "iceicle/form_residual.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/petsc_interface.hpp"
#include "iceicle/fd_utils.hpp"
#include <cmath>
#include <limits>
#include <petscsystypes.h>
#include <set>
#include <petscerror.h>
#include <petscmat.h>
#include <mdspan/mdspan.hpp>

namespace iceicle::solvers {

    /**
     * @brief form the jacobian for the given discretization on the given 
     * finite element space using finite differences.
     * Will simultaneously form the residual
     * The ordering of vector components in the jacobian will match the layout provided
     * for the residual view
     *
     * @tparam T the floaating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions 
     * @tparam  disc_class the discretization type 
     *
     * @param fespace the finite element space 
     * @param disc the discretization
     * @param u_data the solution to get the residual for
     * @param [out] res the residual
     * @param [out] jac a petsc matrix to fill with the jacobian 
     * @param epsilon (optional) the epsilon to use for finite difference
     *                NOTE: this gets scaled by the norm of the compact residual vector
     *
     * @param comm (optional) mpi communicator default: MPI_COMM_WORLD
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
    void form_petsc_jacobian_fd(
        FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        fespan<T, resLayoutPolicy> res,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()),
        MPI_Comm comm = MPI_COMM_WORLD 
    ) {

        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;

        using namespace std::experimental;

        // zero out the residual
        res = 0;

        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dofs.max_el_size_reqirement(disc_class::dnv_comp);
        const std::size_t ncomp = disc_class::dnv_comp;

        // get the start indices for the petsc matrix on this processor
        PetscInt proc_range_beg, proc_range_end;
        PetscCallAbort(comm, MatGetOwnershipRange(jac, &proc_range_beg, &proc_range_end));

        // storage for local solution
        std::vector<T> uL_data(max_local_size);
        std::vector<T> uR_data(max_local_size);

        // storage for local residuals
        std::vector<T> resL_data(max_local_size);
        std::vector<T> resLp_data(max_local_size); // perturbed residual
        std::vector<T> resR_data(max_local_size);
        std::vector<T> resRp_data(max_local_size); // perturbed residual
        
        // storage for compact jacobians
        std::vector<T> jacL_data(max_local_size * max_local_size);
        std::vector<T> jacR_data(max_local_size * max_local_size);

        // boundary faces 
        for(const Trace &trace : fespace.get_boundary_traces()) {
            // compact data views 
            dofspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            dofspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            dofspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            dofspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};

            // compact jacobian views 
            mdspan jacL{jacL_data.data(), extents{resL.size(), uL.size()}};
            std::fill_n(jacL_data.begin(), jacL.size(), 0);

            // extract the compact values from the global u view 
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residuals 
            resL = 0;

            // get the unperturbed residual and send to full residual
            disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resL);
            scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = scale_fd_epsilon(epsilon, resL.vector_norm());

            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];

            // perturb and form jacobian 
            for(IDX idofu = 0; idofu < trace.elL.nbasis(); ++idofu){
                for(IDX iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    IDX jcol = uL.get_layout()[idofu, iequ];

                    // perturb
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resLp = 0;

                    // get the perturbed residual
                    disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resLp);

                    // fill jacobian for this perturbation
                    for(IDX idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = uL.get_layout()[idoff, ieqf];
                            jacL[irow, jcol] += (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                        }
                    }

                    // undo the perturbation
                    uL[idofu, iequ] = old_val;
                }
            }

            // TODO: change to a scatter generalized operation to support CG structures
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_L, jacL);
        }

        // interior faces 
        for(const Trace &trace : fespace.get_interior_traces()) {
            // compact data views 
            dofspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            dofspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            dofspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            dofspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};
            dofspan resR{resR_data.data(), res.create_element_layout(trace.elR.elidx)};
            dofspan resRp{resRp_data.data(), res.create_element_layout(trace.elR.elidx)};


            // extract the compact values from the global u view 
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residuals 
            resL = 0;
            resR = 0;

            // get the unperturbed residual and send to full residual
            disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR);
            scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
            scatter_elspan(trace.elR.elidx, 1.0, resR, 1.0, res);

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
            std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = scale_fd_epsilon(epsilon, std::max(resL.vector_norm(), resR.vector_norm()));

            // perturb and form jacobian wrt uL
            // compact jacobian views 
            mdspan jacL{jacL_data.data(), extents{resL.size(), uL.size()}};
            mdspan jacR{jacR_data.data(), extents{resR.size(), uL.size()}};
            std::fill_n(jacL_data.begin(), jacL.size(), 0);
            std::fill_n(jacR_data.begin(), jacR.size(), 0);
            for(IDX idofu = 0; idofu < trace.elL.nbasis(); ++idofu){
                for(IDX iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    IDX jcol = uL.get_layout()[idofu, iequ];

                    // perturb
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // zero out the residuals
                    resLp = 0;
                    resRp = 0;

                    // get the perturbed residual
                    disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resLp, resRp);

                    // fill jacobian for this perturbation
                    for(IDX idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = uL.get_layout()[idoff, ieqf];
                            jacL[irow, jcol] += (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                        }
                    }
                    for(IDX idoff = 0; idoff < trace.elR.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = uR.get_layout()[idoff, ieqf];
                            jacR[irow, jcol] += (resRp[idoff, ieqf] - resR[idoff, ieqf]) / eps_scaled;
                        }
                    }

                    // undo the perturbation
                    uL[idofu, iequ] = old_val;
                }
            }
            // send the jacobians to the petsc matrix 
            // (note global indices uL, then resL/resR)
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_L, jacL);
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_R, 
                    proc_range_beg + glob_index_L, jacR);
            
            // perturb and form jacobian wrt uR
            // make compact jacobian views
            jacL = mdspan{jacL_data.data(), extents{resL.size(), uR.size()}};
            jacR = mdspan{jacR_data.data(), extents{resR.size(), uR.size()}};
            std::fill_n(jacL_data.begin(), jacL.size(), 0);
            std::fill_n(jacR_data.begin(), jacR.size(), 0);
            for(IDX idofu = 0; idofu < trace.elR.nbasis(); ++idofu){
                for(IDX iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    IDX jcol = uR.get_layout()[idofu, iequ];

                    // perturb
                    T old_val = uR[idofu, iequ];
                    uR[idofu, iequ] += eps_scaled;

                    // zero out the residuals
                    resLp = 0;
                    resRp = 0;

                    // get the perturbed residual
                    disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resLp, resRp);

                    // fill jacobian for this perturbation
                    for(IDX idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = uL.get_layout()[idoff, ieqf];
                            jacL[irow, jcol] += (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                        }
                    }
                    for(IDX idoff = 0; idoff < trace.elR.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = uR.get_layout()[idoff, ieqf];
                            jacR[irow, jcol] += (resRp[idoff, ieqf] - resR[idoff, ieqf]) / eps_scaled;
                        }
                    }

                    // undo the perturbation
                    uR[idofu, iequ] = old_val;
                }
            }
            // send the jacobians to the petsc matrix 
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_R, jacL);
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_R, 
                    proc_range_beg + glob_index_R, jacR);
        }

        // domain integral 
        for(const Element &el : fespace.elements) {
            // compact data views 
            dofspan u_el{uL_data.data(), u.create_element_layout(el.elidx)};

            // residual data views 
            dofspan res_el{resL_data.data(), res.create_element_layout(el.elidx)};
            dofspan resp_el{resLp_data.data(), res.create_element_layout(el.elidx)};

            // jacobian data view
            mdspan jac_el{jacL_data.data(), extents{res_el.size(), u_el.size()}};
            std::fill_n(jacL_data.begin(), jac_el.size(), 0);


            // extract the compact values from the global u view 
            extract_elspan(el.elidx, u, u_el);

            res_el = 0;
            disc.domain_integral(el, u_el, res_el);
            disc.domain_integral_jacobian(el, u_el, jac_el); // TODO: combine

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_el = u.get_layout()[el.elidx, 0, 0];
            // send residual to global residual 
            scatter_elspan(el.elidx, 1.0, res_el, 1.0, res);

            // TODO: change to a scatter generalized operation to support CG structures
            petsc::add_to_petsc_mat(jac, proc_range_beg + glob_index_el, 
                    proc_range_beg + glob_index_el, jac_el);
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
    auto form_petsc_mdg_jacobian_fd(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        node_selection_span auto mdg_residual,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()),
        MPI_Comm comm = MPI_COMM_WORLD 
    ) -> void {
        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;
        using index_type = IDX;
        using namespace std::experimental;

        // zero out the residual 
        mdg_residual = 0;

        // get the start indices for the petsc matrix on this processor
        PetscInt proc_range_beg, proc_range_end, mdg_range_beg;
        PetscCallAbort(comm, MatGetOwnershipRange(jac, &proc_range_beg, &proc_range_end));
        mdg_range_beg = proc_range_beg + u.size();

        // preallocate storage for compact views of u 
        const std::size_t max_local_size =
            fespace.dg_map.max_el_size_reqirement(disc_class::dnv_comp);
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);
        std::vector<T> jacL_storage{};
        std::vector<T> jacR_storage{};
        std::vector<T> res_storage{};
        std::vector<T> resp_storage{};

        const nodeset_dof_map<index_type>& nodeset = mdg_residual.get_layout().nodeset; 

        // jacobian of interface condition residual wrt u
        for(index_type itrace: nodeset.selected_traces){
            Trace& trace = fespace.traces[itrace];
            
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan uL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan uR{uR_storage, uR_layout};

            trace_layout_right<IDX, decltype(mdg_residual)::static_extent()> res_layout{trace};
            res_storage.resize(res_layout.size());
            dofspan res{res_storage, res_layout};
            resp_storage.resize(res_layout.size());
            dofspan resp{resp_storage, res_layout};

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
            std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // get the unperturbed residual and send to full residual
            res = 0;
            disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);
            scatter_facspan(trace, 1.0, res, 1.0, mdg_residual);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

            // form local jacobian wrt uL
            for(IDX idofu = 0; idofu < uL.ndof(); ++idofu){
                for(IDX iequ = 0; iequ < uL.nv(); ++iequ){
                    IDX jcol = proc_range_beg + glob_index_L + uL.get_layout()[idofu, iequ];

                    // perturb 
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // get the perturbed residual
                    resp = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                    // fill jacobian for this perturbation 
                    for(IDX idoff = 0; idoff < res.ndof(); ++idoff) {

                        // only do perturbation if node is actually in nodeset 
                        IDX ignode = trace.face->nodes()[idoff];
                        IDX igdof = nodeset.inv_selected_nodes[ignode];

                        if(igdof != nodeset.selected_nodes.size()){
                            for(IDX ieqf = 0; ieqf < res.nv(); ++ieqf) {
                                // get the global row index based on the dof in node selection
                                IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }
                    }
                    // undo the perturbation
                    uL[idofu, iequ] = old_val;
                }
            }

            // form local jacobian wrt uR
            for(IDX idofu = 0; idofu < uR.ndof(); ++idofu){
                for(IDX iequ = 0; iequ < uR.nv(); ++iequ){
                    IDX jcol = proc_range_beg + glob_index_R + uR.get_layout()[idofu, iequ];

                    // perturb 
                    T old_val = uR[idofu, iequ];
                    uR[idofu, iequ] += eps_scaled;

                    // get the perturbed residual
                    resp = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                    // fill jacobian for this perturbation 
                    for(IDX idoff = 0; idoff < res.ndof(); ++idoff) {

                        // only do perturbation if node is actually in nodeset 
                        IDX ignode = trace.face->nodes()[idoff];
                        IDX igdof = nodeset.inv_selected_nodes[ignode];

                        if(igdof != nodeset.selected_nodes.size()){
                            for(IDX ieqf = 0; ieqf < res.nv(); ++ieqf) {
                                // get the global row index based on the dof in node selection
                                IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }
                    }
                    // undo the perturbation
                    uR[idofu, iequ] = old_val;
                }
            }
        }

        std::vector<T> resL_storage(fespace.dg_map.max_el_size_reqirement(u.nv()));
        std::vector<T> resLp_storage(fespace.dg_map.max_el_size_reqirement(u.nv()));
        std::vector<T> resR_storage(fespace.dg_map.max_el_size_reqirement(u.nv()));
        std::vector<T> resRp_storage(fespace.dg_map.max_el_size_reqirement(u.nv()));

        // Jacobian wrt x 
        for(IDX jmdg = 0; jmdg < mdg_residual.ndof(); ++jmdg){
            // the global node index corresponding to this mdg dof
            IDX inode = nodeset.selected_nodes[jmdg];

            // loop over element surrounding for domain integral
            for(IDX iel : fespace.el_surr_nodes.rowspan(inode)) {
                const Element& el = fespace.elements[iel];

                // set up compact data views
                auto el_layout = u.create_element_layout(iel);
                dofspan u_el{uL_storage, el_layout};
                dofspan res{resL_storage, u.create_element_layout(iel)};
                dofspan resp{resLp_storage, u.create_element_layout(iel)};

                // get the global index to the start of the contiguous component x dof range
                std::size_t glob_index_el = u.get_layout()[iel, 0, 0];

                // extract the compact values from the global u view
                extract_elspan(iel, u, u_el);

                // get the unperturbed residual
                res = 0;
                disc.domain_integral(el, fespace.meshptr->coord, u_el, res);

                // set up the perturbation amount scaled by unperturbed residual 
                T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

                // loop over dimensions of node and perturb
                for(int idim = 0; idim < ndim; ++idim){
                    
                        // the unknowns will always have ndim vector components 
                        IDX jcol = mdg_range_beg + jmdg * ndim + idim;

                        T old_val = fespace.meshptr->coord[inode][idim];
                        fespace.meshptr->coord[inode][idim] += eps_scaled;

                        // get perturbed residual
                        resp = 0;
                        disc.domain_integral(el, fespace.meshptr->coord, u_el, resp);

                        // jacobian contribution 
                        for(IDX idoff = 0; idoff < res.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < res.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_el + res.get_layout()[idoff, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // revert perturbation
                        fespace.meshptr->coord[inode][idim] = old_val;
                }
            }

            // build the extended stencil of face indices around the node
            // use a set to prevent repeats
            std::set<IDX>traces_to_visit{};
            for(IDX iel : fespace.el_surr_nodes.rowspan(inode)){
                for(IDX itrace : fespace.fac_surr_el.rowspan(iel)){
                    traces_to_visit.insert(itrace);
                }
            }

            // loop over traces in extended stencil around the node
            for(IDX itrace : traces_to_visit){
                const Trace& trace = fespace.traces[itrace];

                // set up compact data views
                auto uL_layout = u.create_element_layout(trace.elL.elidx);
                dofspan uL{uL_storage, uL_layout};
                auto uR_layout = u.create_element_layout(trace.elR.elidx);
                dofspan uR{uR_storage, uR_layout};

                // get the global index to the start of the contiguous component x dof range for L/R elem
                std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
                std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

                // extract the compact values from the global u view
                extract_elspan(trace.elL.elidx, u, uL);
                extract_elspan(trace.elR.elidx, u, uR);
                // du/dx  (L and R)
                {
                    dofspan resL{resL_storage, u.create_element_layout(trace.elL.elidx)};
                    dofspan resLp{resLp_storage, u.create_element_layout(trace.elL.elidx)};
                    dofspan resR{resR_storage, u.create_element_layout(trace.elR.elidx)};
                    dofspan resRp{resRp_storage, u.create_element_layout(trace.elR.elidx)};
                    resL = 0; resR = 0;

                    // get the unperturbed residual
                    if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                        disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR);
                    } else {
                        disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resL);
                    }

                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = scale_fd_epsilon(epsilon, std::max(resL.vector_norm(), resR.vector_norm()));

                    // loop over dimensions of node and perturb
                    for(int idim = 0; idim < ndim; ++idim) {
                        // the unknowns will always have ndim vector components 
                        IDX jcol = mdg_range_beg + jmdg * ndim + idim;

                        T old_val = fespace.meshptr->coord[inode][idim];
                        fespace.meshptr->coord[inode][idim] += eps_scaled;

                        // get the perturbed residual
                        resLp = 0; resRp = 0;
                        if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                            disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resLp, resRp);
                        } else {
                            disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resLp);
                        }

                        // resL
                        for(IDX idoff = 0; idoff < resL.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < resL.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_L + resL.get_layout()[idoff, ieqf];
                                T fd_val = (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // resR (only for interior traces)
                        if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                            for(IDX idoff = 0; idoff < resR.ndof(); ++idoff){
                                for(IDX ieqf = 0; ieqf < resR.nv(); ++ieqf){
                                    IDX irow = proc_range_beg + glob_index_R + resR.get_layout()[idoff, ieqf];
                                    T fd_val = (resRp[idoff, ieqf] - resR[idoff, ieqf]) / eps_scaled;
                                    MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                                }
                            }
                        }

                        // revert perturbation
                        fespace.meshptr->coord[inode][idim] = old_val;
                    }
                }

                // dICE/dx
                {
                    auto res_layout = trace_layout_right{trace};
                    res_storage.resize(res_layout.size());
                    dofspan res{res_storage, res_layout};
                    resp_storage.resize(res_layout.size());
                    dofspan resp{resp_storage, res_layout};

                    // get the unperturbed residual
                    res = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);
                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

                    // loop over dimensions of node and perturb
                    for(int idim = 0; idim < ndim; ++idim) {
                        // the unknowns will always have ndim vector components 
                        IDX jcol = mdg_range_beg + jmdg * ndim + idim;

                        // peturb the node
                        T old_val = fespace.meshptr->coord[inode][idim];
                        fespace.meshptr->coord[inode][idim] += eps_scaled;

                        // get the perturbed residual
                        resp = 0;
                        disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                        for(IDX idoff = 0; idoff < res.ndof(); ++idoff){

                            // only scatter if node is actually in nodeset 
                            IDX ignode = trace.face->nodes()[idoff];
                            IDX igdof = nodeset.inv_selected_nodes[ignode];
                            
                            if(igdof != nodeset.selected_nodes.size()){
                                for(IDX ieqf = 0; ieqf < res.nv(); ++ieqf){
                                    // get the global row index based on the dof in node selection
                                    IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                    T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                    MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                                }
                            }
                        }

                        // revert perturbation
                        fespace.meshptr->coord[inode][idim] = old_val;
                    }
                }
            }

        }
    }

    /// @brief from the jacobian terms due to the interface condition enforcement 
    /// NOTE: this works with the non-square matrix
template<
        class T, class IDX, int ndim,
        class disc_class, class uLayoutPolicy, class uAccessorPolicy
    >
    auto form_petsc_mdg_jacobian_fd(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        geospan auto x,
        icespan auto mdg_residual,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()),
        MPI_Comm comm = MPI_COMM_WORLD 
    ) -> void 
    {
        using Element = FiniteElement<T, IDX, ndim>;
        using Trace = TraceSpace<T, IDX, ndim>;
        using index_type = IDX;
        using namespace std::experimental;

        static constexpr int neq = disc_class::nv_comp;

        // zero out the residual 
        mdg_residual = 0;

        // get the start indices for the petsc matrix on this processor
        PetscInt proc_range_beg, proc_range_end, mdg_range_beg;
        PetscCallAbort(comm, MatGetOwnershipRange(jac, &proc_range_beg, &proc_range_end));
        mdg_range_beg = proc_range_beg + u.size();

        // preallocate storage for compact views of u 
        const std::size_t max_local_size =
            fespace.dofs.max_el_size_reqirement(neq);
        std::vector<T> uL_storage(max_local_size);
        std::vector<T> uR_storage(max_local_size);
        std::vector<T> jacL_storage{};
        std::vector<T> jacR_storage{};
        std::vector<T> res_storage{};
        std::vector<T> resp_storage{};

        // get the selected geometry to apply interface condition to
        const geo_dof_map<T, IDX, ndim>& geo_map = x.get_layout().geo_map;

        // apply the x coordinates to the mesh
        update_mesh(x, *(fespace.meshptr));

        // jacobian of ice residual wrt u
        for(index_type itrace: geo_map.selected_traces){
            Trace& trace = fespace.traces[itrace];

            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            dofspan uL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            dofspan uR{uR_storage, uR_layout};

            trace_layout_right res_layout{trace, disc};
            res_storage.resize(res_layout.size());
            dofspan res{res_storage, res_layout};
            resp_storage.resize(res_layout.size());
            dofspan resp{resp_storage, res_layout};

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
            std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

            // extract the compact values from the global u view
            extract_elspan(trace.elL.elidx, u, uL);
            extract_elspan(trace.elR.elidx, u, uR);

            // get the unperturbed residual and send to full residual
            res = 0;
            disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);
            scatter_facspan(trace, 1.0, res, 1.0, mdg_residual);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

            // dICE/duL
            for(IDX idofu = 0; idofu < uL.ndof(); ++idofu){
                for(IDX iequ = 0; iequ < uL.nv(); ++iequ){
                    IDX jcol = proc_range_beg + glob_index_L + uL.get_layout()[idofu, iequ];

                    // perturb 
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // get the perturbed residual
                    resp = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                    // fill jacobian for this perturbation 
                    for(IDX idoff = 0; idoff < res.ndof(); ++idoff) {

                        // only do perturbation if node is actually in nodeset 
                        IDX ignode = trace.face->nodes()[idoff];
                        IDX igdof = geo_map.inv_selected_nodes[ignode];

                        if(igdof != geo_map.selected_nodes.size()){
                            for(IDX ieqf = 0; ieqf < neq; ++ieqf) {
                                // get the global row index based on the dof in node selection
                                IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }
                    }
                    // undo the perturbation
                    uL[idofu, iequ] = old_val;
                }
            }

            // dICE/duR
            for(IDX idofu = 0; idofu < uR.ndof(); ++idofu){
                for(IDX iequ = 0; iequ < uR.nv(); ++iequ){
                    IDX jcol = proc_range_beg + glob_index_R + uR.get_layout()[idofu, iequ];

                    // perturb 
                    T old_val = uR[idofu, iequ];
                    uR[idofu, iequ] += eps_scaled;

                    // get the perturbed residual
                    resp = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                    // fill jacobian for this perturbation 
                    for(IDX idoff = 0; idoff < res.ndof(); ++idoff) {

                        // only do perturbation if node is actually in nodeset 
                        IDX ignode = trace.face->nodes()[idoff];
                        IDX igdof = geo_map.inv_selected_nodes[ignode];

                        if(igdof != geo_map.selected_nodes.size()){
                            for(IDX ieqf = 0; ieqf < neq; ++ieqf) {
                                // get the global row index based on the dof in node selection
                                IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }
                    }
                    // undo the perturbation
                    uR[idofu, iequ] = old_val;
                }
            }
        }

        std::vector<T> resL_storage(fespace.dofs.max_el_size_reqirement(u.nv()));
        std::vector<T> resLp_storage(fespace.dofs.max_el_size_reqirement(u.nv()));
        std::vector<T> resR_storage(fespace.dofs.max_el_size_reqirement(u.nv()));
        std::vector<T> resRp_storage(fespace.dofs.max_el_size_reqirement(u.nv()));

        // Jacobian wrt x 
        for(IDX jmdg = 0; jmdg < mdg_residual.ndof(); ++jmdg){
            // the global node index corresponding to this mdg dof
            IDX inode = geo_map.selected_nodes[jmdg];

            // loop over element surrounding for domain integral
            for(IDX iel : fespace.el_surr_nodes.rowspan(inode)) {
                const Element& el = fespace.elements[iel];

                // set up compact data views
                auto el_layout = u.create_element_layout(iel);
                dofspan u_el{uL_storage, el_layout};
                dofspan res{resL_storage, u.create_element_layout(iel)};
                dofspan resp{resLp_storage, u.create_element_layout(iel)};

                // get the global index to the start of the contiguous component x dof range
                std::size_t glob_index_el = u.get_layout()[iel, 0, 0];

                // extract the compact values from the global u view
                extract_elspan(iel, u, u_el);

                // get the unperturbed residual
                res = 0;
                disc.domain_integral(el, u_el, res);

                // set up the perturbation amount scaled by unperturbed residual 
                T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

                // get the original parameterization 
                auto* parametrization = geo_map.parametric_accessors[jmdg].get();
                std::vector<T> s(parametrization->s_size());
                parametrization->x_to_s(fespace.meshptr->coord[inode], s);

                // loop over dimension of the geometric parameterization and peturb
                for(int is = 0; is < x.nv(jmdg); ++is){
                    
                        IDX jcol = mdg_range_beg + geo_map.cols[jmdg] + is;

                        T old_val = s[is];
                        s[is] += eps_scaled;
                        // TODO: maybe find a way to use the local el.coord_el to do this 
                        // instead of updating the global data structure
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);

                        // get perturbed residual
                        resp = 0;
                        disc.domain_integral(el, u_el, resp);

                        // jacobian contribution 
                        for(IDX idoff = 0; idoff < res.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < res.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_el + res.get_layout()[idoff, ieqf];
                                T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // revert perturbation
                        s[is] = old_val;
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);
                }
            }

            // build the extended stencil of face indices around the node
            // use a set to prevent repeats
            std::set<IDX>traces_to_visit{};
            for(IDX iel : fespace.el_surr_nodes.rowspan(inode)){
                for(IDX itrace : fespace.meshptr->facsuel.rowspan(iel)){
                    traces_to_visit.insert(itrace);
                }
            }

            // loop over traces in extended stencil around the node
            for(IDX itrace : traces_to_visit){
                const Trace& trace = fespace.traces[itrace];

                // set up compact data views
                auto uL_layout = u.create_element_layout(trace.elL.elidx);
                dofspan uL{uL_storage, uL_layout};
                auto uR_layout = u.create_element_layout(trace.elR.elidx);
                dofspan uR{uR_storage, uR_layout};

                // get the global index to the start of the contiguous component x dof range for L/R elem
                std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
                std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

                // extract the compact values from the global u view
                extract_elspan(trace.elL.elidx, u, uL);
                extract_elspan(trace.elR.elidx, u, uR);
                // du/dx  (L and R)
                {
                    dofspan resL{resL_storage, u.create_element_layout(trace.elL.elidx)};
                    dofspan resLp{resLp_storage, u.create_element_layout(trace.elL.elidx)};
                    dofspan resR{resR_storage, u.create_element_layout(trace.elR.elidx)};
                    dofspan resRp{resRp_storage, u.create_element_layout(trace.elR.elidx)};
                    resL = 0; resR = 0;

                    // get the unperturbed residual
                    if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                        disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR);
                    } else {
                        disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resL);
                    }

                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = scale_fd_epsilon(epsilon, std::max(resL.vector_norm(), resR.vector_norm()));

                    // get the original parameterization 
                    auto* parametrization = geo_map.parametric_accessors[jmdg].get();
                    std::vector<T> s(parametrization->s_size());
                    parametrization->x_to_s(fespace.meshptr->coord[inode], s);

                    // loop over dimension of the geometric parameterization and peturb
                    for(int is = 0; is < x.nv(jmdg); ++is){
                        IDX jcol = mdg_range_beg + geo_map.cols[jmdg] + is;

                        T old_val = s[is];
                        s[is] += eps_scaled;
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);

                        // get the perturbed residual
                        resLp = 0; resRp = 0;
                        if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                            disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resLp, resRp);
                        } else {
                            disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resLp);
                        }

                        // resL
                        for(IDX idoff = 0; idoff < resL.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < resL.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_L + resL.get_layout()[idoff, ieqf];
                                T fd_val = (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // resR (only for interior traces)
                        if(trace.face->bctype == BOUNDARY_CONDITIONS::INTERIOR){
                            for(IDX idoff = 0; idoff < resR.ndof(); ++idoff){
                                for(IDX ieqf = 0; ieqf < resR.nv(); ++ieqf){
                                    IDX irow = proc_range_beg + glob_index_R + resR.get_layout()[idoff, ieqf];
                                    T fd_val = (resRp[idoff, ieqf] - resR[idoff, ieqf]) / eps_scaled;
                                    MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                                }
                            }
                        }

                        // revert perturbation
                        s[is] = old_val;
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);
                    }
                }
            }

            // loop over the traces selected for interface conservation 
            for(IDX itrace : geo_map.selected_traces) {
                
                const Trace& trace = fespace.traces[itrace];

                // set up compact data views
                auto uL_layout = u.create_element_layout(trace.elL.elidx);
                dofspan uL{uL_storage, uL_layout};
                auto uR_layout = u.create_element_layout(trace.elR.elidx);
                dofspan uR{uR_storage, uR_layout};

                // get the global index to the start of the contiguous component x dof range for L/R elem
                std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
                std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

                // extract the compact values from the global u view
                extract_elspan(trace.elL.elidx, u, uL);
                extract_elspan(trace.elR.elidx, u, uR);

                // dICE/dx
                {
                    auto res_layout = trace_layout_right{trace, disc};
                    res_storage.resize(res_layout.size());
                    dofspan res{res_storage, res_layout};
                    resp_storage.resize(res_layout.size());
                    dofspan resp{resp_storage, res_layout};

                    // get the unperturbed residual
                    res = 0;
                    disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, res);
                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = scale_fd_epsilon(epsilon, res.vector_norm());

                    // get the original parameterization 
                    auto* parametrization = geo_map.parametric_accessors[jmdg].get();
                    std::vector<T> s(parametrization->s_size());
                    parametrization->x_to_s(fespace.meshptr->coord[inode], s);

                    // loop over dimension of the geometric parameterization and peturb
                    for(int is = 0; is < x.nv(jmdg); ++is){

                        IDX jcol = mdg_range_beg + geo_map.cols[jmdg] + is;

                        T old_val = s[is];
                        s[is] += eps_scaled;
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);

                        // get the perturbed residual
                        resp = 0;
                        disc.interface_conservation(trace, fespace.meshptr->coord, uL, uR, resp);

                        for(IDX idoff = 0; idoff < res.ndof(); ++idoff){

                            // only scatter if node is actually in nodeset 
                            IDX ignode = trace.face->nodes()[idoff];
                            IDX igdof = geo_map.inv_selected_nodes[ignode];
                            
                            if(igdof != geo_map.selected_nodes.size()){
                                for(IDX ieqf = 0; ieqf < neq; ++ieqf){
                                    // get the global row index based on the dof in node selection
                                    IDX irow = mdg_range_beg + mdg_residual.get_layout()[igdof, ieqf];
                                    T fd_val = (resp[idoff, ieqf] - res[idoff, ieqf]) / eps_scaled;
                                    MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                                }
                            }
                        }

                        // revert perturbation
                        s[is] = old_val;
                        parametrization->s_to_x(s, fespace.meshptr->coord[inode]);
                        fespace.meshptr->update_node(inode);
                    }
                }
            }

        }
    }

    template<
        class T, class IDX, int ndim,
        class disc_class, class uLayoutPolicy, class uAccessorPolicy
    >
    auto form_petsc_jacobian_dense_fd(
        FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        geospan auto x,
        Vec res,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon())
    ) -> void 
    {

        petsc::VecSpan res_span{res};
        std::vector<T> resp_data(res_span.size());

        // get the selected geometry to apply interface condition to
        const geo_dof_map<T, IDX, ndim>& geo_map = x.get_layout().geo_map;

        // copy the input data
        std::vector<T> ufull(u.size() + x.size());
        std::copy_n(u.data(), u.size(), ufull.data());
        std::copy_n(x.data(), x.size(), ufull.data() + u.size());

        form_residual(fespace, disc, geo_map, std::span{ufull}, std::span{res_span});

        for(IDX jdof = 0; jdof < ufull.size(); ++jdof) {
            T uold = ufull[jdof];
            ufull[jdof] += epsilon;
            form_residual(fespace, disc, geo_map, std::span{ufull}, std::span{resp_data});

            for(IDX idof = 0; idof < res_span.size(); ++idof){
                T fd_val = (resp_data[idof] - res_span[idof]) / epsilon;
                if(std::abs(fd_val) > 1e-10)
                    MatSetValue(jac, idof, jdof, fd_val, ADD_VALUES);
            }
            // revert
            ufull[jdof] = uold;
        }
    }
}
