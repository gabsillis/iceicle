/**
 * @brief form the jacobian as a petsc matrix
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/petsc_interface.hpp"
#include <cmath>
#include <limits>
#include <petscerror.h>
#include <petscmat.h>
#include <mdspan/mdspan.hpp>

namespace ICEICLE::SOLVERS {

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
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        FE::fespan<T, resLayoutPolicy> res,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()),
        MPI_Comm comm = MPI_COMM_WORLD 
    ) {

        using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Trace = ELEMENT::TraceSpace<T, IDX, ndim>;

        using namespace std::experimental;

        // zero out the residual
        res = 0;

        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dg_map.max_el_size_reqirement(disc_class::dnv_comp);
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
            FE::dofspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            FE::dofspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            FE::dofspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::dofspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};

            // compact jacobian views 
            mdspan jacL{jacL_data.data(), extents{resL.size(), uL.size()}};
            std::fill_n(jacL_data.begin(), jacL.size(), 0);

            // extract the compact values from the global u view 
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residuals 
            resL = 0;

            // get the unperturbed residual and send to full residual
            disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resL);
            FE::scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, resL.vector_norm() * epsilon);

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
                    disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp);

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
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_L, jacL);
        }

        // interior faces 
        for(const Trace &trace : fespace.get_interior_traces()) {
            // compact data views 
            FE::dofspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            FE::dofspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            FE::dofspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::dofspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::dofspan resR{resR_data.data(), res.create_element_layout(trace.elR.elidx)};
            FE::dofspan resRp{resRp_data.data(), res.create_element_layout(trace.elR.elidx)};


            // extract the compact values from the global u view 
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residuals 
            resL = 0;
            resR = 0;

            // get the unperturbed residual and send to full residual
            disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resL, resR);
            FE::scatter_elspan(trace.elL.elidx, 1.0, resL, 1.0, res);
            FE::scatter_elspan(trace.elR.elidx, 1.0, resR, 1.0, res);

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
            std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, std::max(resL.vector_norm(), resR.vector_norm()) * epsilon);

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
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

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
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_L, jacL);
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_R, 
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
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

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
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_L, 
                    proc_range_beg + glob_index_R, jacL);
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_R, 
                    proc_range_beg + glob_index_R, jacR);
        }

        // domain integral 
        for(const Element &el : fespace.elements) {
            // compact data views 
            FE::dofspan u_el{uL_data.data(), u.create_element_layout(el.elidx)};

            // residual data views 
            FE::dofspan res_el{resL_data.data(), res.create_element_layout(el.elidx)};
            FE::dofspan resp_el{resLp_data.data(), res.create_element_layout(el.elidx)};

            // jacobian data view
            mdspan jac_el{jacL_data.data(), extents{res_el.size(), u_el.size()}};
            std::fill_n(jacL_data.begin(), jac_el.size(), 0);

            // extract the compact values from the global u view 
            FE::extract_elspan(el.elidx, u, u_el);

            res_el = 0;
            disc.domainIntegral(el, fespace.meshptr->nodes, u_el, res_el);

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_el = u.get_layout()[el.elidx, 0, 0];
            // send residual to global residual 
            FE::scatter_elspan(el.elidx, 1.0, res_el, 1.0, res);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, res_el.vector_norm() * epsilon);

            // perturb and form jacobian wrt u_el
            for(IDX idofu = 0; idofu < el.nbasis(); ++idofu){
                for(IDX iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    IDX jcol = u_el.get_layout()[idofu, iequ];

                    // perturb
                    T old_val = u_el[idofu, iequ];
                    u_el[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resp_el = 0;

                    // get the perturbed residual
                    disc.domainIntegral(el, fespace.meshptr->nodes, u_el, resp_el);

                    // fill jacobian for this perturbation
                    for(IDX idoff = 0; idoff < el.nbasis(); ++idoff) {
                        for(IDX ieqf = 0; ieqf < ncomp; ++ieqf){
                            IDX irow = u_el.get_layout()[idoff, ieqf];
                            jac_el[irow, jcol] += (resp_el[idoff, ieqf] - res_el[idoff, ieqf]) / eps_scaled;
                        }
                    }

                    // undo the perturbation
                    u_el[idofu, iequ] = old_val;
                }
            }

            // TODO: change to a scatter generalized operation to support CG structures
            ICEICLE::PETSC::add_to_petsc_mat(jac, proc_range_beg + glob_index_el, 
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
        FE::FESpace<T, IDX, ndim>& fespace,
        disc_class& disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        FE::node_selection_span auto mdg_residual,
        Mat jac,
        T epsilon = std::sqrt(std::numeric_limits<T>::epsilon()),
        MPI_Comm comm = MPI_COMM_WORLD 
    ) -> void {
        using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Trace = ELEMENT::TraceSpace<T, IDX, ndim>;
        using index_type = IDX;
        using namespace FE;
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
            FE::dofspan uL{uL_storage, uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            FE::dofspan uR{uR_storage, uR_layout};

            trace_layout_right<IDX, decltype(mdg_residual)::static_extent()> res_layout{trace};
            res_storage.resize(res_layout.size());
            dofspan res{res_storage, res_layout};
            resp_storage.resize(res_layout.size());
            dofspan resp{resp_storage, res_layout};

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
            std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

            // extract the compact values from the global u view
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // get the unperturbed residual and send to full residual
            res = 0;
            disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, res);

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, res.vector_norm() * epsilon);

            // form local jacobian wrt uL
            for(IDX idofu = 0; idofu < uL.ndof(); ++idofu){
                for(IDX iequ = 0; iequ < uL.nv(); ++iequ){
                    IDX jcol = proc_range_beg + glob_index_L + uL.get_layout()[idofu, iequ];

                    // perturb 
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // get the perturbed residual
                    resp = 0;
                    disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, resp);

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
                    disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, resp);

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

        // TODO: Jacobian wrt x 
        for(IDX jmdg = 0; jmdg < mdg_residual.ndof(); ++jmdg){
            // the global node index corresponding to this mdg dof
            IDX inode = nodeset.selected_nodes[jmdg];
            for(IDX itrace : fespace.fac_surr_nodes.rowspan(inode)){
                const Trace& trace = fespace.traces[itrace];

                // set up compact data views
                auto uL_layout = u.create_element_layout(trace.elL.elidx);
                FE::dofspan uL{uL_storage, uL_layout};
                auto uR_layout = u.create_element_layout(trace.elR.elidx);
                FE::dofspan uR{uR_storage, uR_layout};

                // get the global index to the start of the contiguous component x dof range for L/R elem
                std::size_t glob_index_L = u.get_layout()[trace.elL.elidx, 0, 0];
                std::size_t glob_index_R = u.get_layout()[trace.elR.elidx, 0, 0];

                // extract the compact values from the global u view
                FE::extract_elspan(trace.elL.elidx, u, uL);
                FE::extract_elspan(trace.elR.elidx, u, uR);
                // du/dx  (L and R)
                {
                    FE::dofspan resL{resL_storage, u.create_element_layout(trace.elL.elidx)};
                    FE::dofspan resLp{resLp_storage, u.create_element_layout(trace.elL.elidx)};
                    FE::dofspan resR{resR_storage, u.create_element_layout(trace.elR.elidx)};
                    FE::dofspan resRp{resRp_storage, u.create_element_layout(trace.elR.elidx)};

                    // get the unperturbed residual
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resL, resR);

                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = std::max(epsilon, std::max(resL.vector_norm(), resR.vector_norm()) * epsilon);

                    // loop over dimensions of node and perturb
                    for(int idim = 0; idim < ndim; ++idim) {
                        // the unknowns will always have ndim vector components 
                        IDX jcol = jmdg * ndim + idim;

                        T old_val = fespace.meshptr->nodes[inode][idim];
                        fespace.meshptr->nodes[inode][idim] += eps_scaled;

                        // get the perturbed residual
                        disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

                        // resL
                        for(IDX idoff = 0; idoff < resL.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < resL.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_L + resL.get_layout()[idoff, ieqf];
                                T fd_val = (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // resR
                        for(IDX idoff = 0; idoff < resR.ndof(); ++idoff){
                            for(IDX ieqf = 0; ieqf < resR.nv(); ++ieqf){
                                IDX irow = proc_range_beg + glob_index_R + resR.get_layout()[idoff, ieqf];
                                T fd_val = (resRp[idoff, ieqf] - resR[idoff, ieqf]) / eps_scaled;
                                MatSetValue(jac, irow, jcol, fd_val, ADD_VALUES);
                            }
                        }

                        // revert perturbation
                        fespace.meshptr->nodes[inode][idim] = old_val;
                    }
                }

                // dICE/dx
                {
                    auto res_layout = trace_layout_right{trace};
                    res_storage.resize(res_layout.size());
                    FE::dofspan res{res_storage, res_layout};
                    resp_storage.resize(res_layout.size());
                    FE::dofspan resp{resp_storage, res_layout};

                    // get the unperturbed residual
                    disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, res);
                    // set up the perturbation amount scaled by unperturbed residual 
                    T eps_scaled = std::max(epsilon, res.vector_norm() * epsilon);

                    // loop over dimensions of node and perturb
                    for(int idim = 0; idim < ndim; ++idim) {
                        // the unknowns will always have ndim vector components 
                        IDX jcol = jmdg * ndim + idim;

                        // peturb the node
                        T old_val = fespace.meshptr->nodes[inode][idim];
                        fespace.meshptr->nodes[inode][idim] += eps_scaled;

                        // get the perturbed residual
                        disc.interface_conservation(trace, fespace.meshptr->nodes, uL, uR, res);

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
                        fespace.meshptr->nodes[inode][idim] = old_val;
                    }
                }
            }
        }
    }
}
