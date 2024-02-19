/**
 * @brief form the jacobian as a petsc matrix
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "iceicle/fe_function/layout_enums.hpp"
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
            fespace.dg_offsets.max_el_size_reqirement(disc_class::dnv_comp);
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
            FE::elspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            FE::elspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            FE::elspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::elspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};

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

            std::size_t glob_index_L = u.get_layout()(FE::fe_index{(std::size_t) trace.elL.elidx, 0, 0});

            // perturb and form jacobian 
            for(std::size_t idofu = 0; idofu < trace.elL.nbasis(); ++idofu){
                for(std::size_t iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    std::size_t jcol = uL.get_layout()(FE::compact_index{idofu, iequ});

                    // perturb
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resLp = 0;

                    // get the perturbed residual
                    disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp);

                    // fill jacobian for this perturbation
                    for(std::size_t idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = uL.get_layout()(FE::compact_index{idoff, ieqf});
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
            FE::elspan uL{uL_data.data(), u.create_element_layout(trace.elL.elidx)};
            FE::elspan uR{uR_data.data(), u.create_element_layout(trace.elR.elidx)};

            // compact residual views
            FE::elspan resL{resL_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::elspan resLp{resLp_data.data(), res.create_element_layout(trace.elL.elidx)};
            FE::elspan resR{resR_data.data(), res.create_element_layout(trace.elR.elidx)};
            FE::elspan resRp{resRp_data.data(), res.create_element_layout(trace.elR.elidx)};


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
            std::size_t glob_index_L = u.get_layout()(FE::fe_index{(std::size_t) trace.elL.elidx, 0, 0});
            std::size_t glob_index_R = u.get_layout()(FE::fe_index{(std::size_t) trace.elR.elidx, 0, 0});

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, std::max(resL.vector_norm(), resR.vector_norm()) * epsilon);

            // perturb and form jacobian wrt uL
            // compact jacobian views 
            mdspan jacL{jacL_data.data(), extents{resL.size(), uL.size()}};
            mdspan jacR{jacR_data.data(), extents{resR.size(), uL.size()}};
            std::fill_n(jacL_data.begin(), jacL.size(), 0);
            std::fill_n(jacR_data.begin(), jacR.size(), 0);
            for(std::size_t idofu = 0; idofu < trace.elL.nbasis(); ++idofu){
                for(std::size_t iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    std::size_t jcol = uL.get_layout()(FE::compact_index{idofu, iequ});

                    // perturb
                    T old_val = uL[idofu, iequ];
                    uL[idofu, iequ] += eps_scaled;

                    // zero out the residuals
                    resLp = 0;
                    resRp = 0;

                    // get the perturbed residual
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

                    // fill jacobian for this perturbation
                    for(std::size_t idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = uL.get_layout()(FE::compact_index{idoff, ieqf});
                            jacL[irow, jcol] += (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                        }
                    }
                    for(std::size_t idoff = 0; idoff < trace.elR.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = uR.get_layout()(FE::compact_index{idoff, ieqf});
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
            for(std::size_t idofu = 0; idofu < trace.elR.nbasis(); ++idofu){
                for(std::size_t iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    std::size_t jcol = uR.get_layout()(FE::compact_index{idofu, iequ});

                    // perturb
                    T old_val = uR[idofu, iequ];
                    uR[idofu, iequ] += eps_scaled;

                    // zero out the residuals
                    resLp = 0;
                    resRp = 0;

                    // get the perturbed residual
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

                    // fill jacobian for this perturbation
                    for(std::size_t idoff = 0; idoff < trace.elL.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = uL.get_layout()(FE::compact_index{idoff, ieqf});
                            jacL[irow, jcol] += (resLp[idoff, ieqf] - resL[idoff, ieqf]) / eps_scaled;
                        }
                    }
                    for(std::size_t idoff = 0; idoff < trace.elR.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = uR.get_layout()(FE::compact_index{idoff, ieqf});
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
            FE::elspan u_el{uL_data.data(), u.create_element_layout(el.elidx)};

            // residual data views 
            FE::elspan res_el{resL_data.data(), res.create_element_layout(el.elidx)};
            FE::elspan resp_el{resLp_data.data(), res.create_element_layout(el.elidx)};

            // jacobian data view
            mdspan jac_el{jacL_data.data(), extents{res_el.size(), u_el.size()}};
            std::fill_n(jacL_data.begin(), jac_el.size(), 0);

            res_el = 0;
            disc.domainIntegral(el, fespace.meshptr->nodes, u_el, res_el);

            // get the global index to the start of the contiguous component x dof range for L/R elem
            std::size_t glob_index_el = u.get_layout()(FE::fe_index{(std::size_t) el.elidx, 0, 0});

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, res_el.vector_norm() * epsilon);

            // perturb and form jacobian wrt u_el
            for(std::size_t idofu = 0; idofu < el.nbasis(); ++idofu){
                for(std::size_t iequ = 0; iequ < ncomp; ++iequ){
                    // get the compact column index for this dof and component 
                    std::size_t jcol = u_el.get_layout()(FE::compact_index{idofu, iequ});

                    // perturb
                    T old_val = u_el[idofu, iequ];
                    u_el[idofu, iequ] += eps_scaled;

                    // zero out the residual 
                    resp_el = 0;

                    // get the perturbed residual
                    disc.domainIntegral(el, fespace.meshptr->nodes, u_el, resp_el);

                    // fill jacobian for this perturbation
                    for(std::size_t idoff = 0; idoff < el.nbasis(); ++idoff) {
                        for(std::size_t ieqf = 0; ieqf < ncomp; ++ieqf){
                            std::size_t irow = u_el.get_layout()(FE::compact_index{idoff, ieqf});
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
}
