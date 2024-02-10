/**
 * @file form_hypre_jacobian.hpp
 * Form the jacobian for nonlinear solvers as a hypre matrix 
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include "HYPRE_IJ_mv.h"
#include "iceicle/fe_function/layout_enums.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/form_residual.hpp"
#include "HYPRE.h"
#include <limits>
#include <mpi.h>
#include <complex>

namespace ICEICLE::SOLVERS {

    // TODO: pull out to a more general file and clean up interface 
    // https://stackoverflow.com/questions/42490331/generic-mpi-code
    template <typename T>
[[nodiscard]] constexpr MPI_Datatype mpi_get_type() noexcept
{
    MPI_Datatype mpi_type = MPI_DATATYPE_NULL;
    
    if constexpr (std::is_same<T, char>::value)
    {
        mpi_type = MPI_CHAR;
    }
    else if constexpr (std::is_same<T, signed char>::value)
    {
        mpi_type = MPI_SIGNED_CHAR;
    }
    else if constexpr (std::is_same<T, unsigned char>::value)
    {
        mpi_type = MPI_UNSIGNED_CHAR;
    }
    else if constexpr (std::is_same<T, wchar_t>::value)
    {
        mpi_type = MPI_WCHAR;
    }
    else if constexpr (std::is_same<T, signed short>::value)
    {
        mpi_type = MPI_SHORT;
    }
    else if constexpr (std::is_same<T, unsigned short>::value)
    {
        mpi_type = MPI_UNSIGNED_SHORT;
    }
    else if constexpr (std::is_same<T, signed int>::value)
    {
        mpi_type = MPI_INT;
    }
    else if constexpr (std::is_same<T, unsigned int>::value)
    {
        mpi_type = MPI_UNSIGNED;
    }
    else if constexpr (std::is_same<T, signed long int>::value)
    {
        mpi_type = MPI_LONG;
    }
    else if constexpr (std::is_same<T, unsigned long int>::value)
    {
        mpi_type = MPI_UNSIGNED_LONG;
    }
    else if constexpr (std::is_same<T, signed long long int>::value)
    {
        mpi_type = MPI_LONG_LONG;
    }
    else if constexpr (std::is_same<T, unsigned long long int>::value)
    {
        mpi_type = MPI_UNSIGNED_LONG_LONG;
    }
    else if constexpr (std::is_same<T, float>::value)
    {
        mpi_type = MPI_FLOAT;
    }
    else if constexpr (std::is_same<T, double>::value)
    {
        mpi_type = MPI_DOUBLE;
    }
    else if constexpr (std::is_same<T, long double>::value)
    {
        mpi_type = MPI_LONG_DOUBLE;
    }
    else if constexpr (std::is_same<T, std::int8_t>::value)
    {
        mpi_type = MPI_INT8_T;
    }
    else if constexpr (std::is_same<T, std::int16_t>::value)
    {
        mpi_type = MPI_INT16_T;
    }
    else if constexpr (std::is_same<T, std::int32_t>::value)
    {
        mpi_type = MPI_INT32_T;
    }
    else if constexpr (std::is_same<T, std::int64_t>::value)
    {
        mpi_type = MPI_INT64_T;
    }
    else if constexpr (std::is_same<T, std::uint8_t>::value)
    {
        mpi_type = MPI_UINT8_T;
    }
    else if constexpr (std::is_same<T, std::uint16_t>::value)
    {
        mpi_type = MPI_UINT16_T;
    }
    else if constexpr (std::is_same<T, std::uint32_t>::value)
    {
        mpi_type = MPI_UINT32_T;
    }
    else if constexpr (std::is_same<T, std::uint64_t>::value)
    {
        mpi_type = MPI_UINT64_T;
    }
    else if constexpr (std::is_same<T, bool>::value)
    {
        mpi_type = MPI_C_BOOL;
    }
    else if constexpr (std::is_same<T, std::complex<float>>::value)
    {
        mpi_type = MPI_C_COMPLEX;
    }
    else if constexpr (std::is_same<T, std::complex<double>>::value)
    {
        mpi_type = MPI_C_DOUBLE_COMPLEX;
    }
    else if constexpr (std::is_same<T, std::complex<long double>>::value)
    {
        mpi_type = MPI_C_LONG_DOUBLE_COMPLEX;
    }
    
    assert(mpi_type != MPI_DATATYPE_NULL);
    return mpi_type;    
}
    

    /**
     * @brief form a hypre sparse matrix that represents the jacobian 
     * The jacobian is J_{ij} = \frac{\partial f_i}{\partial u_j}
     * 
     * WARNING: this has no way to track the storage duration of the matrix 
     * HYPRE_IJMatrixDestory() must be called to free memory
     *
     * @tparam T the floating point type 
     * @tparam IDX the index type 
     * @tparam ndim the number of dimension
     * @tparam disc_class the discretization type to get the jacobian for 
     * @tparam uLayoutPolicy the layout policy for the solution vector 
     * @tparam uAccessorPolicy the accessor policy for the solution vector 
     *
     * @param fespace the finite element space 
     * @param disc the discretization to get the jacobian for 
     * @param u the solution to evaluate the jacobian at 
     * @param comm (optional) the mpi commmunicator (defaults to world)
     * @return the HYPRE_IJMatrix handle of the jacobian 
     */
    template<
        class T,
        class IDX,
        int ndim,
        class disc_class,
        class uLayoutPolicy,
        class uAccessorPolicy
    >
    HYPRE_IJMatrix form_hypre_jacobian(
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_class &disc,
        FE::fespan<T, uLayoutPolicy, uAccessorPolicy> u,
        MPI_Comm comm = MPI_COMM_WORLD
    ) requires specifies_ncomp<disc_class>
    {
        static constexpr T epsilon = std::sqrt(std::numeric_limits<T>::epsilon());

        // using 
        using namespace std::experimental;

        // aliases
        using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
        using Trace = ELEMENT::TraceSpace<T, IDX, ndim>;

        // MPI info
        int nproc;
        int proc_id;
        MPI_Comm_size(comm, &nproc);
        MPI_Comm_rank(comm, &proc_id);

        // get the number of vector components from the discretization
        const std::size_t ncomp = disc_class::dnv_comp;

        // preallocate storage for compact views of u and res 
        const std::size_t max_local_size =
            fespace.dg_offsets.max_el_size_reqirement(ncomp);
        std::vector<T> uL_data(max_local_size);
        std::vector<T> uR_data(max_local_size);
        std::vector<T> resL_data(max_local_size);
        std::vector<T> resR_data(max_local_size);
        std::vector<T> resLp_data(max_local_size);
        std::vector<T> resRp_data(max_local_size);

        // === determine the range of rows ===
        // NOTE: hypre uses inclusive ranges

        // First communicate the size ndof * ncomp for each process 
        const std::size_t process_size = fespace.dg_offsets.calculate_size_requirement(ncomp);
        std::vector<std::size_t> proc_sizes(nproc);
        proc_sizes[proc_id] = process_size;
        for(int iproc = 0; iproc < nproc; ++iproc){
            MPI_Bcast(&(proc_sizes[iproc]), 1, mpi_get_type<std::size_t>(), iproc, comm);
        }

        // add the sizes for each processor until this one
        IDX ilower = 0, iupper = proc_sizes[0];
        for(int iproc = 0; iproc < proc_id; ++iproc){
            ilower += proc_sizes[iproc];
            iupper += proc_sizes[iproc + 1];
        }

        // create the hypre matrix 
        HYPRE_IJMatrix J;
        HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &J);
        HYPRE_IJMatrixSetObjectType(J, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(J);
//        HYPRE_IJMatrixSetPrintLevel(J, 1);

        // boundary traces (TODO: consider parallel contribution from/to other proceses)
        for(Trace const& trace : fespace.get_boundary_traces()){
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            FE::elspan uL{uL_data.data(), uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            FE::elspan uR{uR_data.data(), uR_layout};

            // using the layout to match the input data (will be most optimal)
            auto resL_layout = u.create_element_layout(trace.elL.elidx);
            FE::elspan resL{resL_data.data(), resL_layout};

            // extract the compact values from the global u view
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residual
            resL = 0;

            // get the unperturbed residual 
            disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resL);

            // finite difference with perturbed solutions
            FE::elspan resLp{resLp_data.data(), resL_layout};

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, resL.vector_norm() * epsilon);

            // all the local information for a call to AddToValues
            std::size_t sizeL = uL.extents().ndof * ncomp;
            std::vector<IDX> rows{}; // use push_back to fill
            std::vector<IDX> ncols(sizeL, sizeL);
            std::vector<IDX> cols_data(sizeL * sizeL);
            mdspan cols{cols_data.data(), extents{sizeL, sizeL}};
            std::vector<T> Jcompact_data(sizeL * sizeL);
            mdspan Jcompact{Jcompact_data.data(), extents{sizeL, sizeL}};

            for(std::size_t idof = 0; idof < uL.extents().ndof; ++idof){
                for(std::size_t iv = 0; iv < ncomp; ++iv){
                    // local jacobian matrix index
                    std::size_t jlocal = idof * ncomp + iv;

                    T old_val = uL[idof, iv];
                    uL[idof, iv] += eps_scaled;

                    // get the perturbed residual
                    disc.boundaryIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp);

                    // get the matrix index corresponding to the perturbed u value 
                    IDX icol = ilower + u.get_layout()(FE::fe_index{(std::size_t) trace.elL.elidx, idof, iv});

                    // add the column index to the rows array 
                    // (NOTE: this is an out of order operation)
                    //  the values don't added in the upcomping loop do not necessarily belong to this row
                    rows.push_back(icol);

                    // scatter finite difference 
                    for(std::size_t idof2 = 0; idof2 < uL.extents().ndof; ++idof2){
                        for(std::size_t iv2 = 0; iv2 < ncomp; ++iv2){
                            // local jacobian matrix index
                            std::size_t ilocal = idof2 * ncomp + iv2;

                            // finite difference value
                            T fd_val = (resLp[idof2, iv2] - resL[idof2, iv2]) / eps_scaled;

                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }
                    // reset the perturbed value 
                    uL[idof, iv] = old_val;
                }
            }

            // call the AddValues 
            HYPRE_IJMatrixAddToValues(J, rows.size(), ncols.data(),
                    rows.data(), cols_data.data(), Jcompact_data.data());
        }

        // interior traces 
        for(Trace const& trace : fespace.get_interior_traces()){
            // set up compact data views
            auto uL_layout = u.create_element_layout(trace.elL.elidx);
            FE::elspan uL{uL_data.data(), uL_layout};
            auto uR_layout = u.create_element_layout(trace.elR.elidx);
            FE::elspan uR{uR_data.data(), uR_layout};

            // using the layout to match the input data (will be most optimal)
            auto resL_layout = u.create_element_layout(trace.elL.elidx);
            FE::elspan resL{resL_data.data(), resL_layout};
            auto resR_layout = u.create_element_layout(trace.elR.elidx);
            FE::elspan resR{resR_data.data(), resR_layout};

            // extract the compact values from the global u view
            FE::extract_elspan(trace.elL.elidx, u, uL);
            FE::extract_elspan(trace.elR.elidx, u, uR);

            // zero out the residual
            resL = 0;
            resR = 0;

            // get the unperturbed residual 
            disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resL, resR);

            // finite difference with perturbed solutions
            FE::elspan resLp{resLp_data.data(), resL_layout};
            FE::elspan resRp{resRp_data.data(), resR_layout};

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, std::max(resL.vector_norm() * epsilon, resR.vector_norm() * epsilon));

            // all the local information for a call to AddToValues
            std::size_t sizeL = uL.extents().ndof * ncomp;
            std::size_t sizeR = uR.extents().ndof * ncomp;
            std::size_t sizeLR = sizeL + sizeR;
            std::vector<IDX> rows{}; // use push_back to fill
            std::vector<IDX> ncols(sizeLR, sizeLR);
            std::vector<IDX> cols_data(sizeLR * sizeLR);
            mdspan cols{cols_data.data(), extents{sizeLR, sizeLR}};
            std::vector<T> Jcompact_data(sizeLR * sizeLR);
            mdspan Jcompact{Jcompact_data.data(), extents{sizeLR, sizeLR}};

            for(std::size_t idof = 0; idof < uL.extents().ndof; ++idof){
                for(std::size_t iv = 0; iv < ncomp; ++iv){
                    // === perturb the left side first ===
                    T old_val = uL[idof, iv];
                    uL[idof, iv] += eps_scaled;

                    // local jacobian matrix index
                    std::size_t jlocal = idof * ncomp + iv;

                    // get the perturbed residual
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

                    // get the matrix index corresponding to the perturbed u value 
                    IDX icol = ilower + u.get_layout()(FE::fe_index{(std::size_t) trace.elL.elidx, idof, iv});
                    // add the column index to the rows array 
                    // (NOTE: this is an out of order operation)
                    //  the values don't added in the upcomping loop do not necessarily belong to this row
                    rows.push_back(icol);

                    // scatter finite difference 
                    for(std::size_t idof2 = 0; idof2 < uL.extents().ndof; ++idof2){
                        for(std::size_t iv2 = 0; iv2 < ncomp; ++iv2){
                            T fd_val = (resLp[idof2, iv2] - resL[idof2, iv2]) / eps_scaled;

                            // local jacobian matrix index
                            std::size_t ilocal = idof2 * ncomp + iv2;

                            // use the layout to take care of indexing
                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }
                    for(int idof2 = 0; idof2 < uR.extents().ndof; ++idof2){
                        for(int iv2 = 0; iv2 < ncomp; ++iv2){
                            T fd_val = (resRp[idof2, iv2] - resR[idof2, iv2]) / eps_scaled;

                            // local jacobian matrix index
                            std::size_t ilocal = sizeL + idof2 * ncomp + iv2;

                            // use the layout to take care of indexing
                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }

                    // reset the perturbed value 
                    uL[idof, iv] = old_val;
                }
            }

            for(std::size_t idof = 0; idof < uR.extents().ndof; ++idof){
                for(std::size_t iv = 0; iv < ncomp; ++iv){
                    // === perturb the right side ===
                    T old_val = uR[idof, iv];
                    uR[idof, iv] += eps_scaled;

                    // local jacobian matrix index
                    std::size_t jlocal = sizeL + idof * ncomp + iv;

                    // get the perturbed residual
                    disc.traceIntegral(trace, fespace.meshptr->nodes, uL, uR, resLp, resRp);

                    // get the matrix index corresponding to the perturbed u value 
                    IDX icol = ilower + u.get_layout()(FE::fe_index{(std::size_t) trace.elR.elidx, idof, iv});
                    // add the column index to the rows array 
                    rows.push_back(icol);

                    // scatter finite difference 
                    for(std::size_t idof2 = 0; idof2 < uL.extents().ndof; ++idof2){
                        for(std::size_t iv2 = 0; iv2 < ncomp; ++iv2){
                            T fd_val = (resLp[idof2, iv2] - resL[idof2, iv2]) / eps_scaled;

                            // local jacobian matrix index
                            std::size_t ilocal = idof2 * ncomp + iv2;

                            // use the layout to take care of indexing
                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }
                    for(int idof2 = 0; idof2 < uR.extents().ndof; ++idof2){
                        for(int iv2 = 0; iv2 < ncomp; ++iv2){
                            T fd_val = (resRp[idof2, iv2] - resR[idof2, iv2]) / eps_scaled;

                            // local jacobian matrix index
                            std::size_t ilocal = sizeL + idof2 * ncomp + iv2;

                            // use the layout to take care of indexing
                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }

                    // reset the perturbed value 
                    uR[idof, iv] = old_val;
                }
            }

            // call the AddValues 
            HYPRE_IJMatrixAddToValues(J, rows.size(), ncols.data(),
                    rows.data(), cols_data.data(), Jcompact_data.data());
        }

        // domain integral
        for(Element const& el : fespace.elements){
            // set up compact data views (reuse the storage defined for traces)
            auto uel_layout = u.create_element_layout(el.elidx);
            FE::elspan u_el{uL_data.data(), uel_layout};

            // using the layout to match the input data (will be most optimal)
            auto ures_layout = u.create_element_layout(el.elidx);
            FE::elspan res_el{resL_data.data(), ures_layout};

            // extract the compact values from the global u view 
            FE::extract_elspan(el.elidx, u, u_el);

            // zero out the residual 
            res_el = 0;

            // unperturbed residual 
            disc.domainIntegral(el, fespace.meshptr->nodes, u_el, res_el);

            // finite difference with perturbed solutions
            FE::elspan res_elp{resLp_data.data(), ures_layout};

            // set up the perturbation amount scaled by unperturbed residual 
            T eps_scaled = std::max(epsilon, res_el.vector_norm() * epsilon);

            // all local information for a call to AddToValues
            std::size_t usize = u_el.extents().ndof * ncomp;
            std::vector<IDX> rows{}; // use push_back to fill
            std::vector<IDX> ncols(usize, usize); // initialized 
            std::vector<IDX> cols_data(usize * usize);
            mdspan cols{cols_data.data(), extents{usize, usize}};
            std::vector<T> Jcompact_data(usize * usize);
            mdspan Jcompact{Jcompact_data.data(), extents{usize, usize}};

            for(std::size_t idof = 0; idof < u_el.extents().ndof; ++idof){
                for(std::size_t iv = 0; iv < ncomp; ++iv){
                    // local jacobian matrix index
                    std::size_t jlocal = idof * ncomp + iv;

                    T old_val = u_el[idof, iv];
                    u_el[idof, iv] += eps_scaled;

                    // get the perturbed residual
                    disc.domainIntegral(el, fespace.meshptr->nodes, u_el, res_elp);

                    // get the matrix index corresponding to the perturbed u value 
                    IDX icol = ilower + u.get_layout()(FE::fe_index{(std::size_t) el.elidx, idof, iv});

                    // add the column index to the rows array 
                    // (NOTE: this is an out of order operation)
                    //  the values don't added in the upcomping loop do not necessarily belong to this row
                    rows.push_back(icol);

                    // scatter finite difference 
                    for(std::size_t idof2 = 0; idof2 < u_el.extents().ndof; ++idof2){
                        for(std::size_t iv2 = 0; iv2 < ncomp; ++iv2){
                            // local jacobian matrix index
                            std::size_t ilocal = idof2 * ncomp + iv2;

                            // finite difference value
                            T fd_val = (res_elp[idof2, iv2] - res_el[idof2, iv2]) / eps_scaled;

                            cols[ilocal, jlocal]= icol;
                            Jcompact[ilocal, jlocal] = fd_val;
                        }
                    }
                    // reset the perturbed value 
                    u_el[idof, iv] = old_val;
                }
            }

            // call the AddValues 
            HYPRE_IJMatrixAddToValues(J, rows.size(), ncols.data(),
                    rows.data(), cols_data.data(), Jcompact_data.data());
        }

        // finish setting up the matrix
        HYPRE_IJMatrixAssemble(J);
        return J;
    }
}
