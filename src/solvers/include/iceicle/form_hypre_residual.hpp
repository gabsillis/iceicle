/**
 * @brief form the residual as a hypre vector
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "iceicle/fespace/fespace.hpp"
#include "mpi.h"
#include "iceicle/form_hypre_jacobian.hpp"
#include <numeric>

namespace ICEICLE::SOLVERS {

    /**
     * @brief take an fespan with default accessor policy 
     * and send that data to a HYPRE IJ vector 
     *
     * WARNING: this has no way to track the storage duration of the matrix 
     * HYPRE_IJVectorDestroy() must be called to free memory
     *
     * @param fespace the finite element space 
     * @param ncomp the number of vector components
     * @param data the data to send 
     * @param comm (optional) mpi communicator default is MPI_COMM_WORLD
     */
    template<
        class T,
        class IDX,
        int ndim,
        class resLayoutPolicy
    >
    HYPRE_IJVector fespan_to_hypre_vec(
        FE::FESpace<T, IDX, ndim> &fespace,
        const std::size_t ncomp,
        FE::fespan<T, resLayoutPolicy> data,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        // MPI info
        int nproc;
        int proc_id;
        MPI_Comm_size(comm, &nproc);
        MPI_Comm_rank(comm, &proc_id);


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
        iupper--; // go to inclusive range

        // create the list of all rows in this process
        std::vector<T> rows(process_size);
        std::iota(rows.begin(), rows.end(), ilower);

        HYPRE_IJVector hypre_vec;
        HYPRE_IJVectorCreate(comm, ilower, iupper, &hypre_vec);
        HYPRE_IJVectorSetObjectType(hypre_vec, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(hypre_vec);

        HYPRE_IJVectorSetValues(hypre_vec, process_size, rows, data.data());

        return hypre_vec;
    }


}
