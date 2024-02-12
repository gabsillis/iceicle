/**
 * @brief utilities to inteface with HYPRE
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_mv.h"
#include "HYPRE_utilities.h"
#include <vector>
#include <complex>
#include "mdspan/mdspan.hpp"
#include "mpi.h"
#include "iceicle/fespace/fespace.hpp"

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
     * @brief an inclusive range which mirrors the ranges used to define 
     * HYPRE IJ structures
     */
    struct HypreBounds{
        /// @brief the lower index of the index range for this processor
        HYPRE_BigInt ilower;

        /// @brief the upper index of the range for this processor (inclusive)
        HYPRE_BigInt iupper;

        /// @brief the size of the range defined by [ilower, iupper]
        constexpr HYPRE_BigInt size() const noexcept {
            return iupper - ilower + 1;
        }
    };

    /**
     * @brief compute the hypre bounds to represent 
     * the degrees of freedom for a finite element space 
     * outer product with the number of vector components to represent 
     * (ndof * ncomp)
     * 
     * @param fespace the finite element space 
     * @param ncomp the number of vector components 
     * @param comm the optional MPI communicator to use 
     *             defaults to MPI_COMM_WORLD
     * @return the hypre bounds to be able to represent data in the fespace
     */
    template<class T, class IDX, int ndim>
    HypreBounds compute_hypre_bounds(
        FE::FESpace<T, IDX, ndim> &fespace,
        std::size_t ncomp,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        // MPI info
        int nproc;
        int proc_id;
        MPI_Comm_size(comm, &nproc);
        MPI_Comm_rank(comm, &proc_id);

        // First communicate the size ndof * ncomp for each process 
        const std::size_t process_size = fespace.dg_offsets.calculate_size_requirement(ncomp);
        std::vector<std::size_t> proc_sizes(nproc);
        proc_sizes[proc_id] = process_size;
        for(int iproc = 0; iproc < nproc; ++iproc){
            MPI_Bcast(&(proc_sizes[iproc]), 1, mpi_get_type<std::size_t>(), iproc, comm);
        }

        // add the sizes for each processor until this one
        HYPRE_BigInt ilower = 0, iupper = proc_sizes[0];
        for(int iproc = 0; iproc < proc_id; ++iproc){
            ilower += proc_sizes[iproc];
            iupper += proc_sizes[iproc + 1];
        }
        iupper--; // go to inclusive range

        return HypreBounds{ilower, iupper};
    }

    /**
     * @brief compute hypre bounds to represent the components of an fespan 
     *
     * @param fedata the fespan to designate the size of data 
     * @param comm the mpi communicator, defaults to MPI_COMM_WORLD
     * @return the hypre bounds to be able to represent the fespan
     */
    template<class T, class LayoutPolicy, class AccessorPolicy>
    HypreBounds compute_hypre_bounds(
        const FE::fespan<T, LayoutPolicy, AccessorPolicy> &fedata,
        MPI_Comm comm = MPI_COMM_WORLD
    ) {
        // MPI info
        int nproc;
        int proc_id;
        MPI_Comm_size(comm, &nproc);
        MPI_Comm_rank(comm, &proc_id);

        // First communicate the size ndof * ncomp for each process 
        const std::size_t process_size = fedata.size();
        std::vector<std::size_t> proc_sizes(nproc);
        proc_sizes[proc_id] = process_size;
        for(int iproc = 0; iproc < nproc; ++iproc){
            MPI_Bcast(&(proc_sizes[iproc]), 1, mpi_get_type<std::size_t>(), iproc, comm);
        }

        // add the sizes for each processor until this one
        HYPRE_BigInt ilower = 0, iupper = proc_sizes[0];
        for(int iproc = 0; iproc < proc_id; ++iproc){
            ilower += proc_sizes[iproc];
            iupper += proc_sizes[iproc + 1];
        }
        iupper--; // go to inclusive range

        return HypreBounds{ilower, iupper};
    }

    /**
     * @brief lightweight interface for a HYPRE_IJVector
     * Implicitly convertible
     * RAII
     *
     * not-copyable (treat like a std::unique_ptr)
     */
    class HypreVec {

        /// @brief the hypre bounds used to create the vector
        HypreBounds bounds;

        /// @brief the hypre vec handler
        HYPRE_IJVector vec;

        /// @brief par vector handle for casts
        HYPRE_ParVector par_vec;

        public:

        /// @brief default constructor
        HypreVec() : vec(nullptr) {}

        /**
         * @brief construct a HypreVec 
         * creates a HYPRE_IJVector with PARCSR format
         * @param bounds the lower and upper bounds 
         * @param comm the mpi communicator (optional default = MPI_COMM_WORLD)
         */
        HypreVec(
            const HypreBounds &bounds,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : bounds(bounds) {
             HYPRE_IJVectorCreate(comm, bounds.ilower, bounds.iupper, &vec);
             HYPRE_IJVectorSetObjectType(vec, HYPRE_PARCSR);
             HYPRE_IJVectorInitialize(vec);
        }

        /**
         * @brief Construct a hypre vector from an fespan
         * first computes bounds and calls the bounds constructor 
         * then copies the data into the hypre vector
         * WARNING: assumes the fespan has contiguous data 
         * TODO: test contiguous from LayoutPolicy 
         *
         * @param fedata the data to copy from 
         * @param comm the mpi communicator 
         */
        template<typename T, class LayoutPolicy, class AccessorPolicy>
        HypreVec(
            const FE::fespan<T, LayoutPolicy, AccessorPolicy> &fedata,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : HypreVec{compute_hypre_bounds(fedata, comm)} {

            // copy over the values
            std::vector<HYPRE_BigInt> rows(bounds.size());
            std::iota(rows.begin(), rows.end(), bounds.ilower);
            HYPRE_IJVectorSetValues(vec, bounds.size(), rows.data(), fedata.data());
        }

        // === remove copy semantics ===
        HypreVec(const HypreVec &) = delete;
        HypreVec &operator=(const HypreVec &) = delete;

        /// @brief move constructor
        HypreVec(HypreVec &&other) noexcept
        : vec{nullptr} {
            std::swap(bounds, other.bounds);
            std::swap(vec, other.vec);
        }

        /// @brief move copy assignment
        HypreVec& operator=(HypreVec &&other) noexcept {
            std::swap(bounds, other.bounds);
            std::swap(vec, other.vec);
            return *this;
        }

        /// @brief destructor
        ~HypreVec(){
            if(vec) HYPRE_IJVectorDestroy(vec);
        }

        /// @brief explicit cast to bool for checking pointer
        explicit operator bool() const {return static_cast<bool>(vec);}

        /// @brief implicit cast to HYPRE_IJVector so this can be used 
        /// as if it is a native hypre object with hypre functions
        /// but with added safety and RAII features
        operator HYPRE_IJVector() { return vec; }

        /// @brief implicit cast to HYPRE_ParVector 
        operator HYPRE_ParVector() {
            HYPRE_IJVectorGetObject(vec, (void**) &par_vec);
            return par_vec;
        }

        /**
         * @brief extract the data in the HYPRE_IJVector to an fespan 
         * WARNING: assumes the fespan layout matches the layout used to construct 
         * the hypre vector 
         *
         * @param fedata the data to fill
         */
        template<typename T, class LayoutPolicy, class AccessorPolicy>
        void extract_data(FE::fespan<T, LayoutPolicy, AccessorPolicy> fedata){
            // copy over the values
            std::vector<HYPRE_BigInt> rows(bounds.size());
            std::iota(rows.begin(), rows.end(), bounds.ilower);
            HYPRE_IJVectorGetValues(vec, bounds.size(), rows.data(), fedata.data());
        }

        /**
         * @brief extract the data in the HYPRE_IJVector to a span
         * WARNING: assumes the fespan layout matches the layout used to construct 
         * the hypre vector 
         *
         * @param data the data to fill
         */
        template<typename T, std::size_t Extent>
        void extract_data(std::span<T, Extent> data){
            // copy over the values
            std::vector<HYPRE_BigInt> rows(bounds.size());
            std::iota(rows.begin(), rows.end(), bounds.ilower);
            HYPRE_IJVectorGetValues(vec, bounds.size(), rows.data(), data.data());
        }

        /**
         * @brief set the values in a given range 
         * @param start_row the processor local start index 
         * @param data the data to copy over 
         */
        template<typename T, std::size_t Extent>
        void set_values(std::size_t start_row, std::span<T, Extent> data){
            std::vector<HYPRE_BigInt> rows(data.extent);
            std::iota(rows.begin(), rows.end(), bounds.ilower + start_row);
            static constexpr HYPRE_Int SET_ACTION = 1;
            HYPRE_IJVectorUpdateValues(vec, data.extent, rows.data(), data.data(), SET_ACTION);
        }

        /**
         * @brief add to the values in a given range 
         * @param start_row the processor local start index 
         * @param data the data to add
         */
        template<typename T, std::size_t Extent>
        void add_values(std::size_t start_row, std::span<T, Extent> data){
            std::vector<HYPRE_BigInt> rows(data.extent);
            std::iota(rows.begin(), rows.end(), bounds.ilower + start_row);
            static constexpr HYPRE_Int ADD_ACTION = 1;
            HYPRE_IJVectorUpdateValues(vec, data.extent, rows.data(), data.data(), ADD_ACTION);
        }
    };

    /**
    * @brief lightweight interface for a HYPRE_IJMatrix
    * Implicitly convertible
    * RAII
    *
    * not-copyable (treat like a std::unique_ptr)
    */
    class HypreMat {

        /// @brief the hypre bounds that define the rows of the matrix
        HypreBounds row_bounds;

        /// @brief the hypre bounds that define the columns of the matrix
        HypreBounds col_bounds;

        /// @brief the hypre matrix handle
        HYPRE_IJMatrix mat;

        /// @brief par csr handle for casts
        HYPRE_ParCSRMatrix parcsr_mat;

        public:

        /// @brief default constructor (nullptr for matrix handle)
        HypreMat() : mat{nullptr} {}

        /**
            * @brief construct a HypreMat 
            * Creates a HYPRE_IJMatrix with PARCSR format 
            * @param row_bounds the index range for this processor for the rows 
            * @param col_bounds the index range for this processor for the columns 
            * @param comm the mpi commuicator (optional) default = MPI_COMM_WORLD
            */
        HypreMat(
            const HypreBounds &row_bounds,
            const HypreBounds &col_bounds,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : row_bounds(row_bounds), col_bounds(col_bounds) {
            HYPRE_IJMatrixCreate(comm, row_bounds.ilower, row_bounds.iupper, 
                col_bounds.ilower, col_bounds.iupper, &mat);
            HYPRE_IJMatrixSetObjectType(mat, HYPRE_PARCSR);
            HYPRE_IJMatrixInitialize(mat);
        }

        /**
            * @brief construct a square HypreMat 
            * Creates a square HYPRE_IJMatrix with PARCSR format 
            * @param square_bounds the row and column index range for this processor 
            * @param comm the mpi communicator (optional)
            */
        HypreMat(
            const HypreBounds &square_bounds,
            MPI_Comm comm = MPI_COMM_WORLD
        ) : HypreMat(square_bounds, square_bounds, comm) {}

        // === remove copy semantics 
        HypreMat(const HypreMat &) = delete;
        HypreMat &operator=(const HypreMat &) = delete;

        /// @brief move constructor 
        HypreMat(HypreMat &&other) noexcept 
        : mat{nullptr} {
            std::swap(row_bounds, other.row_bounds);
            std::swap(col_bounds, other.col_bounds);
            std::swap(mat, other.mat);
        }

        /// @brief move copy assignment 
        HypreMat &operator=(HypreMat &&other) noexcept {
            std::swap(row_bounds, other.row_bounds);
            std::swap(col_bounds, other.col_bounds);
            std::swap(mat, other.mat);
            return *this;
        }

        /// @brief destructor 
        ~HypreMat(){
            if(mat) HYPRE_IJMatrixDestroy(mat);
        }


        /// @brief explicit cast to bool for checking pointer
        explicit operator bool() const {return static_cast<bool>(mat);}

        /// @brief implicit cast to HYPRE_IJMatrix so this can be used 
        /// as if it is a native hypre object with hypre functions
        /// but with added safety and RAII features
        operator HYPRE_IJMatrix() { return mat; }

        /// @brief implicit cast to HYPRE_ParCSRMatrix so this can be used 
        /// as if it is a native hypre object with hypre functions
        operator HYPRE_ParCSRMatrix() { 
            HYPRE_IJMatrixGetObject(mat, (void**) &parcsr_mat);
            return parcsr_mat;
        }

        /**
         * @brief set the values in a portion of the matrix to the given matrix 
         *
         * @param rowstart_loc the processor local row index of the top left corner 
         * of the location to set the matrix
         * @param colstart_loc the processor local column index of the top left corner 
         * of the location to set the matrix
         * @param data the mdspan to set values from
         */
        template<typename T, class Extents, class LayoutPolicy, class AccessorPolicy>
        void set_values(
            std::size_t rowstart_loc,
            std::size_t colstart_loc,
            std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> data
        ){
            // TODO: bounds checking for 
            // given bounds inside HypreMat bounds
            std::vector<T> values(data.extent(1));
            std::vector<HYPRE_BigInt> cols(data.extent(1));
            std::iota(cols.begin(), cols.end(), col_bounds.ilower + colstart_loc);

            HYPRE_BigInt nnz = cols.size();
            for(HYPRE_BigInt irow = 0; irow < data.extent(0); ++irow){
                // global row index 
                HYPRE_BigInt irow_glob = row_bounds.ilower + rowstart_loc + irow;

                // get the values to copy
                for(int jcol = 0; jcol < data.extent(1); ++jcol){
                    values[jcol] = data[irow, jcol];
                }

                // call the hypre interface
                HYPRE_IJMatrixSetValues(mat, 1, &nnz, &irow_glob, cols.data(), values.data());
            }
        }

        /**
         * @brief add the values in a portion of the matrix to the given matrix 
         *
         * @param rowstart_loc the processor local row index of the top left corner 
         * of the location to add the matrix
         * @param colstart_loc the processor local column index of the top left corner 
         * of the location to add the matrix
         * @param data the mdspan to add values from
         */
        template<typename T, class Extents, class LayoutPolicy, class AccessorPolicy>
        void add_values(
            std::size_t rowstart_loc,
            std::size_t colstart_loc,
            std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> data
        ){
            // TODO: bounds checking for 
            // given bounds inside HypreMat bounds
            std::vector<T> values(data.extent(1));
            std::vector<HYPRE_BigInt> cols(data.extent(1));
            std::iota(cols.begin(), cols.end(), col_bounds.ilower + colstart_loc);

            HYPRE_BigInt nnz = cols.size();
            for(HYPRE_BigInt irow = 0; irow < data.extent(0); ++irow){
                // global row index 
                HYPRE_BigInt irow_glob = row_bounds.ilower + rowstart_loc + irow;

                // get the values to copy
                for(int jcol = 0; jcol < data.extent(1); ++jcol){
                    values[jcol] = data[irow, jcol];
                }

                // call the hypre interface
                HYPRE_IJMatrixAddToValues(mat, 1, &nnz, &irow_glob, cols.data(), values.data());
            }
        }
    };
}
