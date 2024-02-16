#pragma once
#include <mpi.h>
#include <petsc.h>
#include <petscerror.h>
#include <mdspan/mdspan.hpp>
#include <petscsystypes.h>
#include <petscvec.h>

namespace ICEICLE::PETSC {

    /**
     * @brief add to a logically dense block from an mdspan to a pestc matrix A 
     * Adds the values, if there previously was no value, just puts the value in that location
     * @param A the petsc matrix to add values to
     * @param rowstart the global starting row index of the block 
     * @param colstart the global starting column index of the block 
     * @param data the logically dense matrix to add 
     * @param comm the mpi communicator
     */
    template<typename T, class Extents, class LayoutPolicy, class AccessorPolicy>
    void add_to_petsc_mat(
        Mat A,
        std::size_t rowstart,
        std::size_t colstart,
        std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> data,
        MPI_Comm comm = MPI_COMM_WORLD
    ) noexcept {
        std::size_t dimprod = data.extent(0) * data.extent(1);
        // indices are the offest plus 0, 1, ... for rows and columns
        // row idxm[i] and column idxn[j] set the value values[i*n + j]
        std::vector<PetscInt> idxm(data.extent(0));
        std::vector<PetscInt> idxn(data.extent(1));
        std::iota(idxm.begin(), idxm.end(), rowstart);
        std::iota(idxn.begin(), idxn.end(), colstart);

        std::vector<T> values{};
        values.reserve(dimprod);
        for(std::size_t i = 0; i < data.extent(0); ++i){
            for(std::size_t j = 0; j < data.extent(1); ++j){
                values.push_back(data[i, j]);
            }
        }
        PetscCallAbort(comm, MatSetValues(A, data.extent(0), idxm.data(), data.extent(1),
                    idxn.data(), values.data(), ADD_VALUES));
    }


    /**
     * @brief insert a logically dense block from an mdspan to a pestc matrix A 
     * inserts the values or overwrites previous values
     * @param A the petsc matrix to add values to
     * @param rowstart the global starting row index of the block 
     * @param colstart the global starting column index of the block 
     * @param data the logically dense matrix to insert
     * @param comm the mpi communicator
     */
    template<typename T, class Extents, class LayoutPolicy, class AccessorPolicy>
    void insert_in_petsc_mat(
        Mat A,
        std::size_t rowstart,
        std::size_t colstart,
        std::experimental::mdspan<T, Extents, LayoutPolicy, AccessorPolicy> data,
        MPI_Comm comm = MPI_COMM_WORLD
    ) noexcept {
        std::size_t dimprod = data.extent(0) * data.extent(1);
        std::vector<PetscInt> idxm{};
        std::vector<PetscInt> idxn{};
        std::vector<T> values{};
        idxm.reserve(dimprod);
        idxn.reserve(dimprod);
        values.reserve(dimprod);
        for(std::size_t i = 0; i < data.extent(0); ++i){
            for(std::size_t j = 0; j < data.extent(1); ++j){
                idxm.push_back(i + rowstart);
                idxn.push_back(j + colstart);
                values.push_back(data[i, j]);
            }
        }
        PetscCallAbort(comm, MatSetValues(A, data.extent(0), idxm.data(),data.extent(1),
                    idxn.data(), data.data(), values.data(), INSERT_VALUES));
    }

    /**
     * @brief a non-owning view of a petsc vec 
     * automatically releases view with destructor or the release() function
     */
    class VecSpan {
        Vec v;
        PetscScalar *_data;
        PetscScalar *_end_data;

        public:

        using iterator = PetscScalar*;
        using reference = PetscScalar&;
        using pointer = PetscScalar*;
        using size_type = std::size_t;

        /**
         * @brief create a span from a Petsc vec 
         */
        inline VecSpan(Vec v) : v(v) {
            PetscInt local_size;
            VecGetLocalSize(v, &local_size);
            VecGetArray(v, &_data);
            _end_data = _data + local_size;
        }

        // delete copy semantics
        VecSpan(const VecSpan &other) = delete;
        VecSpan &operator=(const VecSpan &other) = delete;

        VecSpan(VecSpan &&other) = default;
        VecSpan &operator=(VecSpan &&other) = default;

        /**
         * @brief release this view 
         * afterwards, this will non longer function to access the vec 
         * WARNING: this container is invalid after release()
         * NOTE: A preferred pattern may be to use this in a scoped section of code 
         * and allow the destructor to release the view at the end of the sub-scope
         */
        inline void release(){
            VecRestoreArray(v, &_data);
            _data = nullptr;
            _end_data = nullptr;
        }

        /// @brief destructor
        ~VecSpan(){
            if(_data != nullptr) release();
        }

        /// @brief returns an iterator to the first element of the span,
        /// if the span is empty, the returned iterator will be equal to end 
        inline constexpr iterator begin() const noexcept { return _data; }

        /// @brief returns an iterator to the first element of the span,
        /// if the span is empty, the returned iterator will be equal to end 
        inline constexpr const iterator cbegin() const noexcept { return _data; }

        /// @brief 
        /// Returns an iterator to the element following the last element of the span.
        /// This element acts as a placeholder; attempting to access it results in undefined behavior. 
        inline constexpr iterator end() const noexcept { return _end_data; } 

        /// @brief 
        /// Returns an iterator to the element following the last element of the span.
        /// This element acts as a placeholder; attempting to access it results in undefined behavior. 
        inline constexpr const iterator cend() const noexcept { return _end_data; } 

        /// @brief returns a reverse iterator to the first element of the reversed span
        inline constexpr iterator rbegin() const noexcept { return _end_data - 1; }

        /// @brief returns a reverse iterator to the first element of the reversed span
        inline constexpr iterator const crbegin() const noexcept { return _end_data - 1; }

        /// @brief returns a reverse iterator the last element of the reversed span
        inline constexpr iterator rend() const noexcept { return _data - 1; }

        /// @brief returns a reverse iterator the last element of the reversed span
        inline constexpr iterator const crend() const noexcept { return _data - 1; }

        /// @brief returns a reference to the first element in the span
        inline constexpr reference front() const { return _data[0]; }

        /// @brief returns a reference to the last element in the span
        inline constexpr reference back() const { return _end_data[-1]; }

        /// @brief returns a reference to the idxth element in the _data 
        inline constexpr reference operator[](size_type idx) const { return _data[idx]; } 

        /// @brief get the underlying pointer 
        inline constexpr pointer data() const { return _data; }
    };

}
