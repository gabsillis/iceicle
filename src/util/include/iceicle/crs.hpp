/**
 * @brief compressed row storage 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once 

#include <ranges>
#include <type_traits>
#include <vector>
#include <span>
#include <algorithm>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace iceicle::util {

    /**
     * @brief compressed row storage 
     * @tparam T the data type 
     * @tparam IDX the index type 
     * TODO: Allocator
     */
    template<class T, class IDX = std::size_t>
    class crs {
        public:

        // ============
        // = Typedefs =
        // ============
        using value_type = T;
        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;

        private:

        /// @brief the number of nonzero entries (the data ptr size)
        size_type _nnz;

        /// @brief the number of rows
        size_type _nrow;

        /// @brief the data (size = nnz)
        value_type *_data{nullptr};

        /// @brief the indices of the start of each row (size = nrow + 1)
        index_type *_cols{nullptr};

        public:

        // ================
        // = Constructors =
        // ================
        
        constexpr crs() : _nnz(0), _nrow(0), _data(nullptr), _cols(nullptr) {}

        /// @brief consruct a crs from the indices for the start of each row 
        /// allocates enough data to accomodate 
        /// @param cols the indices for of the start of each row 
        ///             starting with zero 
        ///             also includes 1 past the end of the last row as the last entry
        constexpr 
        crs(std::span<const index_type> cols)
        : _nrow{(size_type) cols.size() - 1}, _cols{new index_type[cols.size()]}
        {
            // get the number of nonzeros
            _nnz = cols.back();

            // copy over the columns
            std::ranges::copy(cols, _cols);

            // allocate the _data 
            _data = new T[_nnz];
        }

        /**
         * @brief generic range based constructor for existing ragged data 
         * Any range of ranges (with arbitrary sizes) 
         * can then be converted into a compressed row storage
         *
         * @param R the range of ranges to copy 
         */
        template< class R >
        constexpr crs( const R &ragged_data )
        requires(std::ranges::range<R> && std::ranges::range<std::ranges::range_value_t<R>>)
        : _nnz{0}, _nrow{(size_type) std::distance(std::ranges::begin(ragged_data), std::ranges::end(ragged_data))},
          _cols{new index_type[_nrow + 1]}
        {
            // count up the number of nonzeros and row lengths
            _cols[0] = 0;
            IDX irow = 0;
            for(auto rowit = std::ranges::begin(ragged_data);
                    rowit != std::ranges::end(ragged_data); ++rowit, ++irow) {
                std::size_t sz = std::distance(std::ranges::begin(*rowit), std::ranges::end(*rowit));
                _nnz += sz;
                _cols[irow + 1] = _cols[irow] + sz;
            }

            // allocate the _data 
            _data = new T[_nnz];

            // copy over the data 
            irow = 0;
            for(auto rowit = std::ranges::begin(ragged_data);
                    rowit != std::ranges::end(ragged_data); ++rowit, ++irow) {
                std::copy(std::ranges::begin(*rowit),
                        std::ranges::end(*rowit), _data + _cols[irow]);
            }
        }

        /// @brief copy constructor
        constexpr crs(const crs& other)
        : _nnz(other._nnz), _nrow(other._nrow),
        _data{new value_type[_nnz]}, _cols{new index_type[_nrow + 1]}
        {
            if(other._data != nullptr)
                std::copy_n(other._data, _nnz, _data);
            if(other._cols != nullptr)
                std::copy_n(other._cols, _nrow + 1, _cols);
        }

        /// @brief move constructor
        constexpr crs(crs&& other)
        : _nnz(other._nnz), _nrow(other._nrow), 
        _data{other._data}, _cols{other._cols}
        {
            other._data = nullptr;
            other._cols = nullptr;
        }

        /// @brief copy assignment
        constexpr auto operator=(const crs& other) -> crs<T, IDX>& {
            _nnz = other._nnz;
            _nrow = other._nrow;

            if(_data != nullptr)
                delete[] _data;
            if(_cols != nullptr)
                delete[] _cols;

            _data = new value_type[_nnz];
            _cols = new index_type[_nrow + 1];

            if(other._data != nullptr)
                std::copy_n(other._data, _nnz, _data);
            if(other._cols != nullptr)
                std::copy_n(other._cols, _nrow + 1, _cols);
            return *this;
        }

        /// @brief move assignment
        constexpr auto operator=(crs&& other) -> crs<T, IDX>& {
            _nnz = other._nnz;
            _nrow = other._nrow;

            if(_data != nullptr){
                delete[] _data;
            }
            if(_cols != nullptr){
                delete[] _cols;
            }
            _data = other._data;
            other._data = nullptr;
            _cols = other._cols;
            other._cols = nullptr;
            return *this;
        }

        /// @brief destructor
        ~crs(){
            if(_data != nullptr){
                delete[] _data;
            }
            if(_cols != nullptr){
                delete[] _cols;
            }
        }

        // =========
        // = Sizes =
        // =========

        /// @brief get the total number of values stored 
        /// abbreviation for "number of non-zeros" from sparse matrix terminology
        inline constexpr 
        auto nnz() const noexcept -> size_type { return _nnz; }

        //// @brief the number of rows represented
        inline constexpr 
        auto nrow() const noexcept -> size_type { return _nrow; }

        /// @brief get the number of elements on the given row 
        /// @param irow the row index 
        /// @return the number of elements in this row
        inline constexpr 
        auto rowsize(index_type irow) const noexcept -> size_type 
        { return _cols[irow + 1] - _cols[irow]; }

        // ============
        // = Indexing =
        // ============
       
        /**
         * @brief index using a 2d index 
         * @param irow the row index
         * @param jcol the column index
         * @return the value at the location of the index pair 
         * (irow, jcol)
         */
        inline constexpr
        auto operator[](index_type irow, index_type jcol) noexcept 
        -> value_type& {
            return _data[_cols[irow] + jcol];
        }
        
        /**
         * @brief index using a 2d index 
         * @param irow the row index
         * @param jcol the column index
         * @return the value at the location of the index pair 
         * (irow, jcol)
         */
        inline constexpr
        auto operator[](index_type irow, index_type jcol) const noexcept 
        -> const value_type& {
            return _data[_cols[irow] + jcol];
        }

        /**
         * @brief get a std::span for a given row 
         * @param irow the index of the row 
         * @return a std::span covering the range of values in the row
         */
        inline constexpr 
        auto rowspan(index_type irow) noexcept 
        -> std::span<value_type> {
            return std::span<value_type>{
                _data + _cols[irow],
                _data + _cols[irow + 1]
            };
        }

        /**
         * @brief get a std::span for a given row 
         * @param irow the index of the row 
         * @return a std::span covering the range of values in the row
         */
        inline constexpr 
        auto rowspan(index_type irow) const noexcept
        -> std::span<const value_type> {
            return std::span<value_type>{
                _data + _cols[irow],
                _data + _cols[irow + 1]
            };
        }

        /// @brief get the raw data pointer
        inline constexpr 
        auto data() noexcept
        -> value_type*
        { return _data; }

        /// @brief get the raw data pointer
        inline constexpr 
        auto data() const noexcept
        -> const value_type*
        { return _data; }

        /// @brief get the raw column index array pointer 
        inline constexpr 
        auto cols() noexcept
        -> index_type*
        { return _cols; }

        /// @brief get the raw column index array pointer 
        inline constexpr 
        auto cols() const noexcept
        -> const index_type*
        { return _cols; }

        inline friend std::ostream& operator<<(std::ostream& out, const crs<T, IDX>& mat){
            for(IDX irow = 0; irow < mat.nrow(); ++irow){
                out << fmt::format("{}\n", mat.rowspan(irow));
            }
            return out;
        }
    };

    template<class T>
    crs(const std::vector<std::vector<T>>&) -> crs<T>;

    template<class Tnew, class IDXnew, class T, class IDX>
    constexpr
    auto convert_crs(const crs<T, IDX>& crs_old)
    -> crs<Tnew, IDXnew>{
        std::vector<IDXnew> colsnew(crs_old.nrow() + 1);
        std::copy_n(crs_old.cols(), crs_old.nrow() + 1, colsnew.begin());
        crs<Tnew, IDXnew> crs_new{colsnew};
        std::copy_n(crs_old.data(), crs_old.nnz(), crs_new.data());
        return crs_new;
    }
}
