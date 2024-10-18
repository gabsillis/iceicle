#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "mdspan/mdspan.hpp"
#include <cstddef>
#include <ostream>
#include <type_traits>
#include <fmt/core.h>
namespace iceicle::linalg {

    /// @brief whether or not something is an mdspan
    template<class T>
    constexpr bool is_mdspan = false;

    /// @brief whether or not something is an mdspan
    template<class ElementType, class Extents, class Layout, class Accessor>
    constexpr bool is_mdspan<std::mdspan<ElementType, Extents, Layout, Accessor>> = true;

    template<class T>
    concept mdspan_specialization = is_mdspan<T>;

    template<class T>
    concept in_vector = is_mdspan<T> && T::rank() == 1;
    template<class T>
    concept inout_vector = is_mdspan<T> && T::rank() == 1;
    template<class T>
    concept out_vector = is_mdspan<T> && T::rank() == 1;

    /// @brief a matrix is a 2d mdspan
    template<class T>
    concept in_matrix = is_mdspan<T> && T::rank() == 2;
    template<class T>
    concept inout_matrix = is_mdspan<T> && T::rank() == 2;

    template<class T>
    concept in_tensor = is_mdspan<T>;

    template<class T>
    concept inout_tensor = is_mdspan<T>;

    /// @brief if data_handle() will give a contiguous sequence
    template<class T>
    concept contiguous_mdspan = mdspan_specialization<T>
        && std::same_as<typename T::layout_type, std::layout_right>;

    /// @brief a matrix that can be written to
    template<class T>
    concept out_matrix = is_mdspan<T> && T::rank() == 2 &&
    std::is_assignable_v<typename T::reference, typename T::element_type> && T::is_always_unique();

    template<class T1, class T2>
    auto copy(T1 x, T2 y) -> void
    requires( in_vector<T1> && !contiguous_mdspan<T1> 
            && out_vector<T2> && !contiguous_mdspan<T2>)
    {
        using index_type = decltype(x)::index_type;
        for(index_type i = 0; i < x.extent(0); ++i)
            y[i] = x[i];
    }

    /// @brief copy the contents of a into b 
    /// @param a the mdspan to copy from 
    /// @param b the mdspan to copy to
    auto copy(contiguous_mdspan auto a, contiguous_mdspan auto b) -> void {
        using index_type = decltype(a)::index_type;
        for(index_type i = 0; i < a.size(); ++i){
            b.data_handle()[i] = a.data_handle()[i];
        }
    }

    /// @brief get a view over a fixed-size-tensor as an mdspan 
    template<class T, std::size_t... sizes>
    [[nodiscard]] constexpr inline
    auto as_mdspan(NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, sizes...>& t)
    -> std::mdspan<T, std::extents<std::size_t, sizes...>>
    { return std::mdspan<T, std::extents<std::size_t, sizes...>>{t.ptr()}; }

    template<class T>
    auto axpy(T alpha, in_vector auto x, inout_vector auto y){
        using index_type = decltype(x)::index_type;
        for(index_type i = 0; i < x.extent(0); ++i)
            y[i] += alpha * x[i];
    }

    inline constexpr 
    auto dot(in_vector auto x, in_vector auto y) -> decltype(x)::value_type
    {
        using index_type = decltype(x)::index_type;
        using value_type = decltype(x)::value_type;

        value_type sum = 0;
        for(index_type i = 0; i < x.extent(0); ++i){
            sum += x[i] * y[i];
        }
        return sum;
    }

    inline constexpr 
    auto transpose_prod(in_matrix auto A, in_vector auto x, out_vector auto y) -> void
    {
        using index_type = decltype(A)::index_type;
        using value_type = decltype(A)::value_type;
        const auto m = A.extent(0);
        const auto n = A.extent(1);
        for(index_type j = 0; j < n; ++j) y[j] = 0;

        for(index_type i = 0; i < m; ++i){
            for(index_type j = 0; j < n; ++j){
                y[j] += A[i, j] * x[i];
            }
        }
    }

    auto operator<<(std::ostream& os, in_matrix auto A) -> std::ostream&
    {
        using index_type = decltype(A)::index_type;
        using value_type = decltype(A)::value_type;
        const auto m = A.extent(0);
        const auto n = A.extent(1);
        
        for(index_type i = 0; i < m; ++i){
            for(index_type j = 0; j < n; ++j){
                double value = A[i, j];
                os << fmt::format("{:10.4f} ", value);
            }
            os << std::endl;
        }
        return os;
    }

        auto operator<<(std::ostream& os, in_vector auto v) -> std::ostream&
    {
        using index_type = decltype(v)::index_type;
        using value_type = decltype(v)::value_type;
        
        os << "[ ";
        for(index_type i = 0; i < v.extent(0); ++i){
            double value = v[i];
            os << fmt::format("{:10.4f} ", value);
        }
        os << "]" << std::endl;
        return os;
    }

}
