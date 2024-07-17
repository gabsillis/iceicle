#pragma once

#include "mdspan/mdspan.hpp"
#include <type_traits>
namespace iceicle::linalg {

    /// @brief whether or not something is an mdspan
    template<class T>
    constexpr bool is_mdspan = false;

    /// @brief whether or not something is an mdspan
    template<class ElementType, class Extents, class Layout, class Accessor>
    constexpr bool is_mdspan<std::mdspan<ElementType, Extents, Layout, Accessor>> = true;

    /// @brief a matrix is a 2d mdspan
    template<class T>
    concept in_matrix = is_mdspan<T> && T::rank() == 2;

    template<class T>
    concept in_tensor = is_mdspan<T>;

    template<class T>
    concept inout_tensor = is_mdspan<T>;

    /// @brief a matrix that can be written to
    template<class T>
    concept out_matrix = is_mdspan<T> && T::rank() == 2 &&
    std::is_assignable_v<typename T::reference, typename T::element_type> && T::is_always_unique();
}
