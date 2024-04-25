#pragma once
#include <algorithm>
#include <string_view>
#include <cctype>
namespace iceicle::util {

    /**
     * @brief check if two string views are equivalent ignoring case 
     * @param a the first string view 
     * @param b the second string view 
     * @return true if a and b are the same string when ignoring case 
     *         false otherwise 
     */
    constexpr bool eq_icase(std::string_view a, std::string_view b) noexcept {
        auto case_insensitive_cmp = [](char a, char b) -> bool {
            return std::tolower(static_cast<unsigned char>(a)) ==
                std::tolower(static_cast<unsigned char> (b));
        };
        return std::ranges::equal(a, b, case_insensitive_cmp);
    }

    /// @brief compare a string_view a to a collection of other string views
    /// ignoring case
    /// @param a the string_view to compare to all the others 
    /// @param to_compare the string_views to compare to 
    /// @return true if a matches any of to_compare
    template<class... StringTs>
    constexpr 
    auto eq_icase_any(std::string_view a,  StringTs... to_compare) -> bool {
        bool any_of = (eq_icase(a, to_compare) || ...);
        return any_of;
    }
}
