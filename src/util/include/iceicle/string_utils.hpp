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
}
