/// @brief useful algorithms

#include <concepts>
#include <ranges>
#include <vector>
#include <algorithm>
namespace iceicle::util {

    /// @brief returns true if a is a (non-strict) subset of b 
    /// @tparam R1 the range type for a
    /// @tparam R2 the range type for b
    /// @param a the first set 
    /// @param b the second set
    template<std::ranges::viewable_range R1, std::ranges::viewable_range R2>
    bool subset(R1&& a, R2&& b)
    requires(
        std::strict_weak_order<std::less<std::ranges::range_value_t<R1>>, 
            std::ranges::range_value_t<R1>, std::ranges::range_value_t<R2>>
    ) {
        std::vector acpy{a};
        std::vector bcpy{b};
        std::ranges::sort(acpy);
        std::ranges::sort(bcpy);

        auto bptr = bcpy.begin();
        for(auto aptr = a.begin(); aptr != a.end(); ++aptr){
            while(*bptr < *aptr) {
                ++bptr;
                if(bptr == bcpy.end()) return false;
            };
            if(*bptr != *aptr) return false;
        }
        return true;
    }

    /// @brief returns true if the sets are equal
    /// @tparam R1 the range type for a
    /// @tparam R2 the range type for b
    /// @param a the first set 
    /// @param b the second set
    template<std::ranges::viewable_range R1, std::ranges::viewable_range R2>
    bool eqset(R1&& a, R2&& b)
    requires(
        std::strict_weak_order<std::less<std::ranges::range_value_t<R1>>, 
            std::ranges::range_value_t<R1>, std::ranges::range_value_t<R2>>
    ){
        if(a.size() != b.size()) return false;

        std::vector acpy{a};
        std::vector bcpy{b};
        std::ranges::sort(acpy);
        std::ranges::sort(bcpy);
        for(std::size_t i = 0; i < acpy.size(); ++i){
            if(acpy[i] != bcpy[i]) return false;
        }
        return true;
    }

}
