/**
 * @brief storage for data on boundaries
 *
 * @author Gianni Absillis(gabsill@ncsu.edu)
 */
#pragma once

#include <span>
#include <vector>
#include <Numtool/fixed_size_tensor.hpp>
namespace iceicle {

    template<class T, std::size_t neq = std::dynamic_extent>
    class DirichletData {
        private:


        std::size_t get_neq();
        public:

    };
}
