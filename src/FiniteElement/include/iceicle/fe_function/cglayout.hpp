/**
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief cglayout represents the memoery layout of
 * a CG representation of a vector-valued fe_function
 */
#pragma once
#include <iceicle/fe_function/layout_enums.hpp>

#include <type_traits>

namespace FE {

    /**
     * @brief LayoutPolicy for a CG representation of a vector-valued fe_function
     */
    template<
        typename T,
        std::size_t ncomp = dynamic_ncomp,
        LAYOUT_VECTOR_ORDER order = DOF_LEFT
    >
    class cg_layout {
        
        /* optional stored values to index */ 

        /// dynamic number of vector components
        std::enable_if<is_dynamic_ncomp<ncomp>::value, int> ncomp_d;

        /// store the number of degrees of freedom if DOF_RIGHT
        std::size_t ndof;

        inline constexpr std::size_t get_ncomp() const {
            if constexpr(is_dynamic_ncomp<ncomp>::value){
                return ncomp_d;
            } else {
                return ncomp;
            }
        }

        public:

        /**
         * Create a CG layout policy 
         * @param ndof the number of nodes that will be represented in the layout 
         * @param ncomp_d (only used if ncomp isn't specified) the number of vector components per node 
         */
        cg_layout(int ndof, std::enable_if<is_dynamic_ncomp<ncomp>::value, int> ncomp_d)
        : ndof(ndof), ncomp_d(ncomp_d) {}

        inline static constexpr int is_global() { return true; }
        inline static constexpr int is_nodal()  { return true; }

        constexpr std::size_t size() const noexcept {
            return ndof * get_ncomp();
        }

        /**
         * @brief convert a multidimensional global node index to a 1D index 
         * @param idx the degree of freedom (node) index and the vector component 
         * @return an index into the 1D memory structure
         */
        constexpr std::size_t operator()(const gnode_index &idx) const {
            if constexpr (order == DOF_LEFT){
                return idx.idof * get_ncomp() + idx.iv;
            } else {
                return idx.iv * ndof + idx.idof;
            }
        }
    };
}
