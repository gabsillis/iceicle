/**
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief common definitions for memory layouts
 */
#pragma once
#include <cstdlib>

namespace FE {

    /**
     * @brief defines how degrees of freedom are organized wrt vector components
     */
    enum LAYOUT_VECTOR_ORDER {
        /**
         * degrees of freedom are the left most index in row major order setting 
         * so each degree of freedom will have a stride of the number of vector components 
         */
        DOF_LEFT = 0,
        /** degrees of freedom are the rightmost index in row major order setting
         * indices are dof fastest and in large chunks for each vector component 
         */
        DOF_RIGHT
    };

    /** @brief tag to specify that the number of vector component is a runtime parameter */
    static constexpr int dynamic_ncomp = -1;

    /**
     * @brief determine if the number of vector components represents a dynamic number 
     */
    template<int ncomp>
    struct is_dynamic_ncomp {

        /// the boolean value
        inline static constexpr bool value = (ncomp == -1);
    };


    /**
     * @brief index into a fespan 
     * collects the 3 required indices 
     * all indices default to zero
     */
    struct fe_index {
        /// the element index 
        std::size_t iel = 0;

        /// the local degree of freedom index 
        std::size_t idof = 0;

        /// the index of the vector component
        std::size_t iv = 0;
    };

    /**
     * @brief index into a global nodal structure
     * collects the two required indices 
     * indices default to zero 
     */
    struct gnode_index {
        /// the node index 
        std::size_t idof = 0;

        /// the index of the vector component
        std::size_t iv = 0;
    };

}
