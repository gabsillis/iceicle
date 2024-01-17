/**
 * @brief enumerations for general use in the finite element API
 */
#pragma once

namespace FE {

    /** 
     * @brief what the reference domain is 
     * This is important for determining equivalent external API objects
     * such as the element type when creating output files 
     */
    enum class DOMAIN_TYPE {
        HYPERCUBE, // maps to a [-1.0, 1.0]^d reference hypercube domain 
        SIMPLEX, // maps to the reference simplex with nodes at  the origin and 1.0 in each coordinate direction
        DYNAMIC // other domain types that will not output cleanly
    };
}
