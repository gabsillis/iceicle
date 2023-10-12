/**
 * @file simplex_element.hpp
 * @brief GeometricElement implementation for simplices
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

namespace ELEMENT {
    
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexGeoElement {
        
        public:
        TRANSFORMATIONS::SimplexElementTransformation<T, IDX, ndim, Pn> transform;
        
    };
}
