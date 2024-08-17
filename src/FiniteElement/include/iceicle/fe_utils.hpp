/**
 * @brief general utility functions that operate on 
 * objects in the FiniteElement library
 */

#include "Numtool/point.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/geo_element.hpp"

#include <random>

#pragma once
namespace iceicle {

    /**
     * @brief given an element generate a random point in the reference domain 
     * TODO: not uniform for simplex
     * @param geo_el the geometric element 
     */
    template<typename T, typename IDX, int ndim>
    MATH::GEOMETRY::Point<T, ndim> random_domain_point(
        const GeometricElement<T, IDX, ndim> *geo_el
    ){
        static std::random_device rdev{};
        static std::default_random_engine engine{rdev()};
        static std::uniform_real_distribution<T> hypercube_dist{(T) -1.0, (T) 1};
        static std::uniform_real_distribution<T> simplex_dist{(T) 0.0, (T) 1};

        MATH::GEOMETRY::Point<T, ndim> randpt{};
        switch(geo_el->domain_type()){
            case DOMAIN_TYPE::HYPERCUBE:
                for(int idim = 0; idim < ndim; ++idim){
                    randpt[idim] = hypercube_dist(engine);
                }
                return randpt;

            case DOMAIN_TYPE::SIMPLEX: 
                // NOTE: this hypercube is [0, 1]^n
                T hypercube_pt[ndim]; 
                for(int idim = 0; idim < ndim; ++idim) 
                    { hypercube_pt[idim] = simplex_dist(engine); }

                // transform to simplex 
                for(int idim = 0; idim < ndim; ++idim){
                    randpt[idim] = hypercube_pt[idim];
                    for(int jdim = idim + 1; jdim < ndim; ++jdim){
                        randpt[idim] *= (1 - hypercube_pt[jdim]);
                    }
                }
                return randpt;

            default:
                break;
        }
        return randpt;
    }
}
