/// @brief form a table of all of the Element Transformations in a given dimension
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/transformations/HypercubeTransformations.hpp"
#include "iceicle/transformations/SimplexElementTransformation.hpp"
namespace iceicle {

    template<class T, class IDX, int ndim>
    class ElementTransformationTable {
        public:
        std::array<ElementTransformation<T, IDX, ndim>, build_config::FESPACE_BUILD_GEO_PN + 1> hypercube_transforms;
        std::array<ElementTransformation<T, IDX, ndim>, build_config::FESPACE_BUILD_GEO_PN + 1> simplex_transforms;


        ElementTransformationTable(){
            NUMTOOL::TMP::constexpr_for_range<1, build_config::FESPACE_BUILD_GEO_PN + 1>(
                [&]<int order>{
                    hypercube_transforms[order] = ElementTransformation<T, IDX, ndim> {
                        .domain_type = DOMAIN_TYPE::HYPERCUBE,
                        .order = order,
                        .nnode = transformations::hypercube<T, IDX, ndim, order>::nnode,
                        .nfac = transformations::hypercube<T, IDX, ndim, order>::nfac,
                        .get_el_coord = transformations::hypercube<T, IDX, ndim, order>::get_el_coord,
                        .transform = transformations::hypercube<T, IDX, ndim, order>::transform,
                        .jacobian = transformations::hypercube<T, IDX, ndim, order>::jacobian,
                        .hessian = transformations::hypercube<T, IDX, ndim, order>::hessian,
                        .face_domain_type = transformations::hypercube<T, IDX, ndim, order>::face_domain_type,
                        .n_face_vert = transformations::hypercube<T, IDX, ndim, order>::n_face_vert,
                        .get_face_vert = transformations::hypercube<T, IDX, ndim, order>::get_face_vert,
                        .n_face_nodes = transformations::hypercube<T, IDX, ndim, order>::n_face_nodes,
                        .get_face_nodes = transformations::hypercube<T, IDX, ndim, order>::get_face_nodes,
                        .get_face_nr = transformations::hypercube<T, IDX, ndim, order>::get_face_nr,
                    };
                }
            );
            if constexpr (ndim == 2){
                simplex_transforms[1] = ElementTransformation<T, IDX, ndim> {
                    .domain_type = DOMAIN_TYPE::SIMPLEX,
                    .order = 1,
                    .nnode = 3,
                    .nfac = 3,
                    .get_el_coord = transformations::triangle<T, IDX>::get_el_coord,
                    .transform = transformations::triangle<T, IDX>::transform,
                    .jacobian = transformations::triangle<T, IDX>::jacobian,
                    .hessian = transformations::triangle<T, IDX>::hessian,
                    .face_domain_type = transformations::triangle<T, IDX>::face_domain_type,
                    .n_face_vert = transformations::triangle<T, IDX>::n_face_vert,
                    .get_face_vert = transformations::triangle<T, IDX>::get_face_vert,
                    .n_face_nodes = transformations::triangle<T, IDX>::n_face_nodes,
                    .get_face_nodes = transformations::triangle<T, IDX>::get_face_nodes,
                    .get_face_nr = transformations::triangle<T, IDX>::get_face_nr,
                };
            }
        }

        /// @brief get a pointer to the transformation for a given element (effectively the vtable pointer)
        ///
        /// @param domain_type the element domain type
        /// @param geo_order the polynomial order of the geometry of the element
        inline constexpr
        auto get_transform(DOMAIN_TYPE domain_type, int geo_order) noexcept
        -> ElementTransformation<T, IDX, ndim>* 
        {
            switch(domain_type){
                case DOMAIN_TYPE::HYPERCUBE:
                    return &hypercube_transforms[geo_order];

                case DOMAIN_TYPE::SIMPLEX:
                    return &simplex_transforms[geo_order];
                default: 
                    return nullptr;
            }
        }
    };

    template<class T, class IDX, int ndim>
    inline static ElementTransformationTable<T, IDX, ndim> transformation_table{};
}

