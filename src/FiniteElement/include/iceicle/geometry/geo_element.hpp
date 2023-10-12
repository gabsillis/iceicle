/**
 * @file geo_element.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Abstract definition for Geometric Elements
 * @date 2023-06-26
 */

#pragma once
#include <vector>
#include <Numtool/point.hpp>
#include <Numtool/matrixT.hpp>

namespace ELEMENT{
    
    /**
     * @brief A Geometric element
     * Contains information and methods for geometric description of elements
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large lists
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim> 
    class GeometricElement {
        private:

        // namespace aliases
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        /**
         * @brief Get the number of nodes
         * 
         * @return int the number of nodes for this element
         */
        virtual
        constexpr int n_nodes() = 0;

        /**
         * @brief Get the nodes array for this element
         * 
         * @return std::ptrdiff_t* the array of nodes
         */
        virtual
        IDX *nodes() = 0;

        /**
         * @brief Get the location of the centroid of the element
         * @param [in] nodeCoords the node coordinates
         * @param [out] centroid the centroid of the element [ndim]
         */
        virtual
        void getCentroid(const std::vector< Point > &nodeCoords, T *centroid) = 0;

         /**
         * @brief Perform geometry precomputation
         * calculates normals, etc.
         * 
         * @param nodeCoords the node coordinate array
         */
        virtual
        void updateGeometry(std::vector< Point > &nodeCoords) = 0;

    };
}
