/**
 * @file face.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief geometric face definition
 */
#pragma once
#include <Numtool/point.hpp>
#include <Numtool/MathUtils.hpp>
#include <vector>
#include <string>

namespace ELEMENT {
    

    enum BOUNDARY_CONDITIONS{
        PERIODIC = 0,
        PARALLEL_COM,
        NEUMANN,
        DIRICHLET,
        RIEMANN,
        NO_SLIP,
        SLIP_WALL,
        WALL_GENERAL, /// General Wall BC, up to the implementation of the pde
        INLET,
        OUTLET,
        INITIAL_CONDITION,
        TIME_UPWIND, /// used for the top of a time slab
        INTERIOR // default condition that does nothing
    };

     /**
     * @brief An interface between two geometric elements
     * 
     * If this face bounday face
     * - real element is elemL
     * - ghost element is elemR
     * - ghost element is IDX -1 unless explicitly generated
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large lists
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim>
    class Face{
        
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        public:

        IDX elemL; /// the element on the left side of this face
        // If This face is a boundary face, then the real cell is the Left cell
        // The ghost cell is the right cell
        IDX elemR; /// the element on the right side of this face
        int faceNrL; /// the face number for the left element
        int faceNrR; /// the face number for the right element
        int orientationL; /// the orientation of this face wrt to the left element
        int orientationR; /// the orientation of this face wrt to te right element
        BOUNDARY_CONDITIONS bctype; /// the boundary condition type
        int bcflag; /// an integer flag to attach to the boundary condition

        explicit
        Face(IDX elemL, IDX elemR, int faceNrL, int faceNrR, int orientationL, int orientationR, BOUNDARY_CONDITIONS bctype = INTERIOR, int bcflag = 0)
         : elemL(elemL), elemR(elemR), faceNrL(faceNrL), faceNrR(faceNrR),
           orientationL(orientationL), orientationR(orientationR),
           bctype(bctype), bcflag(bcflag)
         {}

        ~Face() = default;

         /**
         * @brief Perform geometry precomputation
         * calculates normals, etc.
         * 
         * @param nodeCoords the node coordinate array
         */
        virtual
        void updateGeometry(std::vector< MATH::GEOMETRY::Point<T, ndim> > &nodeCoords) = 0;

        /**
         * @brief Get the area weighted normal at the given point s in the reference domain
         * always points from the left cell into the right cell
         * note: for boundary faces this means the normal always points
         *       out of the computational domain
         * 
         * @param [in] nodeCoords the node coordinates
         * @param [in] s the point in the face reference domain [ndim - 1]
         * @param [out] n The normal [ndim]
         */
        virtual
        void getNormal(
            std::vector< Point > &nodeCoords,
            const MATH::GEOMETRY::Point<T, ndim - 1> &s,
            T *n
        ) = 0;

        /**
         * @brief Get the area of the face
         * 
         * @return T the area of the face
         */
        virtual
        T getArea() = 0;

        /**
         * @brief Get the Unit Normal at the given point s
         * 
         * @param [in] nodeCoords the node coordinates
         * @param [in] s the point in the face reference domain
         * @param [out] result the result of the vector multiplication getnormal(s) / getArea();
         */
        virtual
        void getUnitNormal(
            std::vector< Point > &nodeCoords,
            const MATH::GEOMETRY::Point<T, ndim - 1> &s,
            T *result
        ){
            // naive implementation
            T aream = 1.0 / getArea();
            T normal[ndim];
            getNormal(nodeCoords, s, normal);
            for(int idim = 0; idim < ndim; ++idim) result[idim] = aream * normal[idim];
        }

        /**
         * @brief convert reference domain coordinates to physical coordinates
         *
         * @param [in] nodeCoords the node coordinates vector
         * @param [in] s the point in the face reference domain
         * @param [out] the physical coordinates size = ndim
         */
        virtual
        void convertRefToAct(
            std::vector< Point > &nodeCoords,
            const MATH::GEOMETRY::Point<T, ndim - 1> &s,
            T *result
        ) = 0;


        /**
         * @brief get the centroid of the face
         * @return const GOEMETRY::Point<T, ndim> & the center point of the face
         */
        virtual
        const Point &getCentroid() = 0;

        /**
         * @brief Square root of the Riemann metric 
         * of the face at the given point
         * 
         * @param s the point in the face reference domain
         * @return T the square root of the riemann metric
         */
        virtual
        T rootRiemannMetric(
            std::vector< Point > &nodeCoords,
            const MATH::GEOMETRY::Point<T, ndim - 1> &s
        ) = 0;

        /**
         * @brief the the number of nodes for this element
         * @return the number of nodes
         */
        virtual 
        int n_nodes() = 0;

        /**
         * @brief get a pointer to the array of node indices
         *
         * This array should be garuanteed to be in the same order as the reference degres of freedom
         * for the reference domain corresponding to this face
         * This is so nodal basis functions can be mapped to the global node dofs
         *
         * @return the node indices in the mesh nodes array
         */
        virtual
        IDX *nodes() = 0;

        virtual
        std::string printNodes(){
            std::string output = std::to_string(nodes()[0]);
            for(IDX inode = 1; inode < n_nodes(); ++inode){
                output += ", " + std::to_string(nodes()[inode]);
            }
            return output;
        }

    };
}
