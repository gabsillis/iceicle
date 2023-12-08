/**
 * @file face.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief geometric face definition
 */
#pragma once
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include <Numtool/point.hpp>
#include <Numtool/MathUtils.hpp>
#include <Numtool/fixed_size_tensor.hpp>
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

    /// face_info / this gives the face number 
    /// face_info % this gives the orientation
    static constexpr unsigned int FACE_INFO_MOD = 512;

     /**
     * @brief An interface between two geometric elements
     * 
     * If this face bounday face
     * - real element is elemL
     * - ghost element is elemR
     *
     * face_info: 
     * the face_info integers hold the local face number and orientation
     * that is used for transformations
     * The face number is face info / face_info_mod
     * The face orientation is face_info % face_info_mod
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large lists
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim>
    class Face{
        
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using FacePoint = MATH::GEOMETRY::Point<T, ndim - 1>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim - 1>;
        using MetricTensorType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim - 1, ndim - 1>;
        public:


        IDX elemL; /// the element on the left side of this face
        // If This face is a boundary face, then the real cell is the Left cell
        // The ghost cell is the right cell
        IDX elemR; /// the element on the right side of this face

        /// Face info for the left element
        unsigned int face_infoL;
        /// Face info for the right element 
        unsigned int face_infoR;
        BOUNDARY_CONDITIONS bctype; /// the boundary condition type
        int bcflag; /// an integer flag to attach to the boundary condition

        explicit
        Face(
            IDX elemL, IDX elemR, unsigned int face_infoL, unsigned int face_infoR,
            BOUNDARY_CONDITIONS bctype = INTERIOR, int bcflag = 0
        ) : elemL(elemL), elemR(elemR),
            face_infoL(face_infoL), face_infoR(face_infoR),
            bctype(bctype), bcflag(bcflag)
        {}

        virtual ~Face() = default;


        /**
         * @brief Get the area of the face
         * 
         * @return T the area of the face
         */
        virtual
        T getArea() const {
            // TODO: get rid of this, I hate this pattern
            throw std::logic_error("not implemented.");
        };

        /**
         * @brief convert reference domain coordinates to physical coordinates
         *
         * @param [in] nodeCoords the node coordinates vector
         * @param [in] s the point in the face reference domain
         * @param [out] the physical coordinates size = ndim
         */
        virtual
        void transform(
            FE::NodalFEFunction<T,ndim> &nodeCoords,
            const FacePoint &s,
            T *result
        ) const = 0;

        /**
         * @brief get the Jacobian matrix of the transformation
         * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}{\partial s}
         *
         * TODO: check this should always result in outward normals for 
         * the left element 
         *
         * @param [in] node_coords the coordinates of all the nodes 
         * @param [in] s the point in the global face reference domain 
         * @return the Jacobian matrix 
         */
        virtual 
        JacobianType Jacobian(
            FE::NodalFEFunction<T, ndim> &node_coords,
            const FacePoint &s
        ) const = 0;

        /**
         * @brief get the Riemannian metric tensor for the surface map 
         * @param [in] jac the jacobian as calculated by Jacobian()
         * @param [in] s the point in the global face reference domain 
         * @return the Riemannian metric tensor
         */
        virtual 
        MetricTensorType RiemannianMetric(
            JacobianType &jac,
            const FacePoint &s
        ) const {
            MetricTensorType g;
            if constexpr (ndim == 1) {
                g[0][0] = 1.0;
                return g;
            }

            g = 0.0;
            for(int k = 0; k < ndim - 1; ++k){
                for(int l = 0; l < ndim - 1; ++l){
                    for(int i = 0; i < ndim; ++i){
                        g[k][l] += jac[i][k] * jac[i][l];
                    }
                }
            }
            return g;
        }
        
        /**
         * @brief Square root of the Riemann metric determinant
         * of the face at the given point
         *
         * @param jac the jacobian as calculated by Jacobian()
         * @param s the point in the face reference domain
         * @return T the square root of the riemann metric
         */
        virtual
        T rootRiemannMetric(
            JacobianType &jac,
            const FacePoint &s
        ) const {
           MetricTensorType g = RiemannianMetric(jac, s); 
           return std::sqrt(NUMTOOL::TENSOR::FIXED_SIZE::determinant(g));
        }

        /**
         * @brief the the number of nodes for this element
         * @return the number of nodes
         */
        virtual 
        int n_nodes() const = 0;

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
