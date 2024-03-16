/**
 * @file face.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief geometric face definition
 */
#pragma once
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include <iceicle/geometry/geometry_enums.hpp>
#include <iceicle/string_utils.hpp>
#include <Numtool/point.hpp>
#include <Numtool/MathUtils.hpp>
#include <iceicle/fe_enums.hpp>
#include <Numtool/fixed_size_tensor.hpp>
#include <string>
#include <span>
#include <string_view>

namespace ELEMENT {
    
    enum BOUNDARY_CONDITIONS {
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
        SPACETIME_PAST, /// used for the bottom of a time slab
        SPACETIME_FUTURE, /// used for the top of a time slab
        INTERIOR // default condition that does nothing
    };

    /// @brief get a human readable name for each boundary condition
    constexpr const inline char* bc_name(const BOUNDARY_CONDITIONS bc) noexcept {
        const char *names[] = {
            "Periodic",
            "Parallel_Communication",
            "Neumann",
            "Dirichlet",
            "Riemann Solver (Characteristic)",
            "No slip",
            "Slip wall",
            "General Wall",
            "Inlet",
            "Outlet",
            "spacetime past",
            "spacetime future",
            "Interior face (NO BC)"
        };
        return names[(int) bc];
    }

    constexpr inline BOUNDARY_CONDITIONS get_bc_from_name(std::string_view bcname){
        using namespace ICEICLE::UTIL;

        if(eq_icase(bcname, "dirichlet")){
            return DIRICHLET;
        }

        if(eq_icase(bcname, "neumann")){
            return NEUMANN;
        }

        return INTERIOR;
    }

    /// face_info / this gives the face number 
    /// face_info % this gives the orientation
    static constexpr unsigned int FACE_INFO_MOD = 512;

    /**
     * Provides an interface to the face type specific 
     * bookkeeping utilities in a generic interface 
     */
    template<class T, class IDX, int ndim>
    struct FaceInfoUtils {

        /**
         * @brief get the number of vertices 
         * A vertex is an extreme point on the face
         * WARNING: This does not necesarily include all nodes 
         * which can be on interior features
         */
        virtual
        int n_face_vertices() = 0;

        /**
         * @brief get the global indices of the vertices in order
         * given a face number and element vertices
         *
         * A vertex is an extreme point on the face
         * WARNING: This does not necesarily include all nodes 
         * which can be on interior features
         *
         * @param [in] face_nr the face number
         * @param [in] element_nodes the global node numbers for the element nodes
         * @param [out] face_vertices the vertex global indices for the face
         */
        virtual
        void get_face_vertices(
            int face_nr,
            IDX *element_nodes,
            std::span<IDX> face_vertices
        ) = 0;

        /**
         * @brief get the orientation of the right face given the vertices 
         * of the left and right face 
         * 
         * @param face_vertices_l the vertices of the face in order on the left
         * @param face_vertices_r the vertices of the face in order on the right 
         * @return the orientation of the right face 
         */
        virtual 
        int get_orientation(
            std::span<IDX> face_vertices_l,
            std::span<IDX> face_vertices_r
        ) = 0;

    };

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
        template<typename T1, std::size_t... sizes>
        using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T1, sizes...>;
       
        protected:

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
        IDX bcflag; /// an integer flag to attach to the boundary condition

        explicit
        Face(
            IDX elemL, IDX elemR, unsigned int face_infoL, unsigned int face_infoR,
            BOUNDARY_CONDITIONS bctype = INTERIOR, int bcflag = 0
        ) : elemL(elemL), elemR(elemR),
            face_infoL(face_infoL), face_infoR(face_infoR),
            bctype(bctype), bcflag(bcflag)
        {}

        virtual ~Face() = default;

        /** @brief get the shape that defines the reference domain */
        virtual 
        constexpr FE::DOMAIN_TYPE domain_type() const = 0;

        /** @brief get the geometry polynomial order */
        virtual 
        constexpr int geometry_order() const noexcept = 0;

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
         * @brief transform from the reference domain coordinates 
         * to the physical domain 
         * @param [in] s the point in the face reference domain 
         * @param [in] coord the node coordinates
         * @param [out] result the position in the physical domain 
         */
        virtual 
        void transform(
            const FacePoint &s,
            FE::NodalFEFunction<T, ndim> &coord, 
            T *result
        ) const = 0;

        /**
         * @brief convert reference domain coordinates to 
         * the left element reference domain
         *
         * @param [in] s the point in the face reference domain
         * @param [out] the physical coordinates size = ndim
         */
        virtual
        void transform_xiL(
            const FacePoint &s,
            T *result
        ) const = 0;

        /**
         * @brief convert reference domain coordinates to 
         * the right element reference domain
         *
         * @param [in] s the point in the face reference domain
         * @param [out] the physical coordinates size = ndim
         */
        virtual 
        void transform_xiR(
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

        /**
         * @brief get the array of node indices as a span 
         *
         * This array should be garuanteed to be in the same order as the reference degres of freedom
         * for the reference domain corresponding to this face
         * This is so nodal basis functions can be mapped to the global node dofs
         *
         * @return the node indices in the mesh nodes array
         */
        std::span<IDX> nodes_span(){
            return std::span{nodes(), static_cast<std::size_t>(n_nodes())};
        }

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
