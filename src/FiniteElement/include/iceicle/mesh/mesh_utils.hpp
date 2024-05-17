/**
 * @brief utilities for dealing with meshes
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/basis/tensor_product.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/geometry/geo_primitives.hpp"
#include <random>
#include <vector>
#include <concepts>

namespace iceicle {


    /// @brief find and create all the interior faces for a mesh
    template<class T, class IDX, int ndim>
    auto find_interior_faces(
        AbstractMesh<T, IDX, ndim>& mesh
    ) {
        using namespace util;
        // if elements share at least ndim points, then they have a face
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            int max_faces = mesh.elements[ielem]->n_faces();
            int found_faces = 0;
            for(IDX jelem = ielem + 1; jelem < mesh.nelem(); ++jelem){
                auto face_opt = make_face(ielem, jelem, mesh.elements[ielem], mesh.elements[jelem]);
                if(face_opt){
                    mesh.faces.push_back(face_opt.value());
                    // short circuit if all the faces have been found
                    ++found_faces;
                    if(found_faces == max_faces) break;
                }
            }
        }
    }

    /**
     * @brief create a 2 element mesh with no boundary faces 
     * This is good for testing numerical fluxes 
     * @param centroid1 the centroid of the first element 
     * @param centroid2 the centroid of the second element
     */
    template<class T, int ndim>
    auto create_2_element_mesh(
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> centroid1,
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> centroid2,
        BOUNDARY_CONDITIONS bctype,
        int bcflag
    ) -> AbstractMesh<T, int, ndim>
    {
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;

        T dist = 0;
        for(int idim = 0; idim < ndim; ++idim) dist += SQUARED(centroid2[idim] - centroid1[idim]);
        dist = std::sqrt(dist);

        T el_radius = 0.5 * dist;

        // setup the mesh 
        AbstractMesh<T, int, ndim> mesh{};

        // Create the nodes 
        mesh.nodes.resize(3 * std::pow(2, ndim - 1));

        QTypeIndexSet<int, ndim - 1, 2> node_positions{};
        for(int ipoin = 0; ipoin < node_positions.size(); ++ipoin){
            auto& mindex = node_positions[ipoin];

            // positions in a reference domain where centroid1 is [0, 0, ..., 0]
            // and centroid 2 is at [0, 0, ..., dist]
            std::array<T, ndim> position_reference1{};
            std::array<T, ndim> position_reference2{};
            std::array<T, ndim> position_reference3{};
            for(int idim = 0; idim < ndim -1 ; ++idim){
                T idim_pos = (mindex[idim] == 0) ? (-el_radius) : (el_radius);
                T position_reference1[idim] = idim_pos;
                T position_reference2[idim] = idim_pos;
                T position_reference3[idim] = idim_pos;
            }
            position_reference1[ndim - 1] = -el_radius;
            position_reference2[ndim - 1] = el_radius;
            position_reference3[ndim - 1] = dist + el_radius;
        }

        // TODO: rotation matrix

        // Generate the elements 
        
    }

    /**
     * @brief for every node provide a boolean flag for whether 
     * that node is on a boundary or not 
     *
     * @param mesh the mesh 
     * @return a bool vector of the size of the number of nodes of the mesh 
     *         true if the node at that index is on the boundary 
     */
    template<class T, class IDX, int ndim>
    std::vector<bool> flag_boundary_nodes(AbstractMesh<T, IDX, ndim> &mesh){
        std::vector<bool> is_boundary(mesh.n_nodes(), false);
        using Face = Face<T, IDX, ndim>;
        for(Face *face : mesh.faces){

            // if its not an interior face, set all the node indices to true
            if(face->bctype != BOUNDARY_CONDITIONS::INTERIOR){
                for(IDX node : face->nodes_span()){
                    is_boundary[node] = true;
                }
            }
        }
        return is_boundary;
    }

    /**
     * @brief perturb all the non-fixed nodes 
     *        according to a given perturbation function 
     *
     * @param mesh the mesh 
     * @param perturb_func the function to perturb the node coordinates 
     *        arg 1: [in] span which represents the current node coordinates 
     *        arg 2: [out] span which represents the perturbed coordinates 
     *
     * @param fixed_nodes vector of size n_nodes which is true if the node should not move
     */
    template<class T, class IDX, int ndim>
    void perturb_nodes(
        AbstractMesh<T, IDX, ndim> &mesh, 
        std::invocable<
            std::span<T, (std::size_t) ndim>,
            std::span<T, (std::size_t) ndim>
        > auto& perturb_func,
        std::vector<bool> &fixed_nodes
    ) {
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        for(IDX inode = 0; inode < mesh.n_nodes(); ++inode){
            if( true || !fixed_nodes[inode]){
                // copy current node data to prevent aliasing issues
                Point old_node = mesh.nodes[inode];
                std::span<T, ndim> node_view{mesh.nodes[inode].begin(), mesh.nodes[inode].end()};

                // perturb the node given the current coordinates
                perturb_func(old_node, node_view);
            }
        }
    }


    /// @brief compute the bounding box of the mesh
    /// by nodes
    template<class T, class IDX, int ndim>
    auto compute_bounding_box(
        AbstractMesh<T, IDX, ndim> &mesh
    ) -> BoundingBox<T, ndim> {
        BoundingBox<T, ndim> bbox{};
        std::ranges::fill(bbox.xmin, 1e100);
        std::ranges::fill(bbox.xmax, -1e100);
        for(IDX inode = 0; inode < mesh.n_nodes(); ++inode){
            for(int idim = 0; idim < ndim; ++idim){
                bbox.xmin[idim] = std::min(bbox.xmin[idim], mesh.nodes[inode][idim]);
                bbox.xmax[idim] = std::max(bbox.xmax[idim], mesh.nodes[inode][idim]);
            }
        }
        return bbox;
    }

    namespace PERTURBATION_FUNCTIONS {

        /// @brief randomly perturb the nodes 
        /// in a given range
        /// coord = coord + random in [min_perturb, max_perturb]
        template<typename T, std::size_t ndim>
        struct random_perturb {

            std::random_device rdev;
            std::default_random_engine engine;
            std::uniform_real_distribution<T> dist;

            /// @brief constructor
            /// @param min_perturb the minimum of the range 
            /// @param max_perturb the maximum of the range
            random_perturb(T min_perturb, T max_perturb)
            : rdev{}, engine{rdev()}, dist{min_perturb, max_perturb} {}

            random_perturb(const random_perturb<T, ndim> &other) = default;
            random_perturb(random_perturb<T, ndim> &&other) = default;

            void operator()(std::span<T, ndim> xin, std::span<T, ndim> xout) {
                std::ranges::copy(xin, xout.begin());

                for(int idim = 0; idim < ndim; ++idim){
                    xout[idim] += dist(engine);
                }
            }
        };

        /**
         * @brief perturb by following the Taylor Green Vortex 
         * centered at the middle of the given domain
         * at time = 1
         *
         * slowed down by the distance from the center
         */
        template<typename T, std::size_t ndim>
        struct TaylorGreenVortex {

            /// @brief the velocity of the vortex 
            T v0 = 1.0;

            /// @brief the min corner of the domain
            std::array<T, ndim> xmin;

            /// @brief the max corner of the domain
            std::array<T, ndim> xmax;

            /// @brief the length scale 
            /// (1 means that one vortex will cover the entire domain)
            T L = 1.0;
        
            void operator()(std::span<T, ndim> xin, std::span<T, ndim> xout){
                // copy over the input data
                std::copy(xin.begin(), xin.end(), xout.begin());

                // calculate the domain center 
                std::array<T, ndim> center;
                for(int idim = 0; idim < ndim; ++idim) 
                    center[idim] = (xmin[idim] + xmax[idim]) / 2.0;

                // max length of the domain 
                T domain_len = 0.0;
                for(int idim = 0; idim < ndim; ++idim){
                    domain_len = std::max(domain_len, xmax[idim] - xmin[idim]);
                }

                T dt = 1.0 / (100); // TODO: some kind of cfl condition
                T t = 0.0;

                // preturb with explicit timestepping
                while(t < 1.0){
                    dt = std::min(dt, 1.0 - dt);
                    switch(ndim) {
                        case 2:
                            {
                            T x = (xout[0] - center[0]) / domain_len;
                            T y = (xout[1] - center[1]) / domain_len;

                            T center_dist = x*x + y*y;
                            T mult = v0 * std::exp(-center_dist / 0.3);

                            T u =  mult * std::cos(L * M_PI * x) * std::sin(L * M_PI * y);
                            T v = -mult * std::sin(L * M_PI * x) * std::cos(L * M_PI * y);

                            xout[0] += dt * u;
                            xout[1] += dt * v;
                            }
                            break;

                        case 3:
                            {
                            T x = (xout[0] - center[0]) / domain_len;
                            T y = (xout[1] - center[1]) / domain_len;
                            T z = (xout[2] - center[2]) / domain_len;

                            T center_dist = x*x + y*y + z*z;
                            T mult = v0 * std::exp(-center_dist / 0.5);

                            T u =  mult * std::cos(L * M_PI * x) * std::sin(L * M_PI * y) * sin(L * M_PI * z);
                            T v = -mult * std::sin(L * M_PI * x) * std::cos(L * M_PI * y) * sin(L * M_PI * z);
                            T w =  mult * std::sin(L * M_PI * x) * std::sin(L * M_PI * y) * cos(L * M_PI * z);

                            xout[0] += dt * u;
                            xout[1] += dt * v;
                            xout[2] += dt * w;
                            }
                            break;
                        default:
                            // NOTE: just use the 3D version
                            {
                            T x = (xout[0] - center[0]) / domain_len;
                            T y = (xout[1] - center[1]) / domain_len;
                            T z = (xout[2] - center[2]) / domain_len;

                            T center_dist = x*x + y*y + z*z;
                            T mult = v0 * std::exp(-center_dist / 0.5);

                            T u =  mult * std::cos(L * M_PI * x) * std::sin(L * M_PI * y) * sin(L * M_PI * z);
                            T v = -mult * std::sin(L * M_PI * x) * std::cos(L * M_PI * y) * sin(L * M_PI * z);
                            T w =  mult * std::sin(L * M_PI * x) * std::sin(L * M_PI * y) * cos(L * M_PI * z);

                            xout[0] += dt * u;
                            xout[1] += dt * v;
                            xout[2] += dt * w;
                            }
                            break;
                    }

                    t += dt;
                }
                

            }
        };

        template<typename T, std::size_t ndim>
        struct ZigZag {
            static_assert(ndim >= 2, "Must be at least 2 dimensional.");
            void operator()(std::span<T, ndim> xin, std::span<T, ndim> xout){
                T xp = xin[0];
                T yp = xin[1];

                // keep the x coordinate
                xout[0] = xin[0];

                // for each end of each segment 
                // a represents where y = 0.5 ends up
                T a1 = 0.3;
                T a2 = 0.3;
                T yout1, yout2, xref;

                // zig and zag the y coordinate
                // get ziggy with it
                if(xp < 0.2){
                    if(yp < 0.5){
                        xout[1] = yp * 0.3 / 0.5;
                    } else {
                        xref = (xp - 0.0) / 0.2;
                        xout[1] = ( (1.39 + 0.01 * (xref)) * (yp - 1.0) + 1);
                    }
                } else if(xp < 0.4){
                    a1 = 0.3;
                    a2 = 0.7;
                    xref = (xp - 0.2) / 0.2;
                    if(yp < 0.5){
                        yout1 = yp * a1 / 0.5;
                        yout2 = yp * a2 / 0.5;
                    } else {
                        yout1 = 2 *(1 - a1) * (yp - 1) + 1;
                        yout2 = 2 *(1 - a2) * (yp - 1) + 1;
                    }
                    xout[1] = xref * yout2 + (1 - xref) * yout1;
                } else if (xp < 0.6) {
                    a1 = 0.7;
                    a2 = 0.3;
                    xref = (xp - 0.4) / 0.2;
                    if(yp < 0.5){
                        yout1 = yp * a1 / 0.5;
                        yout2 = yp * a2 / 0.5;
                    } else {
                        yout1 = 2 *(1 - a1) * (yp - 1) + 1;
                        yout2 = 2 *(1 - a2) * (yp - 1) + 1;
                    }
                    xout[1] = xref * yout2 + (1 - xref) * yout1;
                } else if (xp < 0.8) {
                    a1 = 0.3;
                    a2 = 0.7;
                    xref = (xp - 0.6) / 0.2;
                    if(yp < 0.5){
                        yout1 = yp * a1 / 0.5;
                        yout2 = yp * a2 / 0.5;
                    } else {
                        yout1 = 2 *(1 - a1) * (yp - 1) + 1;
                        yout2 = 2 *(1 - a2) * (yp - 1) + 1;
                    }
                    xout[1] = xref * yout2 + (1 - xref) * yout1;
                } else {
                    a1 = 0.7;
                    a2 = 0.7;
                    xref = (xp - 0.8) / 0.2;
                    if(yp < 0.5){
                        yout1 = yp * a1 / 0.5;
                        yout2 = yp * (a2 - 0.01) / 0.5;
                    } else {
                        yout1 = 2 *(1 - a1) * (yp - 1) + 1;
                        yout2 = 2 *(1 - a2) * (yp - 1) + 1;
                    }
                    xout[1] = xref * yout2 + (1 - xref) * yout1;
                }
            }
        };
    }
}
