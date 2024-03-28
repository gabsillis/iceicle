/**
 * @brief utilities for dealing with meshes
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <vector>

namespace MESH {

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
        std::vector<bool> is_boundary(mesh.nodes.n_nodes(), false);
        using Face = ELEMENT::Face<T, IDX, ndim>;
        for(Face *face : mesh.faces){

            // if its not an interior face, set all the node indices to true
            if(face->bctype != ELEMENT::BOUNDARY_CONDITIONS::INTERIOR){
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
        std::function< void(std::span<T, (std::size_t) ndim>, std::span<T, (std::size_t) ndim>) > &perturb_func,
        std::vector<bool> &fixed_nodes
    ) {
        for(IDX inode = 0; inode < mesh.nodes.n_nodes(); ++inode){
            if( true || !fixed_nodes[inode]){
                // copy current node data to prevent aliasing issues
                std::array old_node = mesh.nodes[inode].clone();
                std::span node_view = mesh.nodes[inode].to_span();

                // perturb the node given the current coordinates
                perturb_func(old_node, node_view);
            }
        }
    }

    namespace PERTURBATION_FUNCTIONS {

        /**
         * @brief perturb by following the Taylor Green Vortex 
         * centered at the middle of the given domain
         * at time = 1
         *
         * slowed down by the distance from the center
         */
        template<typename T, int ndim>
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

                    }

                    t += dt;
                }
                

            }
        };

        template<typename T, int ndim>
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
