/**
 * @brief utilities for dealing with meshes
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "Numtool/point.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/geometry/geo_primitives.hpp"
#include <random>
#include <vector>

namespace iceicle {

    /// @brief find and create all the interior faces for a mesh
    template<class T, class IDX, int ndim>
    auto find_interior_faces(
        AbstractMesh<T, IDX, ndim>& mesh
    ) {
        using namespace util;

        // elements surrounding points
        std::vector<std::vector<IDX>> elsup(mesh.n_nodes());
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            for(IDX inode : mesh.conn_el.rowspan(ielem)){
                elsup[inode].push_back(ielem);
            }
        }

        // remove duplicates and sort
        for(IDX inode = 0; inode < mesh.n_nodes(); ++inode){
            std::ranges::sort(elsup[inode]);
            auto unique_subrange = std::ranges::unique(elsup[inode]);
            elsup[inode].erase(unique_subrange.begin(), unique_subrange.end());
        }

        // if elements share at least ndim points, then they have a face
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            int max_faces = mesh.el_transformations[ielem]->nfac;
            std::vector<IDX> connected_elements;
            connected_elements.reserve(max_faces);

            // loop through elements that share a node
            for(IDX inode : mesh.conn_el.rowspan(ielem)){
                for(auto jelem_iter = std::lower_bound(elsup[inode].begin(), elsup[inode].end(), ielem);
                        jelem_iter != elsup[inode].end(); ++jelem_iter){
                    IDX jelem = *jelem_iter;

                    // skip the cases that would lead to duplicate or boundary faces
                    if( ielem == jelem || std::ranges::find(connected_elements, jelem) != std::ranges::end(connected_elements))
                        continue; 

                    // try making the face that is the intersection of the two elements
                    auto face_opt = make_face(ielem, jelem, 
                            mesh.el_transformations[ielem], mesh.el_transformations[jelem],
                            mesh.conn_el.rowspan(ielem), mesh.conn_el.rowspan(jelem));
                    if(face_opt){
                        mesh.faces.push_back(std::move(face_opt.value()));
                        // short circuit if all the faces have been found
                        connected_elements.push_back(jelem);
                        if(connected_elements.size() == max_faces) break;
                    }
                }
            }
        }
    }

    /// @brief form a mixed uniform mesh with square and triangle elements
    ///
    /// @param nelem the number of (quad) elements in each direction 
    /// @param xmin the minimum point of the bounding box 
    /// @param xmax the maximum point of the bounding box 
    /// @param quad_ratio the percentage ratio of quads to tris
    /// @param bcs the boundary conditions (left, bottom, right, top)
    /// @param the boundary condition flags (left, bottom, right, top)
    template<class T, class IDX>
    auto mixed_uniform_mesh(
        std::span<IDX> nelem,
        std::span<T> xmin,
        std::span<T> xmax,
        std::span<T> quad_ratio,
        std::span<BOUNDARY_CONDITIONS> bcs,
        std::span<int> bcflags
    ) -> std::optional<AbstractMesh<T, IDX, 2>> {
        using Point = MATH::GEOMETRY::Point<T, 2>;
        using boundary_face_desc = std::tuple<BOUNDARY_CONDITIONS, int, std::vector<IDX>>;
        using namespace util;

        NodeArray<T, 2> coord;

        // make the nodes
        T dx = (xmax[0] - xmin[0]) / (nelem[0]);
        T dy = (xmax[1] - xmin[1]) / (nelem[1]);
        IDX nnodex = nelem[0] + 1;
        IDX nnodey = nelem[1] + 1;
        for(IDX iy = 0; iy < nnodey; ++iy){
            for(IDX ix = 0; ix < nnodex; ++ix){
                coord.push_back(Point{{xmin[0] + ix * dx, xmin[1] + iy * dy}});
}
        }

        IDX half_quad_x = (IDX) (nelem[0] * quad_ratio[0] / 2);
        IDX half_quad_y = (IDX) (nelem[1] * quad_ratio[1] / 2);

        std::vector< ElementTransformation<T, IDX, 2>* > el_transformations;
        std::vector<std::vector<IDX>> el_conn{};
        // make the elements 
        for(IDX ixquad = 0; ixquad < nelem[0]; ++ixquad){
            for(IDX iyquad = 0; iyquad < nelem[1]; ++iyquad){
                IDX bottomleft = iyquad * nnodex + ixquad;
                IDX bottomright = iyquad * nnodex + ixquad + 1;
                IDX topleft = (iyquad + 1) * nnodex + ixquad;
                IDX topright = (iyquad + 1) * nnodex + ixquad + 1;

                bool is_quad = ixquad < half_quad_x 
                    || (nelem[0] - ixquad) <= half_quad_x
                    || iyquad < half_quad_y 
                    || (nelem[1] - iyquad) <= half_quad_y;

                if(is_quad){
                    el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::HYPERCUBE, 1));
                    std::vector<IDX> nodes{bottomleft, topleft, bottomright, topright};
                    el_conn.push_back(nodes);
                } else {
                    std::vector<IDX> nodes1{bottomleft, bottomright, topleft};
                    std::vector<IDX> nodes2{topleft, bottomright, topright};
                    el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::SIMPLEX, 1));
                    el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::SIMPLEX, 1));
                    el_conn.push_back(nodes1);
                    el_conn.push_back(nodes2);

                }
            }
        }

        // update the element crs matrices 
        util::crs<IDX, IDX> conn_el{el_conn};

        // make array of descriptions of boundary faces
        std::vector<boundary_face_desc> bdy_descriptions{};
        bdy_descriptions.reserve(2 * (nelem[0] + nelem[1]));
        for(IDX ixquad = 0; ixquad < nelem[0]; ++ixquad){
            // bottom face
            bdy_descriptions.emplace_back(bcs[1], bcflags[1], std::vector<IDX>{ixquad, ixquad + 1});

            // top face
            bdy_descriptions.emplace_back(bcs[3], bcflags[3], 
                    std::vector<IDX>{nelem[1] * nnodex + ixquad, nelem[1] * nnodex + ixquad + 1});
        }
        for(IDX iyquad = 0; iyquad < nelem[1]; ++iyquad) {
            // left face
            bdy_descriptions.emplace_back(bcs[0], bcflags[0], 
                std::vector<IDX>{iyquad * nnodex, (iyquad + 1) * nnodex});

            // right face
            bdy_descriptions.emplace_back(bcs[2], bcflags[2], 
                std::vector<IDX>{iyquad * nnodex + nelem[0], (iyquad + 1) * nnodex + nelem[0]});
        }
        
        return std::optional{AbstractMesh{coord, conn_el, el_transformations, bdy_descriptions}};
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
        for(auto& face : mesh.faces){

            // if its not an interior face, set all the node indices to true
            if(face->bctype != BOUNDARY_CONDITIONS::INTERIOR){
                for(IDX node : face->nodes_span()){
                    is_boundary[node] = true;
                }
            }
        }
        return is_boundary;
    }

    /// @brief check that all the normals are facing the right direction 
    /// (normal at the face centroid should be generally pointing from 
    /// the centroid of the face, to the centroid of the left element, 
    /// tested with dot product)
    /// @param [in] mesh the mesh to test 
    /// @param [out] invalid_faces the list of invalid faces gets built (pass in an empty vector)
    /// @return true if no invalid faces are in the list at the end of processing, false otherwise
    template<class T, class IDX, int ndim>
    auto validate_normals(AbstractMesh<T, IDX, ndim> &mesh, std::vector<IDX>& invalid_faces) -> bool {
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
        for(IDX ifac = mesh.interiorFaceStart; ifac < mesh.interiorFaceEnd; ++ifac){
            auto faceptr = mesh.faces[ifac].get();
            auto centroid_fac = face_centroid(*faceptr, mesh.coord);
            auto centroid_l = mesh.el_transformations[faceptr->elemL]
                ->centroid(mesh.get_el_coord(faceptr->elemL));
            auto centroid_r = mesh.el_transformations[faceptr->elemR]
                ->centroid(mesh.get_el_coord(faceptr->elemR));
            Tensor<T, ndim> internal_l, internal_r;
            for(int idim = 0; idim < ndim; ++idim){
                internal_l[idim] = centroid_l[idim] - centroid_fac[idim];
                internal_r[idim] = centroid_r[idim] - centroid_fac[idim];
            }
            // TODO: generalize face centroid ref domain 
            MATH::GEOMETRY::Point<T, ndim - 1> s;
            for(int idim = 0; idim < ndim - 1; ++idim) s[idim] = 0.0; 
            auto normal = calc_normal(*faceptr, mesh.coord, s);
            if(dot(normal, internal_l) > 0.0 || dot(normal, internal_r) < 0.0){
                invalid_faces.push_back(ifac);
            }
        }

        for(IDX ifac = mesh.bdyFaceStart; ifac < mesh.bdyFaceEnd; ++ifac){
            auto faceptr = mesh.faces[ifac].get();
            auto centroid_fac = face_centroid(*faceptr, mesh.coord);
            auto centroid_l = mesh.el_transformations[faceptr->elemL]
                ->centroid(mesh.get_el_coord(faceptr->elemL));
            Tensor<T, ndim> internal_l, internal_r;
            for(int idim = 0; idim < ndim; ++idim){
                internal_l[idim] = centroid_l[idim] - centroid_fac[idim];
            }
            // TODO: generalize face centroid ref domain 
            MATH::GEOMETRY::Point<T, ndim - 1> s;
            for(int idim = 0; idim < ndim - 1; ++idim) s[idim] = 0.0; 
            auto normal = calc_normal(*faceptr, mesh.coord, s);
            if(dot(normal, internal_l) > 0.0){
                invalid_faces.push_back(ifac);
            }
        }
        return invalid_faces.size() == 0;
    }

    template<class T, class IDX>
    auto edge_swap(AbstractMesh<T, IDX, 2>& mesh, IDX ifac)
    -> void {
        IDX elemL = mesh.faces[ifac]->elemL;
        IDX elemR = mesh.faces[ifac]->elemR;
        ElementTransformation<T, IDX, 2> *transL = mesh.el_transformations[elemL];
        ElementTransformation<T, IDX, 2> *transR = mesh.el_transformations[elemR];
        const Face<T, IDX, 2>& old_face = *(mesh.faces[ifac]);

        if(old_face.bctype != BOUNDARY_CONDITIONS::INTERIOR) [[unlikely]]
            util::AnomalyLog::log_anomaly("Cannot edge swap a boundary face.");

        // NOTE: only defined for triangles 
        if(transL->domain_type == DOMAIN_TYPE::SIMPLEX
            && transR->domain_type == DOMAIN_TYPE::SIMPLEX) {

            if(transL->order != 1 || transR->order != 1) [[unlikely]] {
                util::AnomalyLog::log_anomaly("Not implemented");
                return;
            }

            //           c
            //           O
            //         . . .
            //       .   .  .
            //   a O   L .   .
            //      .    . R  O d
            //       .   .   .
            //         . . .
            //           O
            //           b
            //
            // CCW abc 

            // Flip to
            //           c
            //           O
            //         .   .
            //       .   R  .
            //   a O  . .     .
            //      .    . . . O d
            //       .  L    .
            //         .   .
            //           O
            //           b
            //
            // Local nodes: 
            //  elemL : {b, d, a}
            //  elemR : {c, d, a}
            // WARNING: assumption of face number and node structure for simplex element implementation
            // Assumes nodes are ordered vertices first
            // Assumes face number is the local node index of the vertex not on the face
            
            // local face numbers
            int alL = old_face.face_nr_l();
            int blL = (alL + 1) % 3;
            int clL = (blL + 1) % 3;

            int blR, clR;
            int dlR = old_face.face_nr_r();
            if(old_face.orientation_r() == 0){
                blR = (dlR + 1) % 3;
                clR = (blR + 1) % 3;
            } else {
                clR = (dlR + 1) % 3;
                blR = (clR + 1) % 3;

            }

            // global face numbers
            IDX a = mesh.conn_el[elemL, alL];
            IDX b = mesh.conn_el[elemL, blL];
            IDX c = mesh.conn_el[elemL, clL];
            IDX d = mesh.conn_el[elemR, dlR];


            // update element connectivity
            mesh.conn_el[elemL, 0] = b;
            mesh.conn_el[elemL, 1] = d;
            mesh.conn_el[elemL, 2] = a;
            mesh.coord_els[elemL, 0] = mesh.coord[b];
            mesh.coord_els[elemL, 1] = mesh.coord[d];
            mesh.coord_els[elemL, 2] = mesh.coord[a];

            mesh.conn_el[elemR, 0] = c;
            mesh.conn_el[elemR, 1] = d;
            mesh.conn_el[elemR, 2] = a;
            mesh.coord_els[elemR, 0] = mesh.coord[c];
            mesh.coord_els[elemR, 1] = mesh.coord[d];
            mesh.coord_els[elemR, 2] = mesh.coord[a];

            // === make the new faces ===
            
            // shared face 
            {
                std::array<IDX, 2> face_nodes{d, a};
                // chosen to be these
                int face_nr_l = 0;
                int face_nr_r = 0;
                int orient_r = 0;
                auto face_opt = make_face<T, IDX, 2>(
                        DOMAIN_TYPE::HYPERCUBE, DOMAIN_TYPE::SIMPLEX, DOMAIN_TYPE::SIMPLEX,
                        1, elemL, elemR, std::span<const IDX>{face_nodes}, 
                        face_nr_l, face_nr_r, orient_r);
                if(!face_opt){
                    util::AnomalyLog::log_anomaly("failed to make new face.");
                    return;
                } else {
                    std::swap(face_opt.value(), mesh.faces[ifac]);
                }
            }

            // helper function to make new triangle faces using the info we have
            // @param mesh the mesh 
            // @param ifac_prev the index of the face to replace
            // @param ielem the index of the element to be the new left element 
            //        if the previous face is not a boundary face, the other element 
            //        will be found using the element index that was previously attached
            // @param face_nodes the nodes for the face in order 
            // @param face_nr_l the face number for the new left element 
            // @param prev_ielem the element index that was previously attached to this face
            //
            auto generate_new_face = [](AbstractMesh<T, IDX, 2>&mesh, 
                    IDX ifac_prev, IDX ielem, std::array<IDX, 2> face_nodes, int face_nr_l,
                    IDX prev_ielem) -> void {
                const Face<T, IDX, 2>& fac_prev = *(mesh.faces[ifac_prev]);
                if(fac_prev.bctype != BOUNDARY_CONDITIONS::INTERIOR){

                    auto face_opt = make_face<T, IDX, 2>(
                            DOMAIN_TYPE::HYPERCUBE, DOMAIN_TYPE::SIMPLEX, DOMAIN_TYPE::SIMPLEX,
                            1, ielem, ielem, std::span<const IDX>{face_nodes}, 
                            face_nr_l, 0, 0, fac_prev.bctype, fac_prev.bcflag);
                    if(!face_opt){
                        util::AnomalyLog::log_anomaly("failed to make new face.");
                        return;
                    } else {
                        std::swap(face_opt.value(), mesh.faces[ifac_prev]);
                    }
                } else {
                    IDX other_element = (prev_ielem == fac_prev.elemL) ? fac_prev.elemR : fac_prev.elemL;
                    ElementTransformation<T, IDX, 2>* trans_other = mesh.el_transformations[other_element];

                    int face_nr_r = trans_other->get_face_nr(face_nodes, mesh.conn_el.rowspan(other_element));
                    auto face_vert_r = trans_other->get_face_vert(face_nr_r, mesh.conn_el.rowspan(other_element));
                    int orient_r = hypercube_orient_trans<T, IDX, 2>.getOrientation(
                            face_nodes.data(), face_vert_r.data());

                    auto face_opt = make_face<T, IDX, 2>(
                            DOMAIN_TYPE::HYPERCUBE, DOMAIN_TYPE::SIMPLEX, DOMAIN_TYPE::SIMPLEX,
                            1, ielem, other_element, std::span<const IDX>{face_nodes}, 
                            face_nr_l, face_nr_r, orient_r);
                    if(!face_opt){
                        util::AnomalyLog::log_anomaly("failed to make new face.");
                        return;
                    } else {
                        std::swap(face_opt.value(), mesh.faces[ifac_prev]);
                    }
                }
            };

            // left element faces
            std::array<IDX, 4> faces_to_update;
            { // clL equivalent face
              // NOTE: facsuel is still pre-flip state
                IDX ifac_prev = mesh.facsuel[elemL, clL];
                std::array<IDX, 2> face_nodes{a, b};
                int face_nr_l = 1;
                generate_new_face(mesh, ifac_prev, elemL, face_nodes, face_nr_l, elemL);
                faces_to_update[0] = ifac_prev;
            }
            { // clR equivalent face
                IDX ifac_prev = mesh.facsuel[elemR, clR];
                std::array<IDX, 2> face_nodes{b, d};
                int face_nr_l = 2;
                generate_new_face(mesh, ifac_prev, elemL, face_nodes, face_nr_l, elemR);
                faces_to_update[1] = ifac_prev;
            }

            // right element faces
            { // blL equivalent face 
                IDX ifac_prev = mesh.facsuel[elemL, blL];
                std::array<IDX, 2> face_nodes{c, a}; // ccw
                int face_nr_l = 1;
                generate_new_face(mesh, ifac_prev, elemR, face_nodes, face_nr_l, elemL);
                faces_to_update[2] = ifac_prev;
            }
            { // blR equivalent face 
                IDX ifac_prev = mesh.facsuel[elemR, blR];
                std::array<IDX, 2> face_nodes{d, c}; // ccw
                int face_nr_l = 2;
                generate_new_face(mesh, ifac_prev, elemR, face_nodes, face_nr_l, elemR);
                faces_to_update[3] = ifac_prev;
            }

            // === update face connectivity ===
            mesh.facsuel[elemL, 0] = ifac;
            mesh.facsuel[elemR, 0] = ifac;
            for(IDX ifac_update : faces_to_update){
                const Face<T, IDX, 2>& fac = *(mesh.faces[ifac_update]);
                mesh.facsuel[fac.elemL, fac.face_nr_l()] = ifac_update;

                // TODO: right side update may be unnecessary?
                if(fac.bctype == BOUNDARY_CONDITIONS::INTERIOR)
                    mesh.facsuel[fac.elemR, fac.face_nr_r()] = ifac_update;
            }
        }
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
                Point old_node = mesh.coord[inode];
                std::span<T, ndim> node_view{mesh.coord[inode].begin(), mesh.coord[inode].end()};

                // perturb the node given the current coordinates
                perturb_func(old_node, node_view);
            }
        }
        mesh.update_coord_els();
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
                bbox.xmin[idim] = std::min(bbox.xmin[idim], mesh.coord[inode][idim]);
                bbox.xmax[idim] = std::max(bbox.xmax[idim], mesh.coord[inode][idim]);
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
