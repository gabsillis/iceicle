/**
 * @brief utilities for dealing with meshes
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "Numtool/point.hpp"
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

    template<class T, class IDX>
    auto burgers_linear_mesh(bool initial)
    -> std::optional<AbstractMesh<T, IDX, 2>> {
        using Point = MATH::GEOMETRY::Point<T, 2>;

        AbstractMesh<T, IDX, 2> mesh{};
        if(initial){
            mesh.coord = std::vector<Point>{
                Point{{0.00, 0.00}},
                Point{{0.25, 0.00}},
                Point{{0.75, 0.00}},
                Point{{1.00, 0.00}},
                Point{{0.00, 0.25}},
                Point{{0.25, 0.25}},
                Point{{0.75, 0.25}},
                Point{{1.00, 0.25}},
                Point{{0.00, 0.50}},
                Point{{0.25, 0.50}},
                Point{{0.75, 0.50}},
                Point{{1.00, 0.50}}
            };
        } else {
            mesh.coord = std::vector<Point>{
                Point{{0.00, 0.00}},
                Point{{0.25, 0.00}},
                Point{{0.75, 0.00}},
                Point{{1.00, 0.00}},
                Point{{0.00, 0.125}},
                Point{{0.50, 0.125}},
                Point{{0.50, 0.125}},
                Point{{1.00, 0.125}},
                Point{{0.00, 0.50}},
                Point{{0.25, 0.50}},
                Point{{0.50, 0.50}},
                Point{{1.00, 0.50}}
            };
        }

        // make the elements by hand
        std::vector<std::vector<IDX>> el_conn = {
            std::vector<IDX>{0, 4, 1, 5},
            std::vector<IDX> {1, 5, 2, 6},
            std::vector<IDX> {2, 6, 3, 7},
            std::vector<IDX> {4, 8, 5, 9},
            std::vector<IDX> {5, 9, 6, 10},
            std::vector<IDX> {6, 10, 7, 11}
        };

        mesh.conn_el = util::crs<IDX, IDX>(el_conn);
        for(int ielem = 0; ielem < el_conn.size(); ++ielem){
            mesh.el_transformations.push_back(
                transformation_table<T, IDX, 2>.get_transform(DOMAIN_TYPE::HYPERCUBE, 1));
        }
        
        // find the interior faces
        find_interior_faces(mesh);
        mesh.interiorFaceStart = 0;
        mesh.interiorFaceEnd = mesh.faces.size();
        mesh.bdyFaceStart = mesh.faces.size();

        // boundary faces
        auto add_bdy_face = [&mesh](int ielem, std::vector<IDX> nodes, BOUNDARY_CONDITIONS bctype, int bcflag){
            auto face_info = boundary_face_info(nodes, mesh.el_transformations[ielem], mesh.get_el_nodes(ielem));
            if(face_info){
                auto [fac_domn, face_nr_l] = face_info.value();
                ElementTransformation<T, IDX, 2> *trans = mesh.el_transformations[ielem];

                // get the face nodes in the correct order
                std::vector<IDX> ordered_face_nodes = trans->get_face_nodes(face_nr_l, mesh.get_el_nodes(ielem));

                auto face_opt = make_face<T, IDX, 2>(fac_domn, trans->domain_type, trans->domain_type, 1,
                        ielem, ielem, std::span<const IDX>{ordered_face_nodes},
                        face_nr_l, 0, 0, bctype, bcflag);
                if(face_opt){
                    mesh.faces.push_back(std::move(face_opt.value()));
                }
            }
        };

        {
            int ielem = 0;
            std::vector<IDX> nodes{0, 1};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 1;
            std::vector<IDX> nodes{1, 2};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 2;
            std::vector<IDX> nodes{2, 3};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 0;
            std::vector<IDX> nodes{0, 4};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 3;
            std::vector<IDX> nodes{4, 8};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 2;
            std::vector<IDX> nodes{3, 7};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 5;
            std::vector<IDX> nodes{7, 11};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::DIRICHLET;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 3;
            std::vector<IDX> nodes{8, 9};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::SPACETIME_FUTURE;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 4;
            std::vector<IDX> nodes{9, 10};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::SPACETIME_FUTURE;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }
        {
            int ielem = 5;
            std::vector<IDX> nodes{10, 11};
            BOUNDARY_CONDITIONS bctype = BOUNDARY_CONDITIONS::SPACETIME_FUTURE;
            int bcflag = 0;
            add_bdy_face(ielem, nodes, bctype, bcflag);
        }

        mesh.bdyFaceEnd = mesh.faces.size();

        mesh.elsup = to_elsup(mesh.conn_el, mesh.n_nodes());

        // form face dof connectivity
        for(const auto& fac_ptr : mesh.faces){
            mesh.face_extended_conn.push_back(FaceGeoDofConnectivity{*fac_ptr, mesh.conn_el});
        }
        return std::optional{mesh};
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
        using namespace util;

        AbstractMesh<T, IDX, 2> mesh{};

        // make the nodes
        T dx = (xmax[0] - xmin[0]) / (nelem[0]);
        T dy = (xmax[1] - xmin[1]) / (nelem[1]);
        IDX nnodex = nelem[0] + 1;
        IDX nnodey = nelem[1] + 1;
        for(IDX iy = 0; iy < nnodey; ++iy){
            for(IDX ix = 0; ix < nnodex; ++ix){
                mesh.coord.push_back(Point{{xmin[0] + ix * dx, xmin[1] + iy * dy}});
            }
        }

        IDX half_quad_x = (IDX) (nelem[0] * quad_ratio[0] / 2);
        IDX half_quad_y = (IDX) (nelem[1] * quad_ratio[1] / 2);

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
                    mesh.el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::HYPERCUBE, 1));
                    std::vector<IDX> nodes{bottomleft, topleft, bottomright, topright};
                    el_conn.push_back(nodes);
                } else {
                    std::vector<IDX> nodes1{bottomleft, bottomright, topleft};
                    std::vector<IDX> nodes2{topleft, bottomright, topright};
                    mesh.el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::SIMPLEX, 1));
                    mesh.el_transformations.push_back(transformation_table<T, IDX, 2>
                            .get_transform(DOMAIN_TYPE::SIMPLEX, 1));
                    el_conn.push_back(nodes1);
                    el_conn.push_back(nodes2);

                }
            }
        }

        // update the element crs matrices 
        mesh.conn_el = util::crs<IDX, IDX>{el_conn};

        { // build the element coordinates matrix
            mesh.coord_els = util::crs<Point, IDX>{std::span{mesh.conn_el.cols(), mesh.conn_el.cols() + mesh.conn_el.nrow() + 1}};
            for(IDX iel = 0; iel < mesh.nelem(); ++iel){
                for(std::size_t icol = 0; icol < mesh.conn_el.rowsize(iel); ++icol){
                    mesh.coord_els[iel, icol] = mesh.coord[mesh.conn_el[iel, icol]];
                }
            }
        }

        // find the interior faces
        find_interior_faces(mesh);
        mesh.interiorFaceStart = 0;
        mesh.interiorFaceEnd = mesh.faces.size();
        mesh.bdyFaceStart = mesh.faces.size();

        // find the boundary faces TODO: switch to lists for removal efficiency
        std::vector<std::vector<IDX>> bface_nodes{};
        std::vector<std::pair<BOUNDARY_CONDITIONS, int>> bcinfos{};
        for(IDX ixquad = 0; ixquad < nelem[0]; ++ixquad){
            // bottom face
            bface_nodes.push_back(std::vector<IDX>{ixquad, ixquad + 1});
            bcinfos.push_back(std::pair{bcs[1], bcflags[1]});

            // top face
            bface_nodes.push_back(std::vector<IDX>{nelem[1] * nnodex + ixquad, nelem[1] * nnodex + ixquad + 1});
            bcinfos.push_back(std::pair{bcs[3], bcflags[3]});
        }
        for(IDX iyquad = 0; iyquad < nelem[1]; ++iyquad) {
            // left face
            bface_nodes.push_back(std::vector<IDX>{iyquad * nnodex, (iyquad + 1) * nnodex});
            bcinfos.push_back(std::pair{bcs[0], bcflags[0]});

            // right face
            bface_nodes.push_back(std::vector<IDX>{iyquad * nnodex + nelem[0], (iyquad + 1) * nnodex + nelem[0]});
            bcinfos.push_back(std::pair{bcs[2], bcflags[2]});
        }
        for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
            for(IDX inodelist = 0; inodelist < bface_nodes.size(); ++inodelist){
                auto face_info = boundary_face_info(std::span<const IDX>{bface_nodes[inodelist]}, 
                    mesh.el_transformations[ielem], mesh.conn_el.rowspan(ielem)); 
                if(face_info){
                    auto [fac_domn, face_nr_l] = face_info.value();
                    ElementTransformation<T, IDX, 2> *trans = mesh.el_transformations[ielem];

                    // get the face nodes in the correct order
                    std::vector<IDX> ordered_face_nodes = 
                        trans->get_face_nodes(face_nr_l, mesh.conn_el.rowspan(ielem));

                    auto [bctype, bcflag] = bcinfos[inodelist];
                    DOMAIN_TYPE domain_type = trans->domain_type;
                    auto face_opt = make_face<T, IDX, 2>(fac_domn, domain_type, domain_type, 1,
                            ielem, ielem, std::span<const IDX>{ordered_face_nodes}, face_nr_l, 0, 0, bctype, bcflag);
                    if(face_opt){
                        mesh.faces.push_back(std::move(face_opt.value()));
                    }

                    // remove the found bface 
                    bface_nodes.erase(bface_nodes.begin() + inodelist);
                    bcinfos.erase(bcinfos.begin() + inodelist);
                    inodelist--; // decrement after removal
                }
            }
        }
        mesh.bdyFaceEnd = mesh.faces.size();
        mesh.elsup = to_elsup(mesh.conn_el, mesh.n_nodes());

        // form face dof connectivity
        for(const auto& fac_ptr : mesh.faces){
            mesh.face_extended_conn.push_back(FaceGeoDofConnectivity{*fac_ptr, mesh.conn_el});
        }
        return std::optional{mesh};
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
