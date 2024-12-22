/**
 * @file mesh.hpp
 * @brief Abstract mesh definitionn
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @date 2023-06-27
 */
#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include "iceicle/geometry/hypercube_face.hpp"
#include "iceicle/geometry/simplex_element.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include "iceicle/transformations/HypercubeTransformations.hpp"
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <iceicle/crs.hpp>
#include <optional>
#include <ostream>
#include <ranges>
#include <type_traits>
#include <memory>
#include <list>
#include <iceicle/basis/dof_mapping.hpp>
#ifndef NDEBUG
#include <iomanip>
#endif
namespace iceicle {

    /// @brief generate the node connectivity matrix in crs format from the element connectivity matrix 
    /// elsup -> elements surrounding points
    /// @param conn_el the element connectivity matrix
    /// @param nnode the number of nodes
    template<class IDX, int ndim>
    constexpr
    auto to_elsup(dof_map<IDX, ndim, h1_conformity(ndim)>& conn_el)
    -> util::crs<IDX, IDX>
    {
        std::vector<std::vector<IDX>> elsup_ragged(conn_el.size());
        for(IDX iel = 0; iel < conn_el.nelem(); ++iel){
            for(const IDX inode : conn_el.rowspan(iel)){
                elsup_ragged[inode].push_back(iel);
            }
        }
        return util::crs<IDX, IDX>{elsup_ragged};
    }

    /// @brief generate the element connectivity matrix from the face list 
    /// elsuel -> elements surrounding elements 
    /// @param face_list the list of faces 
    /// @param nelem - the number of elements
    /// @return the connectivity of elements to elements
    template<class T, class IDX, int ndim>
    constexpr
    auto to_elsuel(IDX nelem, const std::span< const std::unique_ptr< Face<T, IDX, ndim> > >& face_list)
    -> util::crs<IDX> 
    {
        std::vector<std::vector<IDX>> elsuel_dynamic(nelem, std::vector<IDX>{});

        for(const auto& face : face_list){
            IDX elemL = face->elemL;
            IDX elemR = face->elemR;
            if(elemR != -1){
                elsuel_dynamic[elemL].push_back(elemR);  
                elsuel_dynamic[elemR].push_back(elemL);  
            }
        }
        return util::crs{elsuel_dynamic};
    }

    template<class T, class IDX, int ndim>
    constexpr
    auto create_element(DOMAIN_TYPE domain, int geo_order, std::span<IDX> nodes)
    -> std::optional< std::unique_ptr<GeometricElement<T, IDX, ndim>> >
    {
        // validate nodes 
        for(IDX node : nodes)
            if(node < 0) return std::nullopt;

        switch(domain){
            case DOMAIN_TYPE::HYPERCUBE:
                {
                    return NUMTOOL::TMP::invoke_at_index(
                        NUMTOOL::TMP::make_range_sequence<int, 1, build_config::FESPACE_BUILD_GEO_PN>{},
                        geo_order,
                        [&]<int order> -> std::optional< std::unique_ptr< GeometricElement<T, IDX, ndim> > >{
                            HypercubeElement<T, IDX, ndim, order> el{};
                            for(int inode = 0; inode < el.n_nodes(); ++inode){
                                el.setNode(inode, nodes[inode]);
                            }
                            return std::optional{std::make_unique<HypercubeElement<T, IDX, ndim, order>>(el)};
                        }
                    );
                }
                break;
            case DOMAIN_TYPE::SIMPLEX:
            {
                if constexpr(ndim == 2)
                    if(geo_order == 1) {
                        TriangleElement<T, IDX> el{};
                        std::ranges::copy(nodes, el.node_idxs.begin());
                        return std::optional{std::make_unique<TriangleElement<T, IDX>>(el)};
}

                return std::nullopt;
            }
            default:
                return std::nullopt;
                break;
        }
    }


    /// @brief contains the information to represent an elmement Geometrically
    /// for elements communicated from another MPI partition
    template<class T, class IDX, int ndim>
    struct CommElementInfo {
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        ElementTransformation<T, IDX, ndim> *trans;
        std::vector<IDX> conn_el;
        std::vector<Point> coord_el;
    };

    /// @brief node indices for a face and the connected elements. 
    /// Duplicates are removed.
    ///
    /// First face dofs (shared), then left dofs, then right dofs
    ///
    /// @tparam IDX the index type
    template<class IDX>
    struct FaceGeoDofConnectivity {
        struct Dofs {
            IDX gdof; // global node index 
            IDX trace_dof; // local trace node index or -1 if n/a
            IDX left_dof; // local node index for left element or -1 if n/a
            IDX right_dof; // local node index for right element or -1 if n/a
        };

        std::vector<Dofs> dofs; 

        private:
        template<class T, int ndim, class crs_index_t>
        [[nodiscard]] static constexpr inline
        auto create_dofs_array(const Face<T, IDX, ndim>& face, const util::crs<IDX, crs_index_t>& el_conn)
        -> std::vector<Dofs>
        {
            std::size_t nnode_l = el_conn.rowsize(face.elemL);
            std::size_t nnode_r = face.elemR != -1 ? el_conn.rowsize(face.elemR) : 0;
            std::size_t size = nnode_r == 0 ? nnode_l : nnode_l + nnode_r - face.n_nodes();
            std::vector<Dofs> ret(size);

            std::list<std::pair<IDX, IDX>> left_ldofs{};
            for(IDX ilnode = 0; ilnode < nnode_l; ++ilnode)
                left_ldofs.push_back(std::pair{ilnode, el_conn[face.elemL, ilnode]});
            std::list<std::pair<IDX, IDX>> right_ldofs{};
            for(IDX irnode = 0; irnode < nnode_r; ++irnode)
                right_ldofs.push_back(std::pair{irnode, el_conn[face.elemR, irnode]});

            // get the trace nodes and remove duplicates from the lists
            IDX inode;
            for(inode = 0; inode < face.n_nodes(); ++inode){
                ret[inode].gdof = face.nodes()[inode];
                ret[inode].trace_dof = inode;

                for(auto it = left_ldofs.begin(); it != left_ldofs.end(); ){
                    if(it->second == ret[inode].gdof){
                        ret[inode].left_dof = it->first;
                        it = left_ldofs.erase(it); // remove to not double count
                        break;
                    } else {
                        ++it;
                    }
                }
                for(auto it = right_ldofs.begin(); it != right_ldofs.end(); ){
                    if(it->second == ret[inode].gdof){
                        ret[inode].right_dof = it->first;
                        it = right_ldofs.erase(it); // remove to not double count
                        break;
                    } else {
                        ++it;
                    }
                }
            }
          
            // left nodes 
            for(std::pair<IDX, IDX> lnode_gnode_pair : left_ldofs){
                ret[inode].left_dof = lnode_gnode_pair.first;
                ret[inode].gdof = lnode_gnode_pair.second;
                ret[inode].trace_dof = -1;
                ret[inode].right_dof = -1;
                ++inode;
            }

            // right nodes 
            for(std::pair<IDX, IDX> lnode_gnode_pair : right_ldofs){
                ret[inode].right_dof = lnode_gnode_pair.first;
                ret[inode].gdof = lnode_gnode_pair.second;
                ret[inode].trace_dof = -1;
                ret[inode].left_dof = -1;
                ++inode;
            }
            return ret;
        }

        public:

        // @brief given a face and the elemnt connectivity matrix 
        // form the dof connectivity
        template<class T, int ndim, class crs_index_t>
        FaceGeoDofConnectivity(const Face<T, IDX, ndim>& face, const util::crs<IDX, crs_index_t>& el_conn)
        : dofs{create_dofs_array(face, el_conn)}
        {}

        // @brief get the size of connected degrees of freedom
        [[nodiscard]] constexpr  inline
        auto size() const
        -> std::size_t 
        { return dofs.size(); }
    };
    template<class T, class IDX, int ndim, class crs_index_t>
    FaceGeoDofConnectivity(const Face<T, IDX, ndim>& face, const util::crs<IDX, crs_index_t>& el_conn)
    -> FaceGeoDofConnectivity<IDX>;

    /**
     * @brief generate the set of nodes for each direction uniformly spaced
     * The cartesian product of these 1D arrays for each dimension 
     * can then be used to generate all the nodes in the mesh
     * @param xmin the minimum point of the domain bounding box
     * @param xmax the maximum point of the domain bounding box
     * @param nelem the number of elements in each direction
     * @param order the polynomial order of the hypercube elements
     */
    template< int ndim >
    [[nodiscard]] inline constexpr
    auto generate_directional_nodes(
        std::ranges::range auto xmin, 
        std::ranges::range auto xmax,
        std::ranges::range auto nelem,
        int order
    ) noexcept -> std::array<std::vector< std::ranges::range_value_t<decltype(xmin)> >, ndim> 
    requires(
        std::convertible_to<std::ranges::range_value_t<decltype(xmax)>,
            std::ranges::range_value_t<decltype(xmin)> >
        && std::is_integral_v<std::ranges::range_value_t<decltype(nelem)> >
    ) {
        using T = std::ranges::range_value_t<decltype(xmin)>;
        std::array<std::vector<T>, ndim> nodes_1d;
        auto it_xmin = xmin.begin();
        auto it_xmax = xmax.begin();
        auto it_nelem = nelem.begin();
        for(int idim = 0; idim < ndim; ++idim, ++it_xmin, ++it_xmax, ++it_nelem){
            T pt_min = *it_xmin;
            std::size_t nelem_dir = *it_nelem;
            T dx = (*it_xmax - *it_xmin) / (nelem_dir * order);
            std::vector<T> nodes_dir(nelem_dir + 1);
            nodes_dir[0] = pt_min;
            for(std::size_t i = 0; i < nelem_dir; ++i){
                nodes_dir[i + 1] = nodes_dir[i] + dx;
            }
            nodes_1d[idim] = nodes_dir;
        }
        return nodes_1d;
    }

    /**
     * @brief Abstract class that defines a mesh
     *
     * @tparam T The floating point type
     * @tparam IDX The index type
     * @tparam ndim the number of dimensions
     */
    template<typename T, typename IDX, int ndim>
    class AbstractMesh {
        private:
        // ================
        // = Type Aliases =
        // ================
        using face_t = Face<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        // ===========================
        // = Primary Data Structures =
        // ===========================

        /// The node coordinates
        NodeArray<T, ndim> coord;

        /// Connectivity array for the elements to the nodes
        dof_map<IDX, ndim, h1_conformity(ndim)> conn_el;

        /// The node coordinates for each element 
        /// NOTE: updates to coord must be propogated to this array
        util::crs<Point, IDX> coord_els;

        /// @brief the element transformations for each element
        std::vector<ElementTransformation<T, IDX, ndim>* > el_transformations;

        /// All faces (internal and boundary) 
        /// interior faces must be a contiguous set
        std::vector<std::unique_ptr<face_t>> faces;

        /// index of the start of interior faces (interior faces must be consecutive)
        IDX interiorFaceStart;
        /// index of one past the end of interior faces
        IDX interiorFaceEnd; 
        // index of the start of the boundary faces (must be consecutive)
        IDX bdyFaceStart;
        /// index of one past the end of the boundary faces
        IDX bdyFaceEnd;

        /// @brief The connectivity array ELements SUrrounding Points 
        /// represents the element indices that surround each node index
        util::crs<IDX, IDX> elsup;

        /// @brief the faces surrounding elements
        /// in order of face number: 
        /// i.e 
        /// Let element (iel) have a face (ifac) where iel is the Left element 
        /// and face_nr_l == 1
        /// then
        /// facsuel[iel, 1] = ifac
        util::crs<IDX, IDX> facsuel;

        /// @brief The parallel partitioning of elements
        pindex_map<IDX> element_partitioning;

        /// @brief get the number of elements
        [[nodiscard]] inline constexpr 
        auto nelem() const noexcept -> IDX
        { return conn_el.nelem(); }

        // ===============
        // = Constructor =
        // ===============

        /** @brief construct an empty mesh */
        AbstractMesh() 
        : coord{}, conn_el{}, coord_els{}, el_transformations{}, faces{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0), elsup{}, facsuel{} {}

        /// @brief A description of a boundary face 
        /// contains all the information needed to generate the face data structure 
        /// first the boundary condition type 
        /// then the boundary condition integer flag 
        /// then the nodes of the boundary face
        using boundary_face_desc = std::tuple<BOUNDARY_CONDITIONS, int, std::vector<IDX>>;

        /// @brief Construct a mesh from provided connectivity information
        /// @param coord the mesh coordinates 
        /// @param conn_el_arg compressed row storage of element connectivity
        /// @param el_transformations array of pointers to the corresponding transformation for each element
        /// @param boundary_face_descriptions tuple of BOUNDARY_CONDITIONS (type), integer (flag), 
        ///        and array of indices (the nodes) that describe boundary faces
        AbstractMesh(
            NodeArray<T, ndim>& coord,
            auto&& conn_el_arg,
            std::vector< ElementTransformation<T, IDX, ndim>* > el_transformations,
            const std::vector<boundary_face_desc>& boundary_face_descriptions
        ) : coord{coord}, conn_el{conn_el_arg}, coord_els{},
            el_transformations{el_transformations}
        {
            { // build the element coordinates matrix
                coord_els = util::crs<Point, IDX>{std::span{conn_el.dof_connectivity.cols(),
                    conn_el.dof_connectivity.cols() + conn_el.dof_connectivity.nrow() + 1}};
                for(IDX iel = 0; iel < conn_el.nelem(); ++iel){
                    for(std::size_t icol = 0; icol < conn_el.ndof_el(iel); ++icol){
                        coord_els[iel, icol] = coord[conn_el[iel, icol]];
                    }
                }
            }

            // form elements sourrounding points 
            
            // elements surrounding points
            std::vector<std::vector<IDX>> elsup_ragged(n_nodes());
            for(IDX ielem = 0; ielem < nelem(); ++ielem){
                for(IDX inode : conn_el.rowspan(ielem)){
                    elsup_ragged[inode].push_back(ielem);
                }
            }

            // remove duplicates and sort
            for(IDX inode = 0; inode < n_nodes(); ++inode){
                std::ranges::sort(elsup_ragged[inode]);
                auto unique_subrange = std::ranges::unique(elsup_ragged[inode]);
                elsup_ragged[inode].erase(unique_subrange.begin(), unique_subrange.end());
            }

            elsup = util::crs<IDX, IDX>{elsup_ragged};

            // find the interior faces
            // if elements share at least ndim points, then they have a face
            for(IDX ielem = 0; ielem < nelem(); ++ielem){
                int max_faces = el_transformations[ielem]->nfac;
                std::vector<IDX> connected_elements;
                connected_elements.reserve(max_faces);

                // loop through elements that share a node
                for(IDX inode : conn_el.rowspan(ielem)){
                    for(auto jelem_iter = std::lower_bound(elsup.rowspan(inode).begin(), elsup.rowspan(inode).end(), ielem);
                            jelem_iter != elsup.rowspan(inode).end(); ++jelem_iter){
                        IDX jelem = *jelem_iter;

                        // skip the cases that would lead to duplicate or boundary faces
                        if( ielem == jelem || std::ranges::find(connected_elements, jelem) != std::ranges::end(connected_elements) )
                            continue; 

                        // try making the face that is the intersection of the two elements
                        auto face_opt = make_face(ielem, jelem, 
                                el_transformations[ielem], el_transformations[jelem],
                                conn_el.rowspan(ielem), conn_el.rowspan(jelem));
                        if(face_opt){
                            faces.push_back(std::move(face_opt.value()));
                            // short circuit if all the faces have been found
                            connected_elements.push_back(jelem);
                            if(connected_elements.size() == max_faces) break;
                        }
                    }
                }
            }

            interiorFaceStart = 0;
            interiorFaceEnd = faces.size();
            bdyFaceStart = interiorFaceEnd;

            // make the boundary faces
            for(const boundary_face_desc& info : boundary_face_descriptions){
                auto [bc_type, bc_flag, boundary_nodes] = info;
                // search the elements around the first node 
                for(IDX ielem : elsup.rowspan(boundary_nodes[0])){
                    ElementTransformation<T, IDX, ndim>* trans = el_transformations[ielem];
                    std::span<IDX> elnodes = get_el_nodes(ielem);
                    auto fac_info_optional = boundary_face_info(boundary_nodes, trans, elnodes);
                    if(fac_info_optional){
                        // make the face and add it
                        auto [fac_domain, face_nr] = fac_info_optional.value();
                        std::vector<IDX> face_nodes = trans->get_face_nodes(face_nr, elnodes);
                        auto fac_opt = make_face<T, IDX, ndim>(fac_domain, trans->domain_type, trans->domain_type, 
                            trans->order, ielem, ielem, face_nodes, face_nr, 0, 0, bc_type, bc_flag);
                        if(fac_opt)
                            faces.push_back(std::move(fac_opt.value()));
                        else 
                            util::AnomalyLog::log_anomaly(util::Anomaly{"Cannot form boundary face", util::general_anomaly_tag{}});
                    }
                }
            }
            bdyFaceEnd = faces.size();

            // faces surrounding elements
            std::vector<std::vector<IDX>> facsuel_ragged(nelem());
            for(IDX iel = 0; iel < nelem(); ++iel){
                facsuel_ragged[iel].resize(el_transformations[iel]->nfac);
            }
            for(IDX ifac = interiorFaceStart; ifac < interiorFaceEnd; ++ifac) {
                facsuel_ragged[faces[ifac]->elemL][faces[ifac]->face_nr_l()] = ifac;
                facsuel_ragged[faces[ifac]->elemR][faces[ifac]->face_nr_r()] = ifac;
            }
            for(IDX ifac = bdyFaceStart; ifac < bdyFaceEnd; ++ifac) {
                facsuel_ragged[faces[ifac]->elemL][faces[ifac]->face_nr_l()] = ifac;
            }
            facsuel = util::crs<IDX, IDX>{facsuel_ragged};
        }

        AbstractMesh(const AbstractMesh<T, IDX, ndim>& other) 
        : coord{other.coord}, conn_el{other.conn_el}, coord_els{other.coord_els}, 
          el_transformations{other.el_transformations}, faces{},
          interiorFaceStart(other.interiorFaceStart), interiorFaceEnd(other.interiorFaceEnd),
          bdyFaceStart(other.bdyFaceStart), bdyFaceEnd(other.bdyFaceEnd), elsup{other.elsup},
          facsuel{other.facsuel}
        {
            for(auto& facptr : other.faces){
                faces.push_back(std::move(facptr->clone()));
            }
        }

        AbstractMesh(AbstractMesh<T, IDX, ndim>&& other) = default;

        AbstractMesh<T, IDX, ndim>& operator=(const AbstractMesh<T, IDX, ndim>& other){
            if(this != &other){
                coord = other.coord;
                conn_el = other.conn_el;
                coord_els = other.coord_els;
                el_transformations = other.el_transformations;
                faces.clear();
                faces.reserve(other.faces.size());
                for(auto& facptr : other.faces){
                    faces.push_back(std::move(facptr->clone()));
                }
                interiorFaceStart = other.interiorFaceStart;
                interiorFaceEnd = other.interiorFaceEnd;
                bdyFaceStart = other.bdyFaceStart;
                bdyFaceEnd = other.bdyFaceEnd;
                elsup = other.elsup;
                facsuel = other.facsuel;
            }
            return *this;
        }

        AbstractMesh<T, IDX, ndim>& operator=(AbstractMesh<T, IDX, ndim>&& other) = default;

        AbstractMesh(std::size_t nnode) 
        : coord{nnode}, conn_el{}, coord_els{}, faces{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0) {}

        private:

        inline static constexpr NUMTOOL::TENSOR::FIXED_SIZE::Tensor<BOUNDARY_CONDITIONS, 2*ndim>
        all_periodic = [](){
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<BOUNDARY_CONDITIONS, 2*ndim> ret;
            for(int i = 0; i < 2*ndim; ++i) ret[i] = BOUNDARY_CONDITIONS::PERIODIC;
            return ret;
        }();

        inline static constexpr NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2 * ndim>
        all_zero = [](){
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2*ndim> ret;
            for(int i = 0; i < 2*ndim; ++i) ret[i] = 0;
            return ret;
        }();

        public:

        /**
         * @brief generate a mesh from the nodes in each direction 
         * the nodes of the mesh become the cartesian product of the nodes in each direction 
         * @param nodes_1d the nodes in each direction 
         * @param order the polynomial order of the hypercube element
         * @param bctypes the boundary conditions for each face of the whole domain,
         *                following the hypercube numbering convention
         *                i.e the coordinate direction index (x: 0, y:1, z:2, ...) = face_number % ndim
         *                the negative side face is face_number / ndim == 0, and positive side otherwise 
         *                so for 2d this would be: 
         *                0: left face 
         *                1: bottom face 
         *                2: right face 
         *                3: top face
         *
         * @param bcflags the boundary condition flags for each face of the whole domain,
         *                same layout
         */
        template<
            std::ranges::random_access_range R_bctype,
            std::ranges::random_access_range R_bcflags
        >
        AbstractMesh(
            std::array<std::vector<T>, ndim> nodes_1d,
            int order,
            R_bctype&& bctypes,
            R_bcflags&& bcflags
        ) requires(
            std::same_as<std::ranges::range_value_t<R_bctype>, BOUNDARY_CONDITIONS> &&
            std::convertible_to<std::ranges::range_value_t<R_bcflags>, int>
        ) : coord{}, conn_el{}, coord_els{}, faces{}
        {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            std::array<IDX, ndim> directional_nelem;
            for(int idim = 0; idim < ndim; ++idim){
                directional_nelem[idim] = nodes_1d[idim].size() - 1;
            }

            // determine the number of nodes to generate
            int nnodes = 1;
            int nelem = 1;
            IDX nnode_dir[ndim];
            IDX stride[ndim];
            IDX stride_nodes[ndim];
            T dx[ndim];
            for(int idim = 0; idim < ndim; ++idim) {
                stride_nodes[idim] = 1;
                stride[idim] = 1;
                nnode_dir[idim] = directional_nelem[idim] * (order) + 1;
                nnodes *= nnode_dir[idim];
                nelem *= directional_nelem[idim];
            }

            for(int idim = 0; idim < ndim; ++idim){
                for(int jdim = 0; jdim < idim; ++jdim){
                    stride[idim] *= directional_nelem[jdim];
                    stride_nodes[idim] *= nnode_dir[jdim];
                }
            }
            coord.resize(nnodes);

            // Generate the nodes 
            std::array<IDX, ndim> ijk;
            std::ranges::fill(ijk, 0);
            for(int inode = 0; inode < nnodes; ++inode){
                // calculate the coordinates 
                for(int idim = 0; idim < ndim; ++idim){
                    coord[inode][idim] = nodes_1d[idim][ijk[idim]];
                }
                // increment
                ++ijk[0];
                for(int idim = 0; idim < ndim - 1; ++idim){
                    if(ijk[idim] == nnode_dir[idim]){
                        ijk[idim] = 0;
                        ++ijk[idim + 1];
                    } else {
                        // short circuit
                        break;
                    }
                }
            }


           // ENTERING ORDER TEMPLATED SECTION
           // here we find the compile time function to call based on the order input 
            NUMTOOL::TMP::constexpr_for_range<1, MAX_DYNAMIC_ORDER + 1>([&]<int Pn>{
                using FaceType = HypercubeFace<T, IDX, ndim, Pn>;

                transformations::HypercubeElementTransformation<T, IDX, ndim, Pn> trans{};
                if(order == Pn){

                    // WARNING: initializing this outside of order templated section breaks in O3
                    std::vector<std::vector<IDX>> ragged_conn_el(nelem);
                    // form the element connectivity matrix
                    std::array<IDX, ndim> ijk;
                    std::ranges::fill(ijk, 0);
                    for(int ielem = 0; ielem < nelem; ++ielem){
                        // create the element
                        el_transformations.push_back( 
                                (transformation_table<T, IDX, ndim>.get_transform(DOMAIN_TYPE::HYPERCUBE, order)) );

                        // get the nodes 
                        for(IDX inode = 0; inode < trans.n_nodes(); ++inode){
                            IDX iglobal = 0;
                            IDX ijk_gnode[ndim];
                            for(int idim = 0; idim < ndim; ++idim){
                                ijk_gnode[idim] = ijk[idim] * order + trans.tensor_prod.ijk_poin[inode][idim];
                            }

                            for(int idim = 0; idim < ndim; ++idim){
                                iglobal += ijk_gnode[idim] * stride_nodes[idim];
                            }
                            ragged_conn_el[ielem].push_back(iglobal);
                        }

                        // increment
                        ++ijk[0];
                        for(int idim = 0; idim < ndim - 1; ++idim){
                            if(ijk[idim] == directional_nelem[idim]){
                                ijk[idim] = 0;
                                ++ijk[idim + 1];
                            } else {
                                // short circuit
                                break;
                            }
                        }
                    }

                    conn_el = util::crs<IDX, IDX>{ragged_conn_el};
                    { // build the element coordinates matrix
                        coord_els = util::crs<Point, IDX>{
                            std::span{conn_el.dof_connectivity.cols(),
                                conn_el.dof_connectivity.cols() 
                                    + conn_el.dof_connectivity.nrow() + 1}};
                        for(IDX iel = 0; iel < nelem; ++iel){
                            for(std::size_t icol = 0; icol < conn_el.ndof_el(iel); ++icol){
                                coord_els[iel, icol] = coord[conn_el[iel, icol]];
                            }
                        }
                    }

                    // ===========================
                    // = Interior Face Formation =
                    // ===========================
                    
                    // loop over each direction 
                    for(int idir = 0; idir < ndim; ++idir){

                        // oordinates of the left element
                        std::array<IDX, ndim> ijk;
                        std::ranges::fill(ijk, 0);

                        //function to increment the ijk of the left element 
                        auto next_ijk = [&](std::array<IDX, ndim>& ijk) -> bool {
                            for(int idim = 0; idim < ndim; ++idim){
                                if(idim == idir){
                                    // we have n-1 left elements in the given direction, n otherwise
                                    if(ijk[idim] >= directional_nelem[idim] - 2){
                                        // go on to the next oordinate
                                        ijk[idim] = 0; 
                                    } else {
                                        ijk[idim]++;
                                        return true; // increment complete
                                    }
                                } else {
                                    // n elements in the given direction
                                    if(ijk[idim] >= directional_nelem[idim] - 1){
                                        // go on to the next oordinate
                                        ijk[idim] = 0; 
                                    } else {
                                        ijk[idim]++;
                                        return true; // increment complete
                                    }
                                }
                            }
                            return false;
                        };

                        // do loop safegaurded against empty faces in that direction
                        if(directional_nelem[idir] > 1) do {
                            // make the face 
                            std::array<IDX, ndim> ijk_r;
                            std::ranges::copy(ijk, ijk_r.begin());
                            ijk_r[idir]++; // increment in the face direction

                            // get the element number from the ordinates
                            IDX iel = 0;
                            IDX ier = 0;
                            for(int jdim = 0; jdim < ndim; ++jdim){
                                iel += ijk[jdim] * stride[jdim];
                                ier += ijk_r[jdim] * stride[jdim];
                            }

                            // get the face numbers
                            int face_nr_l = ndim + idir; // positive side
                            int face_nr_r = idir;

                            Tensor<IDX, FaceType::trans.n_nodes> face_nodes;
                            // TODO: Generalize
                            auto &transl = trans;
                            auto &transr = trans;
                            transl.get_face_nodes(
                                face_nr_l,
                                &conn_el[iel, 0],
                                face_nodes.data()
                            );

                            // get the orientations
                            static constexpr int nfacevert = MATH::power_T<2, ndim-1>::value;
                            IDX vert_l[nfacevert];
                            IDX vert_r[nfacevert];
                            transl.get_face_vert(face_nr_l, &conn_el[iel, 0], vert_l);
                            transr.get_face_vert(face_nr_r, &conn_el[ier, 0], vert_r);
                            int orientationr = FaceType::orient_trans.getOrientation(vert_l, vert_r);

                            faces.emplace_back(std::make_unique<FaceType>(
                                iel, ier, face_nodes, face_nr_l, face_nr_r,
                                orientationr, BOUNDARY_CONDITIONS::INTERIOR, 0));

                        } while (next_ijk(ijk));
                    }

                    interiorFaceStart = 0;
                    interiorFaceEnd = faces.size();

                    // ===========================
                    // = Boundary Face Formation =
                    // ===========================
                    
                    // loop over major axis directions
                    for(int idim = 0; idim < ndim; ++idim){

                        // get the number of faces on one boundary normal to idim
                        IDX nbfac_dir = 1;
                        for(int jdim = 0; jdim < ndim; ++jdim) if (jdim != idim) {
                            nbfac_dir *= directional_nelem[jdim];
                        }

                        // reset the ordinates
                        for(int jdim = 0; jdim < ndim; ++jdim) ijk[jdim] = 0;

                        for(IDX ifac = 0; ifac < nbfac_dir; ++ifac){

                            // form the -1 face 
                            // get the element number from the ordinates
                            IDX iel = 0;
                            for(int jdim = 0; jdim < ndim; ++jdim){
                                iel += ijk[jdim] * stride[jdim];
                            }

                            // get the face numbers 
                            int face_nr_l = idim; // this is the negative side 
                            int face_nr_r = 0; // boundary

                            // get the global face node indices
                            Tensor<IDX, FaceType::trans.n_nodes> face_nodes;
                            // TODO: Generalize: CRTP?
                            auto &transl = trans;
                            transl.get_face_nodes(
                                face_nr_l,
                                &conn_el[iel, 0],
                                face_nodes.data()
                            );

                            int orientationr = 0; // choose the simplest one for the boundary

                            int bc_idx = idim;
                            auto faceA = std::make_unique<FaceType>(
                                iel, -1, face_nodes, face_nr_l, face_nr_r,
                                orientationr, bctypes[bc_idx], bcflags[bc_idx]
                            );

#ifndef NDEBUG
                            std::cout << "Boundary Face A |" << " iel: " << std::setw(3) <<  faceA->elemL
                                << " | ier: " << std::setw(3) << faceA->elemR
                                << " | #: " << face_nr_l << " | orient: " << orientationr << " | nodes [ ";
                            for(int i = 0; i < FaceType::trans.n_nodes; ++i){
                                std::cout << std::setw(3) << face_nodes[i] << " ";
                            }
                            std::cout << "]" 
                                " | bctype: " << bc_name(faceA->bctype) 
                                << " | bcflag" << std::setw(2) << faceA->bcflag << std::endl;
#endif // !DEBUG

                            // form the +1 face
                            // set to the farthest element
                            ijk[idim] = directional_nelem[idim] - 1; 
                            iel = 0;
                            for(int jdim = 0; jdim < ndim; ++jdim){
                                iel += ijk[jdim] * stride[jdim];
                            }

                            // get the face numbers 
                            face_nr_l = idim + ndim; // this is the positive side 
                            face_nr_r = 0; // boundary

                            // get the global face node indices
                            // TODO: Generalize
                            auto &transl2 = trans;
                            transl2.get_face_nodes(
                                face_nr_l,
                                &conn_el[iel, 0],
                                face_nodes.data()
                            );

                            orientationr = 0; // choose the simplest one for the boundary

                            bc_idx = ndim + idim;
                            auto faceB = std::make_unique<FaceType>(
                                iel, -1, face_nodes, face_nr_l, face_nr_r,
                                orientationr, bctypes[bc_idx], bcflags[bc_idx]
                            );
#ifndef NDEBUG
                            std::cout << "Boundary Face B |" << " iel: " << std::setw(3) <<  faceB->elemL
                                << " | ier: " << std::setw(3) << faceB->elemR
                                << " | #: " << face_nr_l << " | orient: " << orientationr << " | nodes [ ";
                            for(int i = 0; i < FaceType::trans.n_nodes; ++i){
                                std::cout << std::setw(3) << face_nodes[i] << " ";
                            }
                            std::cout << "]" 
                                " | bctype: " << bc_name(faceB->bctype) 
                                << " | bcflag" << std::setw(2) << faceB->bcflag << std::endl;
#endif // !DEBUG

                            // Take care of periodic bc 
                            if(bctypes[idim] == BOUNDARY_CONDITIONS::PERIODIC){
                                // get the global face indices 
                                IDX faceA_idx = faces.size();
                                IDX faceB_idx = faceA_idx + 1;

                                // assign the bcflag to be the periodic face index 
                                faceA->bcflag = faceB_idx;
                                faceB->bcflag = faceA_idx;
                            }

                            // add to the face list 
                            faces.emplace_back(std::move(faceA));
                            faces.emplace_back(std::move(faceB));

                            // reset ordinate of this direction
                            ijk[idim] = 0;

                            // increment the ordinates 
                            int first_dir = (idim == 0) ? 1 : 0;
                            if constexpr(ndim > 1) ++ijk[first_dir];
                            for(int jdim = first_dir; jdim < ndim - 1; ++jdim){
                                if(jdim == idim){
                                    // skip over the boundary normal direction
                                } else if(ijk[jdim] == directional_nelem[jdim]){
                                    ijk[jdim] = 0;
                                    ++ijk[jdim + 1];
                                } else {
                                    // short circuit
                                    break;
                                }
                            }
                        }
                    }
                    bdyFaceStart = interiorFaceEnd;
                    bdyFaceEnd = faces.size();
                }
            });
            // EXITING ORDER TEMPLATED SECTION

            // set up additional connectivity array
            elsup = to_elsup(conn_el);

            // faces surrounding elements
            std::vector<std::vector<IDX>> facsuel_ragged(nelem);
            for(IDX iel = 0; iel < nelem; ++iel){
                facsuel_ragged[iel].resize(el_transformations[iel]->nnode);
            }
            for(IDX ifac = interiorFaceStart; ifac < interiorFaceEnd; ++ifac) {
                facsuel_ragged[faces[ifac]->elemL][faces[ifac]->face_nr_l()] = ifac;
                facsuel_ragged[faces[ifac]->elemR][faces[ifac]->face_nr_r()] = ifac;
            }
            for(IDX ifac = bdyFaceStart; ifac < bdyFaceEnd; ++ifac) {
                facsuel_ragged[faces[ifac]->elemL][faces[ifac]->face_nr_l()] = ifac;
            }
            facsuel = util::crs<IDX, IDX>{facsuel_ragged};
        } 

        /**
         * @brief generate a uniform mesh of n-dimensional hypercubes
         * aligned with the axis
         * @param xmin the [-1, -1, ..., -1] corner of the domain
         * @param xmax the [1, 1, ..., 1] corner of the domain
         * @param directional_nelem, the number of elements in each coordinate direction
         * @param order the polynomial order of the hypercubes
         * @param bctypes the boundary conditions for each face of the whole domain,
         *                following the hypercube numbering convention
         *                i.e the coordinate direction index (x: 0, y:1, z:2, ...) = face_number % ndim
         *                the negative side face is face_number / ndim == 0, and positive side otherwise 
         *                so for 2d this would be: 
         *                0: left face 
         *                1: bottom face 
         *                2: right face 
         *                3: top face
         *
         * @param bcflags the boundary condition flags for each face of the whole domain,
         *                same layout
         */
        template<
            std::ranges::random_access_range R_xmin,
            std::ranges::random_access_range R_xmax,
            std::ranges::random_access_range R_nelem,
            std::ranges::random_access_range R_bctype,
            std::ranges::random_access_range R_bcflags
        >
        AbstractMesh(
            iceicle::tmp::from_range_t,
            R_xmin&& xmin, 
            R_xmax&& xmax,
            R_nelem&& directional_nelem,
            int order,
            R_bctype&& bctypes,
            R_bcflags&& bcflags
        ) requires(
            std::convertible_to<std::ranges::range_value_t<R_xmin>, T> &&
            std::convertible_to<std::ranges::range_value_t<R_xmax>, T> &&
            std::convertible_to<std::ranges::range_value_t<R_nelem>, IDX> &&
            std::same_as<std::ranges::range_value_t<R_bctype>, BOUNDARY_CONDITIONS> &&
            std::convertible_to<std::ranges::range_value_t<R_bcflags>, int>
        ) : AbstractMesh<T, IDX, ndim>(generate_directional_nodes<ndim>(xmin, xmax, directional_nelem, order),
                order, bctypes, bcflags) {}

        /// @brief default argument version of uniform mesh constructor 
        template<
            std::ranges::random_access_range R_xmin,
            std::ranges::random_access_range R_xmax,
            std::ranges::random_access_range R_nelem
        >
        AbstractMesh(
            tmp::from_range_t range_arg,
            R_xmin&& xmin, 
            R_xmax&& xmax,
            R_nelem&& directional_nelem,
            int order = 1
        ) : AbstractMesh(range_arg, xmin, xmax, directional_nelem, order, all_periodic, all_zero) {}

        /// @brief version of uniform mesh constructor using Tensor
        /// so that initializer lists can be used for each range 
        AbstractMesh(
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> &xmin,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> &xmax,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, ndim> &directional_nelem,
            int order = 1,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<BOUNDARY_CONDITIONS, 2*ndim> &bctypes = all_periodic,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2*ndim> &bcflags = all_zero
        ): AbstractMesh(tmp::from_range_t{}, xmin, xmax, directional_nelem, order, bctypes, bcflags) {}

        /// @brief get the number of nodes 
        inline constexpr
        auto n_nodes() const -> std::make_unsigned_t<IDX> { return coord.size(); }

        // ===========
        // = Utility =
        // ===========

        /// @brief update the element coordinate data to match the coord array 
        /// by using the element connectivity
        void update_coord_els(){
            for(IDX i = 0; i < conn_el.size(); ++i){
                IDX inode = conn_el.dof_connectivity.data()[i];
                coord_els.data()[i] = coord[inode];
            }
        }

        /// @brief element coordinate data for all elements affected by the given node
        void update_node(IDX inode) {
            for(IDX iel : elsup.rowspan(inode)){
                for(int ilocal = 0; ilocal < conn_el.rowsize(iel); ++ilocal){
                    if(conn_el[iel, ilocal] == inode)
                        coord_els[iel, ilocal] = coord[conn_el[iel, ilocal]];
                }
            }
        }

        /// @brief get a span of the node indices for the given element
        [[nodiscard]] inline constexpr
        auto get_el_nodes(IDX ielem) noexcept
        -> std::span<IDX> 
        { return conn_el.rowspan(ielem); }

        /// @brief get a span of the node coordinates for the given element
        [[nodiscard]] inline constexpr 
        auto get_el_coord(IDX ielem) noexcept 
        -> std::span<Point>
        { return coord_els.rowspan(ielem); }

        // ================
        // = Diagonostics =
        // ================
        void printNodes(std::ostream &out){
            for(IDX inode = 0; inode < coord.size(); ++inode){
                out << "Node: " << inode << " { ";
                for(int idim = 0; idim < ndim; ++idim)
                    out << coord[inode][idim] << " ";
                out << "}" << std::endl;
            }
        }

        void printElements(std::ostream &out){
            for(IDX iel = 0; iel < nelem(); ++iel){
                out << "Element: " << iel << "\n";
                out << "Nodes: { ";
                for(IDX inode : conn_el.rowspan(iel)){
                    out << inode << " ";
                }
                out << "}\n";
            }
        }

        void printFaces(std::ostream &out){
            out << "\nInterior Faces\n";
            for(int ifac = interiorFaceStart; ifac < interiorFaceEnd; ++ifac){
                face_t &fac = *(faces[ifac]);
                out << "Face index: " << ifac << "\n";
                out << "Nodes: { ";
                std::span<const IDX> nodeslist = fac.nodes_span();
                for(int inode = 0; inode < fac.n_nodes(); ++inode){
                    out << nodeslist[inode] << " ";
                }
                out << "}\n";
                out << "ElemL: " << fac.elemL << " | ElemR: " << fac.elemR << "\n"; 
                out << "FaceNrL: " << fac.face_infoL / FACE_INFO_MOD << " | FaceNrR: " << fac.face_infoR / FACE_INFO_MOD << "\n";
                out << "bctype: " << bc_name(fac.bctype) << " | bcflag: " << fac.bcflag << std::endl;
                out << "-------------------------\n";
           }

            out << "\nBoundary Faces\n";
            for(int ifac = bdyFaceStart; ifac < bdyFaceEnd; ++ifac){
                face_t &fac = *(faces[ifac]);
                out << "Face index: " << ifac << "\n";
                out << "Nodes: { ";
                std::span<const IDX> nodeslist = fac.nodes_span();
                for(int inode = 0; inode < fac.n_nodes(); ++inode){
                    out << nodeslist[inode] << " ";
                }
                out << "}\n";
                out << "ElemL: " << fac.elemL << " | ElemR: " << fac.elemR << "\n"; 
                out << "FaceNrL: " << fac.face_infoL / FACE_INFO_MOD << " | FaceNrR: " << fac.face_infoR / FACE_INFO_MOD << "\n";
                out << "bctype: " << bc_name(fac.bctype) << " | bcflag: " << fac.bcflag << std::endl;
                out << "-------------------------\n";
           }
        }

        ~AbstractMesh() = default;
    };
}
