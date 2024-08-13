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
#include "iceicle/geometry/hypercube_face.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <iceicle/crs.hpp>
#include <optional>
#include <ostream>
#include <ranges>
#include <type_traits>
#include <memory>
#ifndef NDEBUG
#include <iomanip>
#endif
namespace iceicle {

    /// @brief generate the node connectivity matrix in crs format from the element list
    /// elsup -> elements surrounding points
    /// @param element_list 
    template<class T, class IDX, int ndim>
    constexpr
    auto to_elsup(const std::span< const std::unique_ptr< GeometricElement<T, IDX, ndim> > > element_list)
    -> util::crs<IDX, IDX>
    {
        std::vector<IDX> cols{};
        cols.reserve(element_list.size() + 1);
        IDX icol = 0;
        cols.push_back(icol);
        for(const auto& el : element_list){
            icol += el->n_nodes();
            cols.push_back(icol);
        }
        return util::crs<IDX, IDX>{cols};
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

            default:
                return std::nullopt;
                break;
        }
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
        using Element = GeometricElement<T, IDX, ndim>;
        using face_t = Face<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        // ===========================
        // = Primary Data Structures =
        // ===========================

        /// The node coordinates
        NodeArray<T, ndim> nodes;

        /// A list of pointers to geometric elements
        /// These are owned by the mesh and destroyed when the mesh is destroyed
        std::vector<std::unique_ptr<Element>> elements;

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

        /// For each process i store a list of (this-local) element indices 
        /// that need to be sent 
        std::vector<std::vector<IDX>> el_send_list;

        /// For each process i store a list of (i-local) element indices 
        /// that need to be recieved
        std::vector<std::vector<IDX>> el_recv_list;

        std::vector< std::vector< std::unique_ptr<Element> > > communicated_elements;

        inline IDX nelem() { return elements.size(); }

        // ===============
        // = Constructor =
        // ===============

        /** @brief construct an empty mesh */
        AbstractMesh() 
        : nodes{}, elements{}, faces{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0), el_send_list(mpi::mpi_world_size()), 
          el_recv_list(mpi::mpi_world_size()), communicated_elements(mpi::mpi_world_size()){}

        AbstractMesh(const AbstractMesh<T, IDX, ndim>& other) 
        : nodes{other.nodes}, elements{}, faces{},
          interiorFaceStart(other.interiorFaceStart), interiorFaceEnd(other.interiorFaceEnd),
          bdyFaceStart(other.bdyFaceStart), bdyFaceEnd(other.bdyFaceEnd), 
          el_send_list(other.el_send_list), el_recv_list(other.el_recv_list)
        {
            elements.reserve(other.elements.size());
            faces.reserve(other.faces.size());
            for(auto& elptr : other.elements){
                elements.push_back(std::move(elptr->clone()));
            }
            for(auto& facptr : other.faces){
                faces.push_back(std::move(facptr->clone()));
            }

            for(int irank = 0; irank < other.communicated_elements.size(); ++irank){
                communicated_elements.push_back(std::vector< std::unique_ptr<Element> >{});
                for(auto& elptr : other.communicated_elements[irank]){
                    communicated_elements[irank].push_back(std::move(elptr->clone()));
                }
            }
        }

        AbstractMesh<T, IDX, ndim>& operator=(const AbstractMesh<T, IDX, ndim>& other){
            if(this != &other){
                nodes = other.nodes;
                elements.clear();
                faces.clear();
                elements.reserve(other.elements.size());
                faces.reserve(other.faces.size());
                for(auto& elptr : other.elements){
                    elements.push_back(std::move(elptr->clone()));
                }
                for(auto& facptr : other.faces){
                    faces.push_back(std::move(facptr->clone()));
                }
                interiorFaceStart = other.interiorFaceStart;
                interiorFaceEnd = other.interiorFaceEnd;
                bdyFaceStart = other.bdyFaceStart;
                bdyFaceEnd = other.bdyFaceEnd;
                el_send_list = other.el_send_list;
                el_recv_list = other.el_recv_list;

                for(int irank = 0; irank < other.communicated_elements.size(); ++irank){
                    communicated_elements.push_back(std::vector< std::unique_ptr<Element> >{});
                    for(auto& elptr : other.communicated_elements[irank]){
                        communicated_elements[irank].push_back(std::move(elptr->clone()));
                    }
                }
            }
            return *this;
        }

        AbstractMesh(std::size_t nnode) 
        : nodes{nnode}, elements{}, faces{}, interiorFaceStart(0), interiorFaceEnd(0), 
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

        ) : nodes{}, elements{}, faces{}, el_send_list(mpi::mpi_world_size()), 
          el_recv_list(mpi::mpi_world_size()), communicated_elements(mpi::mpi_world_size()) 
        {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;

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
                dx[idim] = (xmax[idim] - xmin[idim]) / (directional_nelem[idim] * order);
            }

            for(int idim = 0; idim < ndim; ++idim){
                for(int jdim = 0; jdim < idim; ++jdim){
                    stride[idim] *= directional_nelem[jdim];
                    stride_nodes[idim] *= nnode_dir[jdim];
                }
            }
            nodes.resize(nnodes);

            // Generate the nodes 
            IDX ijk[ndim] = {0};
            for(int inode = 0; inode < nnodes; ++inode){
                // calculate the coordinates 
                for(int idim = 0; idim < ndim; ++idim){
                    nodes[inode][idim] = xmin[idim] + ijk[idim] * dx[idim];
                }
#ifndef NDEBUG
                // print out the node 
                std::cout << "node " << inode << ": [ ";
                for(int idim = 0; idim < ndim; ++idim){
                    std::cout << nodes[inode][idim] << " ";
                }
                std::cout << "]" << std::endl;
#endif

                // increment
                ++ijk[0];
                for(int idim = 0; idim < ndim; ++idim){
                    if(ijk[idim] == nnode_dir[idim]){
                        ijk[idim] = 0;
                        ++ijk[idim + 1];
                    } else {
                        // short circuit
                        break;
                    }
                }
            }

            elements.resize(nelem);

           // ENTERING ORDER TEMPLATED SECTION
           // here we find the compile time function to call based on the order input 
            NUMTOOL::TMP::constexpr_for_range<1, MAX_DYNAMIC_ORDER + 1>([&]<int Pn>{
                using FaceType = HypercubeFace<T, IDX, ndim, Pn>;
                using ElementType = HypercubeElement<T, IDX, ndim, Pn>;
                if(order == Pn){

                    // form all the elements 
                    for(int idim = 0; idim < ndim; ++idim) ijk[idim] = 0;
                    for(int ielem = 0; ielem < nelem; ++ielem){
                        // create the element 
                        auto el = std::make_unique<ElementType>(); 

                        // get the nodes 
                        auto &trans = el->transformation;
                        for(IDX inode = 0; inode < trans.n_nodes(); ++inode){
                            IDX iglobal = 0;
                            IDX ijk_gnode[ndim];
                            for(int idim = 0; idim < ndim; ++idim){
                                ijk_gnode[idim] = ijk[idim] * order + trans.tensor_prod.ijk_poin[inode][idim];
                            }

                            for(int idim = 0; idim < ndim; ++idim){
                                iglobal += ijk_gnode[idim] * stride_nodes[idim];
                            }
                            el->setNode(inode, iglobal);
                        }

#ifndef NDEBUG 
                        std::cout << "Element " << ielem << ": [ ";
                        for(int inode = 0; inode < trans.n_nodes(); ++inode){
                            std::cout << el->nodes()[inode] << " ";
                        }
                        std::cout << "]" << std::endl;
#endif

                        // assign it 
                        elements[ielem] = std::move(el);

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

                    // ===========================
                    // = Interior Face Formation =
                    // ===========================
                    
                    // loop over each direction 
                    for(int idir = 0; idir < ndim; ++idir){

                        // oordinates of the left element
                        int ijk[ndim] = {0};

                        //function to increment the ijk of the left element 
                        auto next_ijk = [&](int ijk[ndim]) -> bool {
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
                            IDX ijk_r[ndim];
                            std::copy_n(ijk, ndim, ijk_r);
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
                            auto &transl = ElementType::transformation;
                            auto &transr = ElementType::transformation;
                            transl.get_face_nodes(
                                face_nr_l,
                                elements[iel]->nodes(),
                                face_nodes.data()
                            );

                            // get the orientations
                            static constexpr int nfacevert = MATH::power_T<2, ndim-1>::value;
                            IDX vert_l[nfacevert];
                            IDX vert_r[nfacevert];
                            transl.get_face_vert(face_nr_l, elements[iel]->nodes(), vert_l);
                            transr.get_face_vert(face_nr_r, elements[ier]->nodes(), vert_r);
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
                            auto &transl = ElementType::transformation;
                            transl.get_face_nodes(
                                face_nr_l,
                                elements[iel]->nodes(),
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
                            auto &transl2 = ElementType::transformation;
                            transl2.get_face_nodes(
                                face_nr_l,
                                elements[iel]->nodes(),
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
                            ++ijk[first_dir];
                            for(int jdim = first_dir; jdim < ndim; ++jdim){
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
        } 


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
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> xmin,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> xmax,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, ndim> directional_nelem,
            int order = 1,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<BOUNDARY_CONDITIONS, 2*ndim> bctypes = all_periodic,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2*ndim> bcflags = all_zero
        ): AbstractMesh(tmp::from_range_t{}, xmin, xmax, directional_nelem, order, bctypes, bcflags) {}

//        TODO: Do this in a separate file so we can build without mfem 
//        AbstractMesh(mfem::FiniteElementSpace &mfem_mesh);

        /** @brief construct a mesh from file (currently supports gmsh) */
        AbstractMesh(std::string_view filepath);

        /// @brief get the number of nodes 
        inline constexpr
        auto n_nodes() const -> std::make_unsigned_t<IDX> { return nodes.size(); }

        // ================
        // = Diagonostics =
        // ================
        void printNodes(std::ostream &out){
            for(IDX inode = 0; inode < nodes.size(); ++inode){
                out << "Node: " << inode << " { ";
                for(int idim = 0; idim < ndim; ++idim)
                    out << nodes[inode][idim] << " ";
                out << "}" << std::endl;
            }
        }

        void printElements(std::ostream &out){
            int iel = 0;
            for(auto &elptr : elements){
                out << "Element: " << iel << "\n";
                out << "Nodes: { ";
                for(int inode = 0; inode < elptr->n_nodes(); ++inode){
                    out << elptr->nodes()[inode] << " ";
                }
                out << "}\n";
                ++iel;
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
