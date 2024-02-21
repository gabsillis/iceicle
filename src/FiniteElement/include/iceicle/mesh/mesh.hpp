/**
 * @file mesh.hpp
 * @brief Abstract mesh definitionn
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @date 2023-06-27
 */
#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/geometry/hypercube_face.hpp"
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <ostream>
#include <iceicle/fe_function/nodal_fe_function.hpp>
#include <iomanip>
namespace MESH {

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
        using Element = ELEMENT::GeometricElement<T, IDX, ndim>;
        using Face = ELEMENT::Face<T, IDX, ndim>;
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        // ===========================
        // = Primary Data Structures =
        // ===========================

        /// The node coordinates
        FE::NodalFEFunction<T, ndim> nodes;

        /// A list of pointers to geometric elements
        /// These are owned by the mesh and destroyed when the mesh is destroyed
        std::vector<Element *> elements;

        /// All faces (internal and boundary) 
        /// interior faces must be a contiguous set
        std::vector<Face *> faces;

        /// index of the start of interior faces (interior faces must be consecutive)
        IDX interiorFaceStart;
        /// index of one past the end of interior faces
        IDX interiorFaceEnd; 
        // index of the start of the boundary faces (must be consecutive)
        IDX bdyFaceStart;
        /// index of one past the end of the boundary faces
        IDX bdyFaceEnd;

        inline IDX nelem() { return elements.size(); }

        // ===============
        // = Constructor =
        // ===============

        /** @brief construct an empty mesh */
        AbstractMesh() 
        : nodes{}, elements{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0), faces{} {}
        
        AbstractMesh(std::size_t nnode) 
        : nodes{nnode}, elements{}, faces{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0) {}

        private:

        inline static constexpr NUMTOOL::TENSOR::FIXED_SIZE::Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2*ndim>
        all_periodic = [](){
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2*ndim> ret;
            for(int i = 0; i < 2*ndim; ++i) ret[i] = ELEMENT::BOUNDARY_CONDITIONS::PERIODIC;
            return ret;
        }();

        public:
        /**
         * @brief generate a uniform mesh of n-dimensional hypercubes
         * aligned with the axis
         * @param xmin the [-1, -1, ..., -1] corner of the domain
         * @param xmax the [1, 1, ..., 1] corner of the domain
         * @param directional nelem, the number of elements in each coordinate direction
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
        AbstractMesh(
            const T xmin[ndim], 
            const T xmax[ndim],
            const IDX directional_nelem[ndim],
            int order,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes,
            const int bcflags[2 * ndim]
        ) : nodes{}, elements{}, faces{} {
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
                dx[idim] = (xmax[idim] - xmin[idim]) / directional_nelem[idim];
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
            NUMTOOL::TMP::constexpr_for_range<1, ELEMENT::MAX_DYNAMIC_ORDER + 1>([&]<int Pn>{
                using namespace ELEMENT;
                using FaceType = HypercubeFace<T, IDX, ndim, Pn>;
                using ElementType = HypercubeElement<T, IDX, ndim, Pn>;
                if(order == Pn){

                    // form all the elements 
                    for(int idim = 0; idim < ndim; ++idim) ijk[idim] = 0;
                    for(int ielem = 0; ielem < nelem; ++ielem){
                        // create the element 
                        ElementType *el = new ElementType(); 

                        // get the nodes 
                        auto &trans = el->transformation;
                        for(IDX inode = 0; inode < trans.n_nodes(); ++inode){
                            IDX iglobal = 0;
                            IDX ijk_gnode[ndim];
                            for(int idim = 0; idim < ndim; ++idim){
                                ijk_gnode[idim] = ijk[idim] * order + trans.ijk_poin[inode][idim];
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
                        elements[ielem] = el;

                        // increment
                        ++ijk[0];
                        for(int idim = 0; idim < ndim; ++idim){
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

                            faces.push_back(new FaceType(
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
                            FaceType *faceA = new FaceType(
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
                            FaceType *faceB = new FaceType(
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
                            faces.push_back(faceA);
                            faces.push_back(faceB);

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

        /**
         * Overload for uniform mesh generation to support initializer lists
         */
        AbstractMesh(
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> &xmin,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim> &xmax,
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<IDX, ndim> &directional_nelem,
            int order = 1,
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes = all_periodic, 
            const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2* ndim> &bcflags = [](){
                NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, 2 * ndim> ret{};
                for(int i = 0; i < 2 * ndim; ++i) ret[i] = 0;
                return ret;
            }()
        ) : AbstractMesh(xmin.data(), xmax.data(), directional_nelem.data(), order, bctypes, bcflags.data()) {}

//        TODO: Do this in a separate file so we can build without mfem 
//        AbstractMesh(mfem::FiniteElementSpace &mfem_mesh);

        /** @brief construct a mesh from file (currently supports gmsh) */
        AbstractMesh(std::string_view filepath);

        // ================
        // = Diagonostics =
        // ================
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
                Face &fac = *(faces[ifac]);
                out << "Face index: " << ifac << "\n";
                if constexpr(ndim == 2){
                    MATH::GEOMETRY::Point<T, 1> s = {0};
                    Point normal;
                    fac.getNormal(nodes, s, normal);
                    out << "normal at s=0: (" <<  normal[0] << ", " << normal[1] << ")\n";
                }
                out << "Nodes: { ";
                IDX *nodeslist = fac.nodes();
                for(int inode = 0; inode < fac.n_nodes(); ++inode){
                    out << nodeslist[inode] << " ";
                }
                out << "}\n";
                out << "ElemL: " << fac.elemL << " | ElemR: " << fac.elemR << "\n"; 
                out << "FaceNrL: " << fac.face_infoL / ELEMENT::FACE_INFO_MOD << " | FaceNrR: " << fac.face_infoR / ELEMENT::FACE_INFO_MOD << "\n";
                out << "-------------------------\n";
           }
        }

        ~AbstractMesh(){
            for(Element *el_ptr : elements){
                delete el_ptr;
            }

            for(Face *fac_ptr : faces){
                delete fac_ptr;
            }
        }
    };
}
