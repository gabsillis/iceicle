/**
 * @file mesh.hpp
 * @brief Abstract mesh definitionn
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @date 2023-06-27
 */
#pragma once
#include "iceicle/geometry/hypercube_face.hpp"
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <memory>
#include <ostream>
#include <string>
#include <cassert>
#include <iceicle/fe_function/nodal_fe_function.hpp>
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

        /**
         * @brief generate a uniform mesh of n-dimensional hypercubes
         * aligned with the axis
         * @param xmin the [-1, -1, ..., -1] corner of the domain
         * @param xmax the [1, 1, ..., 1] corner of the domain
         * @param directional nelem, the number of elements in each coordinate direction
         * @param order the polynomial order of the hypercubes
         * @param bctypes the boundary conditions for each face of the whole domain,
         *                following the hypercube numbering convention
         *                TODO: defaults to periodic
         *
         * @param bcflags the boundary condition flags for each face of the whole domain,
         *                same layout
         */
        AbstractMesh(
            T xmin[ndim], 
            T xmax[ndim],
            IDX directional_nelem[ndim],
            int order = 1,
            ELEMENT::BOUNDARY_CONDITIONS bctypes[2 * ndim] = ELEMENT::BOUNDARY_CONDITIONS::PERIODIC, 
            int bcflags[2 * ndim] = 0
        ) : nodes{}, elements{}, faces{} {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;

            // determine the number of nodes to generate
            int nnodes = 1;
            int nelem = 1;
            IDX nnode_dir[ndim];
            IDX directional_prod[ndim] = {1};
            T dx[ndim];
            for(int idim = 0; idim < ndim; ++idim) {
                nnode_dir[idim] = directional_nelem[idim] * (order) + 1;
                nnodes *= nnode_dir[idim];
                nelem *= directional_nelem[idim];
                for(int jdim = idim; jdim < ndim; ++jdim) directional_prod[jdim] *= directional_nelem[idim];
                dx[idim] = (xmax[idim] - xmin[idim]) / directional_nelem[idim];
            }
            nodes.resize(nnodes);

            // Generate the nodes 
            IDX ijk[ndim] = {0};
            for(int inode = 0; inode < nnodes; ++inode){
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

                // calculate the coordinates 
                for(int idim = 0; idim < ndim; ++idim){
                    nodes[inode][idim] = xmin[idim] + ijk[idim] * dx[idim];
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

                        // create the element 
                        ElementType *el = new ElementType(); 

                        // get the nodes 
                        auto &trans = el->transformation;
                        for(IDX inode = 0; inode < trans.n_nodes(); ++inode){
                            IDX iglobal = 0;
                            for(int idim = 0; idim < ndim; ++idim){
                                iglobal += ijk[idim] * order + trans.ijk_poin(inode)[idim];
                            }
                            el->setNode(inode, iglobal);
                        }

                        // assign it 
                        elements[ielem] = el;
                    }

                    // ===========================
                    // = Interior Face Formation =
                    // ===========================
                    //
                    // form all the interior faces
                    // first calculate the total number
                    interiorFaceStart = 0;
                    interiorFaceEnd = 1;
                    IDX niface_1direction = 1;
                    IDX nface_total = 1;
                    for(int idim = 0; idim < ndim; ++idim) {
                        interiorFaceEnd *= ndim * (directional_nelem[idim] - 1);
                        niface_1direction *= directional_nelem[idim] - 1;
                        nface_total *= ndim * (directional_nelem[idim] + 1);
                    }

                    // reset the ordinates
                    for(int idim = 0; idim < ndim; ++idim) ijk[idim] = 0;

                    // reserve space
                    faces.reserve(nface_total);

                    // loop over all the interior faces
                    for(IDX ifac = 0; ifac < niface_1direction; ++ifac){
                        // increment the oordinates
                        ++ijk[0];
                        for(int idim = 0; idim < ndim; ++idim){
                            if(ijk[idim] == directional_nelem[idim] - 1){
                                ijk[idim] = 0;
                                ++ijk[idim + 1];
                            } else {
                                // short circuit
                                break;
                            }
                        }

                        // ijk represents the left element
                        // loop over three face directions
                        for(int idim = 0; idim < ndim; ++idim){
                            IDX ijk_r[ndim];
                            std::copy_n(ijk, ndim, ijk_r);
                            ijk_r[idim]++; // increment in the face direction

                            // get the element number from the ordinates
                            IDX iel = 0;
                            IDX ier = 0;
                            for(int jdim = 0; jdim < ndim; ++jdim){
                                iel += ijk[jdim] * directional_prod[jdim];
                                ier += ijk_r[jdim] * directional_prod[jdim];
                            }

                            // get the face numbers
                            int face_nr_l = ndim + idim; // positive side
                            int face_nr_r = idim;

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
                            static constexpr int nfacevert = MATH::power_T<2, ndim>::value;
                            IDX vert_l[nfacevert];
                            IDX vert_r[nfacevert];
                            transl.get_face_vert(face_nr_l, elements[iel]->nodes(), vert_l);
                            transr.get_face_vert(face_nr_r, elements[ier]->nodes(), vert_r);
                            int orientationr = FaceType::orient_trans.getOrientation(vert_l, vert_r);

                            faces.push_back(new FaceType(
                                iel, ier, face_nodes, face_nr_l, face_nr_r,
                                orientationr, BOUNDARY_CONDITIONS::INTERIOR, 0));

                        }

                        // record the number of interior faces
                        if(interiorFaceEnd != faces.size()){
                            throw std::logic_error("Incorrect number of interior faces.");
                        } 

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
                                // increment the ordinates 
                                int first_dir = (idim == 0) ? 1 : 0;
                                ++ijk[first_dir];
                                for(int jdim = 0; jdim < ndim; ++jdim){
                                    if(jdim == idim){
                                        // skip over the boundary normal direction
                                    } else if(ijk[jdim] == directional_nelem[jdim] - 1){
                                        ijk[jdim] = 0;
                                        ++ijk[jdim + 1];
                                    } else {
                                        // short circuit
                                        break;
                                    }
                                }

                                // form the -1 face 
                                // get the element number from the ordinates
                                IDX iel = 0;
                                for(int jdim = 0; jdim < ndim; ++jdim){
                                    iel += ijk[jdim] * directional_prod[jdim];
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

                                FaceType *faceA = new FaceType(
                                    iel, -1, face_nodes, face_nr_l, face_nr_r,
                                    orientationr, bctypes[idim], bcflags[idim]
                                );

                                // form the +1 face
                                // set to the farthest element
                                ijk[idim] = directional_nelem[idim] - 1; 
                                iel = 0;
                                for(int jdim = 0; jdim < ndim; ++jdim){
                                    iel += ijk[jdim] * directional_prod[jdim];
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
                                                  
                                FaceType *faceB = new FaceType(
                                    iel, -1, face_nodes, face_nr_l, face_nr_r,
                                    orientationr, bctypes[idim], bcflags[idim]
                                );

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
                            }
                        }
                        bdyFaceStart = interiorFaceEnd;
                        bdyFaceEnd = faces.size();

                    }
                }
            });
            // EXITING ORDER TEMPLATED SECTION
        } 

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
