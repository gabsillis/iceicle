/**
 * @file mesh.hpp
 * @brief Abstract mesh definitionn
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @date 2023-06-27
 */
#pragma once
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <memory>
#include <ostream>
#include <string>
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
         *                defaults to periodic
         * @param bcflags the boundary condition flags for each face of the whole domain,
         *                same layout
         */
        AbstractMesh(
            T xmin[ndim], 
            T xmax[ndim],
            IDX directional_nelem[ndim],
            int order = 1,
            ELEMENT::BOUNDARY_CONDITIONS bctypes[2 * ndim] = {ELEMENT::BOUNDARY_CONDITIONS::PERIODIC},
            int bcflags[2 * ndim] = {0}
        ) : nodes{}, elements{}, faces{} {

            // determine the number of nodes to generate
            int nnodes = 1;
            int nelem = 1;
            int nnode_dir[ndim];
            T dx[ndim];
            for(int idim = 0; idim < ndim; ++idim) {
                nnode_dir[idim] = directional_nelem[idim] * (order) + 1;
                nnodes *= nnode_dir;
                nelem *= directional_nelem[idim];
                dx[idim] = (xmax[idim] - xmin[idim]) / directional_nelem[idim];
            }
            nodes.resize(nnodes);

            // Generate the nodes 
            int ijk[ndim] = {0};
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

           // ENTERING ORDER TEMPLATED SECTION
           // here we find the compile time function to call based on the order input 
            NUMTOOL::TMP::constexpr_for_range<1, ELEMENT::MAX_DYNAMIC_ORDER + 1>([&]<int Pn>{
                using namespace ELEMENT;
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
                        HypercubeElement<T, IDX, ndim, Pn> *el = new HypercubeElement<T, IDX, ndim, Pn>();

                        // get the nodes 
                        auto &trans = el->transformation;
                        for(int inode = 0; inode < trans.n_nodes(); ++inode){
                            int iglobal = 0;
                            for(int idim = 0; idim < ndim; ++idim){
                                iglobal += ijk[idim] * order + trans.ijk_poin(inode)[idim];
                            }
                            el->setNode(inode, iglobal);
                        }

                        // assign it 
                        elements[ielem] = el;
                    }

                    // form all the faces
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
