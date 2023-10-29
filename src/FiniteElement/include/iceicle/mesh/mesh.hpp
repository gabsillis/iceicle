/**
 * @file mesh.hpp
 * @brief Abstract mesh definitionn
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @date 2023-06-27
 */
#pragma once
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <memory>
#include <ostream>
#include <string>
#include "mfem.hpp"
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

        /// A list of unique pointers of geometric elements
        std::vector<std::unique_ptr<Element>> elements;

        /// index of the start of interior faces (interior faces must be consecutive)
        IDX interiorFaceStart;
        /// index of one past the end of interior faces
        IDX interiorFaceEnd; 
        // index of the start of the boundary faces (must be consecutive)
        IDX bdyFaceStart;
        /// index of one past the end of the boundary faces
        IDX bdyFaceEnd;
        
        /// All faces (internal and boundary) 
        /// interior faces must be a contiguous set
        std::vector<std::unique_ptr<Face>> faces;

        inline IDX nelem() { return elements.size(); }

        // ===============
        // = Constructor =
        // ===============

        /** @brief construct an empty mesh */
        AbstractMesh() 
        : nodes{}, elements{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0), faces{} {}
        
        AbstractMesh(int nnode) 
        : nodes{nnode}, elements{}, interiorFaceStart(0), interiorFaceEnd(0), 
          bdyFaceStart(0), bdyFaceEnd(0), faces{} {}

        AbstractMesh(mfem::FiniteElementSpace &mfem_mesh);

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
                out << "FaceNrL: " << fac.faceNrL << " | FaceNrR: " << fac.faceNrR << "\n";
                out << "orientL: " << fac.orientationL << " | orientR: " << fac.orientationR << "\n";
                out << "-------------------------\n";
           }
        }
    };
}
