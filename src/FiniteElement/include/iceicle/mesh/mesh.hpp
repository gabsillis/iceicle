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
        using Point = GEOMETRY::Point<T, ndim>;

        public:
        // ===========================
        // = Primary Data Structures =
        // ===========================

        /// The node coordinates
        std::vector< GEOMETRY::Point<T, ndim> > nodes;

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

        /**
         * @brief precompute geometric quantities
         * e.g. Jacobians
         */
        void computeGeometry(){
            for(auto &elptr : elements) elptr->updateGeometry(nodes);
            for(auto &facptr : faces) facptr->updateGeometry(nodes);
        }

        inline IDX nelem() { return elements.size(); }

        // ================
        // = Diagonostics =
        // ================
        void printElements(std::ostream &out){
            computeGeometry();
            int iel = 0;
            for(auto &elptr : elements){
                out << "Element: " << iel << "\n";
                out << "Nodes: { ";
                for(int inode = 0; inode < elptr->getNNodes(); ++inode){
                    out << elptr->getNodes()[inode] << " ";
                }
                out << "}\n";
                ++iel;
            }
        }

        void printFaces(std::ostream &out){
            computeGeometry();
            out << "\nInterior Faces\n";
            for(int ifac = interiorFaceStart; ifac < interiorFaceEnd; ++ifac){
                Face &fac = *(faces[ifac]);
                out << "Face index: " << ifac << "\n";
                if constexpr(ndim == 2){
                     GEOMETRY::Point<T, 1> s = {0};
                    Point normal;
                    fac.getNormal(nodes, s, normal.data.data());
                    out << "normal at s=0: (" <<  normal[0] << ", " << normal[1] << ")\n";
                }
                out << "Nodes: { ";
                IDX *nodeslist = fac.getNodes();
                for(int inode = 0; inode < fac.nnodes(); ++inode){
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

    enum BOUNDARY_CONDITIONS{
        PERIODIC = 0,
        PARALLEL_COM,
        NEUMANN,
        DIRICHLET,
        RIEMANN,
        NO_SLIP,
        SLIP_WALL,
        WALL_GENERAL, /// General Wall BC, up to the implementation of the pde
        INLET,
        OUTLET,
        INITIAL_CONDITION,
        TIME_UPWIND, /// used for the top of a time slab
        INTERIOR // default condition that does nothing
    };

}
