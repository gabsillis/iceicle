/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <ostream>
#include <vector>
#include <string>
#include <iceicle/mesh/mesh.hpp>
namespace ICEICLE::IO {

    namespace impl{

        struct XMLField{
            std::string name;
            std::string value;
        };

        /** @brief write an xml field */
        void write(const XMLField &field, std::ostream &out);

        struct XMLTag {
            std::string name;
            std::vector<XMLField> fields; 
        };

        /** @brief write an opening XML tag <tagname fields...>*/
        void write_open(const XMLTag &tag, std::ostream &out);

        /**@brief write an empty XML tag <tagname fields... />*/
        void write_empty(const XMLTag &tag, std::ostream &out);

        /**@brief write a closing XML tag </tagname> */
        void write_close(const XMLTag &tag, std::ostream &out);

        /** write the VTKFile header tag */
        void write_vtu_header(std::ostream &out);

        /** close the VTKFile header tag */
        void write_vtu_footer(std::ostream &out);

        // =====================
        // = VTK Element Types =
        // =====================

        template<typename T, int ndim>
        struct VTKElement {
            std::vector<MATH::GEOMETRY::Point<T, ndim>> nodes;

            int vtk_id;
        };

        // === Linear 2D ===

        template<typename T>
        static VTKElement<T, 2> VTK_TRIANGLE = {
            .nodes = {
                { 1.0,  0.0},
                { 0.0,  1.0},
                { 0.0,  0.0},
            },
            .vtk_id = 5
        };

        template<typename T>
        static VTKElement<T, 2> VTK_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
            },
            .vtk_id = 9
        };

        // === Linear 3D ===

        template<typename T>
        static VTKElement<T, 3> VTK_TETRA = {
            .nodes = {
                { 1.0, 0.0, 0.0},
                { 0.0, 1.0, 0.0},
                { 0.0, 0.0, 1.0},
                { 0.0, 0.0, 0.0},
            },
            .vtk_id = 10
        };


        template<typename T>
        static VTKElement<T, 3> VTK_HEXAHEDRON  = {
            .nodes = {
                {-1.0,-1.0,-1.0},
                { 1.0,-1.0,-1.0},
                { 1.0, 1.0,-1.0},
                {-1.0, 1.0,-1.0},
                {-1.0,-1.0, 1.0},
                { 1.0,-1.0, 1.0},
                { 1.0, 1.0, 1.0},
                {-1.0, 1.0, 1.0},
            },
            .vtk_id = 10
        };
    }

    template<typename T, typename IDX, int ndim>
    class PVDWriter{
        MESH::AbstractMesh<T, IDX, ndim> *meshptr;
        
        public:
        PVDWriter(MESH::AbstractMesh<T, IDX, ndim> *meshptr)
        : meshptr(meshptr) {}

        void write_mesh(std::ostream &out){
            using namespace impl;
            write_vtu_header(out);
           

            write_vtu_footer(out);
        }
    };


}
