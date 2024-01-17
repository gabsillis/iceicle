/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/fe_enums.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iceicle/mesh/mesh.hpp>

#include <iomanip>
#include <fstream>
#include <filesystem>
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

        /**
         * @brief get the VTKElement based on the Element 
         */
        template<typename T, typename IDX, int ndim>
        VTKElement<T, ndim> &get_vtk_element(const ELEMENT::GeometricElement<T, IDX, ndim> *el){
            if constexpr (ndim == 2){
                switch(el->domain_type()){

                    // Triangle type elements
                    case FE::DOMAIN_TYPE::SIMPLEX:
                        switch(el->geometry_order()){
                            case 1:
                                return VTK_TRIANGLE<T>;
                            default:
                                return VTK_TRIANGLE<T>;
                        }
                   
                    // Quad type elements 
                    case FE::DOMAIN_TYPE::HYPERCUBE:
                        switch (el->geometry_order()) {
                            case 1:
                                return VTK_QUAD<T>;
                            default:
                                return VTK_QUAD<T>;
                        }
                    // default case
                    default:
                        throw std::logic_error("Unsupported element domain for vtu writer.");
                        return VTK_QUAD<T>;
                }

            } else if constexpr(ndim == 3) {

                switch(el->domain_type()){

                    // Triangle type elements
                    case FE::DOMAIN_TYPE::SIMPLEX:
                        switch(el->geometry_order()){
                            case 1:
                                return VTK_TETRA<T>;
                            default:
                                return VTK_TETRA<T>;
                        }
                   
                    // Quad type elements 
                    case FE::DOMAIN_TYPE::HYPERCUBE:
                        switch (el->geometry_order()) {
                            case 1:
                                return VTK_HEXAHEDRON<T>;
                            default:
                                return VTK_HEXAHEDRON<T>;
                        }
                    // default case
                    default:
                        throw std::logic_error("Unsupported element domain for vtu writer.");
                        return VTK_HEXAHEDRON<T>;
                }
            } else {
                throw std::logic_error("Unsupported dimension for pvd writer.");
                return VTK_HEXAHEDRON<T>;
            }
        }
    }

    template<typename T, typename IDX, int ndim>
    class PVDWriter{
        MESH::AbstractMesh<T, IDX, ndim> *meshptr;
        
        public:

        int print_precision = 8;
        std::string collection_name = "iceicle_data";
        std::filesystem::path data_directory;

        PVDWriter(MESH::AbstractMesh<T, IDX, ndim> *meshptr)
        : meshptr(meshptr), data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        void register_mesh(MESH::AbstractMesh<T, IDX, ndim> *newptr){
            meshptr = newptr;
        }

        void write_mesh(){
            using namespace impl;
            using Element = ELEMENT::GeometricElement<T, IDX, ndim>;

            // create the path if it doesn't exist
            std::filesystem::create_directories(data_directory);

            // create the mesh file 
            std::filesystem::path mesh_path = data_directory;
            mesh_path /= "mesh.vtu";

            std::ofstream out{mesh_path};
            if(!out) {
                throw std::logic_error("could not open mesh file for writing.");
            }

            if(!meshptr){
                throw std::logic_error("mesh doesn't exist");
            }

            // setup the output stream 
            out << std::setprecision(print_precision);

            write_vtu_header(out);
            
            // count the number of nodes (duplicate for each element)
            std::size_t nodecount = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                nodecount += vtk_el.nodes.size();
            }

            write_open(XMLTag{"UnstructuredGrid"}, out);
            write_open(XMLTag{"Piece", {
                {"NumberOfPoints", std::to_string(nodecount)},
                {"NumberOfCells", std::to_string(meshptr->elements.size())}
            }}, out);

            // ===================
            // = write the nodes =
            // ===================

            write_open(XMLTag{"Points"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Float64"},
                {"Name", "Points"},
                {"NumberOfComponents", "3"},
                {"format", "ascii"}
            }}, out);

            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                MATH::GEOMETRY::Point<T, ndim> point_phys;
                for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                    // interpolate the point from the vtk element 
                    elptr->transform(meshptr->nodes, refnode, point_phys);

                    // print the physical node
                    for(int idim = 0; idim < ndim; ++idim)
                    { out << point_phys[idim] << " "; }

                    // fill with zeros
                    for(int idim = ndim; idim < 3; ++idim)
                    { out << 0.0 << " "; }
                    out << std::endl;
                }
            }
            write_close(XMLTag{"DataArray"}, out);
            write_close(XMLTag{"Points"}, out);

            // ===============
            // = write cells =
            // ===============
            
            // === connectivity ===
            write_open(XMLTag{"Cells"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "connectivity"},
                {"format", "ascii"}
            }}, out);
            
            std::size_t ignode = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                    out << std::to_string(ignode++) << " ";
                }
                out << std::endl;
            }
            write_close(XMLTag{"DataArray"}, out);

            // === offsets ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "offsets"},
                {"format", "ascii"}
            }}, out);
            std::size_t goffset = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                goffset += vtk_el.nodes.size();
                out << std::to_string(goffset) << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            // === cell types ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "types"},
                {"format", "ascii"}
            }}, out);
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                out << vtk_el.vtk_id  << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            write_close(XMLTag{"Cells"}, out);

            write_close(XMLTag{"Piece"}, out);
            write_close(XMLTag{"UnstructuredGrid"}, out);
            write_vtu_footer(out);
        }
    };


}
