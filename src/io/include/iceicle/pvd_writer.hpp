/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_enums.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fe_function/layout_enums.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <type_traits>
#include <vector>
#include <string>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/fespace/fespace.hpp>

#include <cassert>
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
            std::vector<XMLField> fields{}; 
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

        template<typename T>
        static VTKElement<T, 2> VTK_QUADRATIC_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
                { 0.0, -1.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
                {-1.0,  0.0}
            },
            .vtk_id = 23
        };

        template<typename T>
        static VTKElement<T, 2> VTK_BIQUADRATIC_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
                { 0.0, -1.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
                {-1.0,  0.0},
                { 0.0,  0.0}
            },
            .vtk_id = 28
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
         * @param el the element 
         * @param basis_order an optional basis order argument
         *        uses the maximum polynomial order between the element and basis order 
         */
        template<typename T, typename IDX, int ndim>
        VTKElement<T, ndim> &get_vtk_element(const ELEMENT::GeometricElement<T, IDX, ndim> *el, int basis_order = 1){
            int max_order = std::max(el->geometry_order(), basis_order);
            if constexpr (ndim == 2){
                switch(el->domain_type()){

                    // Triangle type elements
                    case FE::DOMAIN_TYPE::SIMPLEX:
                        switch(max_order){
                            case 1:
                                return VTK_TRIANGLE<T>;
                            default:
                                return VTK_TRIANGLE<T>;
                        }
                   
                    // Quad type elements 
                    case FE::DOMAIN_TYPE::HYPERCUBE:
                        switch (max_order) {
                            case 0: // use case 1
                            case 1:
                                return VTK_QUAD<T>;
                            case 2:
                                return VTK_BIQUADRATIC_QUAD<T>;
                            default:
                                return VTK_QUADRATIC_QUAD<T>;
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
                        switch(max_order){
                            case 1:
                                return VTK_TETRA<T>;
                            default:
                                return VTK_TETRA<T>;
                        }
                   
                    // Quad type elements 
                    case FE::DOMAIN_TYPE::HYPERCUBE:
                        switch (max_order) {
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

        /**
        * @brief a data field for output 
        * overridable write_data function
        */
        struct writable_field {
            /// @brief adds the xml DataArray tags and data to a given vtu file 
            virtual void write_data(std::ofstream &vtu_file, FE::FESpace<T, IDX, ndim> &fespace) const = 0;

            virtual ~writable_field(){};
        };

        /**
        * @brief a data field for output 
        * contains a name list that will be the field names in VTK 
        * and an fespan that points to the data 
        */
        template< class LayoutPolicy, class AccessorPolicy>
        struct PVDDataField final : public writable_field {
            /// the data view
            FE::fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            /// @brief constructor with argument forwarding for the vector constructor
            template<class... VecArgs>
            PVDDataField(FE::fespan<T, LayoutPolicy, AccessorPolicy> fedata, VecArgs&&... vec_args)
            : fedata(fedata), field_names({std::forward<VecArgs>(vec_args)...}){}

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &vtu_file, FE::FESpace<T, IDX, ndim> &fespace) const override
            {
                using namespace impl;
                using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
                for(std::size_t ifield = 0; ifield < field_names.size(); ++ifield){

                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", field_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());

                        // get the solution for each point in the vtk element
                        for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                            el.evalBasis(refnode, basis_data.data());
                            T field_value = 0;
                            for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                field_value += fedata[FE::fe_index{(std::size_t) el.elidx, idof, ifield}] 
                                    * basis_data[idof];
                            }

                            vtu_file << field_value << " ";
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }

            }
        };

        private:

        MESH::AbstractMesh<T, IDX, ndim> *meshptr;
        FE::FESpace<T, IDX, ndim> *fespace_ptr;
        std::vector<std::unique_ptr<writable_field>> fields;

        public:

        int print_precision = 8;
        std::string collection_name = "iceicle_data";
        std::filesystem::path data_directory;

        PVDWriter() : data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        PVDWriter(MESH::AbstractMesh<T, IDX, ndim> *meshptr)
        : meshptr(meshptr), data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        void register_mesh(MESH::AbstractMesh<T, IDX, ndim> *newptr){
            meshptr = newptr;
        }

        /// @brief register an fespace to this writer 
        /// will overwrite the registered mesh to the one in the fespace
        void register_fespace(FE::FESpace<T, IDX, ndim> &fespace){
            fespace_ptr = &fespace;
            meshptr = fespace.meshptr;
        }

        /**
         * @brief register a set of fields represented in an fespan 
         * @param fedata the global data view to write to files 
         * @param field_names the names for each field in fe_data
         */
        template< class LayoutPolicy, class AccessorPolicy, class... FieldNameTs >
        void register_fields(FE::fespan<T, LayoutPolicy, AccessorPolicy> &fedata, FieldNameTs&&... field_names){
            // make sure the size matches
            assert(fedata.get_layout().get_ncomp() == sizeof...(field_names));

            // create the field handle and add it to the list
            auto field_ptr = std::make_unique<PVDDataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, std::forward<FieldNameTs>(field_names)...);
            fields.push_back(std::move(field_ptr));
        }

        /**
         * @brief write the mesh and field values in a .vtu file 
         * @param itime the timestep
         * @param time the time value
         * NOTE: the user is responsible for making sure itime and time are unique 
         * (aside from parallel case, separate file names are generated per process in parallel)
         */
        void write_vtu(int itime, T time){

            if(fespace_ptr == nullptr) {
                throw std::logic_error("fespace pointer not set");
            }

            using namespace impl;
            using Element = ELEMENT::FiniteElement<T, IDX, ndim>;

            // create the path if it doesn't exist
            std::filesystem::create_directories(data_directory);

            // create the mesh file 
            std::filesystem::path mesh_path = data_directory;
            mesh_path /= ("data." + std::to_string(itime) + ".vtu");

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
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
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

            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
                MATH::GEOMETRY::Point<T, ndim> point_phys;
                for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                    // interpolate the point from the vtk element 
                    el.transform(meshptr->nodes, refnode, point_phys);

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
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
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
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
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
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
                out << vtk_el.vtk_id  << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            write_close(XMLTag{"Cells"}, out);

            // ===================
            // = write PointData =
            // ===================
            write_open(XMLTag{"PointData"}, out);
            for(auto &field_ptr : fields){
                field_ptr->write_data(out, *fespace_ptr);
            }
            write_close(XMLTag{"PointData"}, out);

            write_close(XMLTag{"Piece"}, out);
            write_close(XMLTag{"UnstructuredGrid"}, out);
            write_vtu_footer(out);
        }
        

        /**
         * @brief writes a .vtu file called "mesh.vtu"
         * that only contains the mesh 
         *
         * NOTE: this has no context of the finite element space 
         * so will use the minimum polynomial order to capture the 
         * mesh geometry (may not match up to solutions)
         */
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
