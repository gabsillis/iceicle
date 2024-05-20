/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/anomaly_log.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/fespace/fespace.hpp>

#include <cassert>
#include <iomanip>
#include <fstream>
#include <filesystem>
namespace iceicle::io {

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
        VTKElement<T, ndim> &get_vtk_element(const GeometricElement<T, IDX, ndim> *el, int basis_order = 1){

            static VTKElement<T, ndim> NO_ELEMENT{};

            int max_order = std::max(el->geometry_order(), basis_order);
            if constexpr (ndim == 2){
                switch(el->domain_type()){

                    // Triangle type elements
                    case DOMAIN_TYPE::SIMPLEX:
                        switch(max_order){
                            case 1:
                                return VTK_TRIANGLE<T>;
                            default:
                                return VTK_TRIANGLE<T>;
                        }
                   
                    // Quad type elements 
                    case DOMAIN_TYPE::HYPERCUBE:
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
                    case DOMAIN_TYPE::SIMPLEX:
                        switch(max_order){
                            case 1:
                                return VTK_TETRA<T>;
                            default:
                                return VTK_TETRA<T>;
                        }
                   
                    // Quad type elements 
                    case DOMAIN_TYPE::HYPERCUBE:
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
                return NO_ELEMENT; 
            }
        }
    }


    template<typename T, typename IDX, int ndim>
    class PVDWriter{

        /**
        * @brief a data field for output 
        * overridable write_data function
        */
        struct writeable_field {
            /// @brief adds the xml DataArray tags and data to a given vtu file 
            virtual void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim> &fespace) const = 0;

            /// @brief if this data is in dg format and requires duplicated mesh nodes
            virtual auto is_dg_format() const -> bool { return true; }

            virtual auto clone() const -> std::unique_ptr<writeable_field> = 0;

            virtual ~writeable_field(){};
        };

        /**
        * @brief a data field for output 
        * contains a name list that will be the field names in VTK 
        * and an fespan that points to the data 
        */
        template< class LayoutPolicy, class AccessorPolicy>
        struct PVDDataField final : public writeable_field {
            /// the data view
            fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            /// @brief constructor with argument forwarding for the vector constructor
            template<class... VecArgs>
            PVDDataField(fespan<T, LayoutPolicy, AccessorPolicy> fedata, VecArgs&&... vec_args)
            : fedata(fedata), field_names({std::forward<VecArgs>(vec_args)...}){}

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim> &fespace) const override
            {
                using namespace impl;
                using Element = FiniteElement<T, IDX, ndim>;
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
                                field_value += fedata[el.elidx, idof, ifield] 
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

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<PVDDataField<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };

        /**
         * @brief a data field for MDG nodal vector data output 
         * contains field_name
         * and a node_selection_span for the data 
         *
         * writes nodal data for a nodal version of the mesh 
         * WARNING: do not couple with dg mesh
         */
        template< class LayoutPolicy, class AccessorPolicy>
        struct MDGVectorDataField final : public writeable_field {
            using value_type = T; 
            using index_type = LayoutPolicy::index_type;
            ///  the data view 
            mutable dofspan<T, LayoutPolicy, AccessorPolicy> mdgdata;
            static_assert(node_selection_span<decltype(mdgdata)>, 
                    "Must be a node selection span to be mdg data");

            /// the field name 
            std::string field_name;

            /// @brief constructor with argument forwarding for the vector constructor
            MDGVectorDataField(dofspan<value_type, LayoutPolicy, AccessorPolicy> mdgdata, std::string field_name)
            : mdgdata(mdgdata), field_name(field_name){}

            void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim>& fespace) const override {
                using namespace impl;
                const nodeset_dof_map<index_type>& nodeset = mdgdata.get_layout().nodeset;

                write_open(XMLTag{"DataArray", {
                    {"type", "Float64"},
                    {"Name", field_name}, 
                    {"NumberOfComponents", std::to_string(mdgdata.nv())},
                    {"format", "ascii"}
                }}, vtu_file);

                for(index_type inode = 0; inode < fespace.meshptr->n_nodes(); ++inode){
                    index_type idof = nodeset.inv_selected_nodes[inode];
                    if(idof == nodeset.selected_nodes.size()){
                        for(index_type iv = 0; iv < mdgdata.nv(); ++iv){
                            vtu_file << "0.0 ";
                        }
                        vtu_file << std::endl;
                    } else {
                        for(index_type iv = 0; iv < mdgdata.nv(); ++iv){
                            vtu_file << mdgdata[idof, iv] << " ";
                        }
                        vtu_file << std::endl;
                    }
                }
                write_close(XMLTag{"DataArray"}, vtu_file);
            }

            auto is_dg_format() const -> bool override {
                return false;
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<MDGVectorDataField<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };

        private:

        AbstractMesh<T, IDX, ndim> *meshptr = nullptr;
        FESpace<T, IDX, ndim> *fespace_ptr = nullptr;
        std::vector<std::unique_ptr<writeable_field>> fields;

        public:
        using value_type = T;

        int print_precision = 8;
        std::string collection_name = "data";
        std::filesystem::path data_directory;

        PVDWriter() : data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        PVDWriter(AbstractMesh<T, IDX, ndim> *meshptr)
        : meshptr(meshptr), data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        PVDWriter(const PVDWriter<T, IDX, ndim>& other)
            : meshptr(other.meshptr), fespace_ptr(other.fespace_ptr), fields{}, print_precision(other.print_precision),
              collection_name(other.collection_name), data_directory(other.data_directory)
        {
            for(const std::unique_ptr<writeable_field>& field : other.fields){
                fields.push_back(field->clone());
            }
        }

        PVDWriter(PVDWriter<T, IDX, ndim>&& other) = default;
              

        void register_mesh(AbstractMesh<T, IDX, ndim> *newptr){
            meshptr = newptr;
        }

        /// @brief register an fespace to this writer 
        /// will overwrite the registered mesh to the one in the fespace
        void register_fespace(FESpace<T, IDX, ndim> &fespace){
            fespace_ptr = &fespace;
            meshptr = fespace.meshptr;
        }

        /**
         * @brief register a set of fields represented in an fespan 
         * @param fedata the global data view to write to files 
         * @param field_names the names for each field in fe_data
         */
        template< class LayoutPolicy, class AccessorPolicy, class... FieldNameTs >
        void register_fields(fespan<T, LayoutPolicy, AccessorPolicy> &fedata, FieldNameTs&&... field_names){
            // make sure the size matches
            assert(fedata.get_layout().nv() == sizeof...(field_names));

            // create the field handle and add it to the list
            auto field_ptr = std::make_unique<PVDDataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, std::forward<FieldNameTs>(field_names)...);
            fields.push_back(std::move(field_ptr));
        }

        template< class LayoutPolicy, class AccessorPolicy>
        void register_fields(dofspan<T, LayoutPolicy, AccessorPolicy>& nodal_data, std::string_view field_name){
            auto field_ptr = std::make_unique<MDGVectorDataField<LayoutPolicy, AccessorPolicy>>(
                    nodal_data, std::string{field_name});
            fields.push_back(std::move(field_ptr));
        }

        private:
        auto write_cg_unstructured_grid(std::ofstream& out){
            using namespace impl;
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;
            write_open(XMLTag{"Piece", {
                {"NumberOfPoints", std::to_string(meshptr->n_nodes())},
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

            for(IDX inode = 0; inode < meshptr->n_nodes(); ++inode){
                for(int idim = 0; idim < ndim; ++idim){
                    out << meshptr->nodes[inode][idim] << " ";
                }
                for(int idim = ndim; idim < 3; ++idim){
                    out << 0.0;
                }
                out << std::endl;
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
            
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.geo_el->geometry_order());
                if(vtk_el.nodes.size() != el.geo_el->n_nodes()){
                    AnomalyLog::log_anomaly(Anomaly{"must have a matching vtk element for cg data", general_anomaly_tag{}});
                }

                // TODO: FIX AND GENERALIZE TO OTHER ELEMENT TYPES 
                // convert to paraview ordering
                std::span<const IDX> nodes = el.geo_el->nodes_span();
                out << std::to_string(nodes[0]) << " "
                    << std::to_string(nodes[2]) << " "
                    << std::to_string(nodes[3]) << " "
                    << std::to_string(nodes[1]) << " ";

//                for(IDX inode : el.geo_el->nodes_span()){
//                    out << std::to_string(inode) << " ";
//                }
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
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.geo_el->geometry_order());
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
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.geo_el->geometry_order());
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
        }

        auto write_dg_unstructured_grid(std::ofstream &out) -> void {
            using namespace impl;
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;
            // count the number of nodes (duplicate for each element)
            std::size_t nodecount = 0;
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.geo_el, el.basis->getPolynomialOrder());
                nodecount += vtk_el.nodes.size();
            }
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

        }

        public:
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
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;

            // create the path if it doesn't exist
            std::filesystem::create_directories(data_directory);

            // create the mesh file 
            std::filesystem::path mesh_path = data_directory;
            mesh_path /= (collection_name + std::to_string(itime) + ".vtu");

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
            
            bool use_dg_mesh = fields[0]->is_dg_format();

            for(int ifield = 1; ifield < fields.size(); ++ifield){
                if(fields[ifield]->is_dg_format() != use_dg_mesh){
                    AnomalyLog::log_anomaly(Anomaly{"Cannot mix dg and cg fields", general_anomaly_tag{}});
                }
            }

            write_open(XMLTag{"UnstructuredGrid"}, out);

            if(use_dg_mesh) write_dg_unstructured_grid(out);
            else write_cg_unstructured_grid(out);

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
            using Element = GeometricElement<T, IDX, ndim>;

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
