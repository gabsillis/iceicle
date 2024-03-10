#pragma once

#include "iceicle/fespace/fespace.hpp"
#include <fstream>
#include <format>
#include <filesystem>
#include <string>
#include <fstream>
namespace ICEICLE::IO {

    template<class T, class IDX, int ndim>
    class DatWriter{

        struct writable_field {
            virtual void write_data(std::ofstream &out, FE::FESpace<T, IDX, ndim> &fespace) const = 0;

            virtual ~writable_field(){}
        };

        template< class LayoutPolicy, class AccessorPolicy>
        struct DataField final : public writable_field {
            /// the data view 
            FE::fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            /// @brief constructor with argument forwarding for the vector constructor
            template<class... VecArgs>
            DataField(FE::fespan<T, LayoutPolicy, AccessorPolicy> fedata, VecArgs&&... vec_args)
            : fedata(fedata), field_names({std::forward<VecArgs>(vec_args)...}){}

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &out, FE::FESpace<T, IDX, ndim> &fespace) const override
            {
                using Element = ELEMENT::FiniteElement<T, IDX, ndim>;
                constexpr int field_width = 18;
                constexpr int precision = 10;
                constexpr int npoin = 100;
                if constexpr(ndim == 1){
                    // headers
                    out << std::format("{:>{}}", "x", field_width);
                    for(const std::string &s : field_names){
                        out << " " << std::format("{:>{}}", s, field_width);
                    }
                    out << std::endl;

                    // loop over the elements
                    for(Element &el : fespace.elements){
                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());

                        for(int ipoin = 0; ipoin < npoin; ++ipoin){
                            MATH::GEOMETRY::Point<T, ndim> refnode{-1.0 + 2.0 / (npoin - 1) * ipoin};
                            MATH::GEOMETRY::Point<T, ndim> physnode{};
                            el.transform(fespace.meshptr->nodes, refnode, physnode);
                            out << std::format("{:{}.{}e}", physnode[0], field_width, precision);

                            for(std::size_t ifield = 0; ifield < field_names.size(); ++ifield){
                                el.evalBasis(refnode, basis_data.data());
                                T field_value = 0;
                                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                    field_value += fedata[FE::fe_index{(std::size_t) el.elidx, idof, ifield}] 
                                        * basis_data[idof];
                                }
                                out << " " << std::format("{:>{}.{}e}", field_value, field_width, precision);
                            }

                            out << std::endl;
                        }
                    }

                    // add an extra linebreak after each element 
                    // so that gnuplot can plot in line segments per element
                    out << std::endl;
                }
            }
        };

        private:
        MESH::AbstractMesh<T, IDX, ndim> *meshptr;
        FE::FESpace<T, IDX, ndim> *fespace_ptr;
        std::vector<std::unique_ptr<writable_field>> fields;

        public:

        std::string collection_name = "iceicle_data";
        std::filesystem::path data_directory;

        DatWriter(FE::FESpace<T, IDX, ndim> &fespace)
        : fespace_ptr{&fespace}, meshptr{fespace.meshptr}, data_directory{std::filesystem::current_path()}{
            data_directory /= "iceicle_data";
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
            auto field_ptr = std::make_unique<DataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, std::forward<FieldNameTs>(field_names)...);
            fields.push_back(std::move(field_ptr));
        }

        void write_dat(int itime, T time){

            if(fespace_ptr == nullptr) {
                throw std::logic_error("fespace pointer not set");
            }


            std::filesystem::create_directories(data_directory);

            for(int i = 0; i < fields.size(); ++i){
                auto &field = *(fields[i]);
                std::filesystem::path field_path = data_directory;
                field_path /= ("fieldset" + std::to_string(i) +"_i" + std::to_string(itime)
                        + "_t" + std::to_string(time) + ".dat");

                
                std::ofstream out{field_path};
                if(!out) {
                    throw std::logic_error("could not open mesh file for writing.");
                }

                if(!meshptr){
                    throw std::logic_error("mesh doesn't exist");
                }
                field.write_data(out, *fespace_ptr);
            }
        }
    };
}
