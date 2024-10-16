#pragma once

#include "iceicle/fespace/fespace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include <fstream>
#include <fmt/core.h>
#include <filesystem>
#include <memory>
#include <string>
namespace iceicle::io {

    template<class T, class IDX, int ndim>
    class DatWriter{

        struct writeable_field {
            virtual void write_data(std::ofstream &out, FESpace<T, IDX, ndim> &fespace) const = 0;

            virtual auto clone() const -> std::unique_ptr<writeable_field> = 0;

            virtual ~writeable_field(){}
        };

        template< class LayoutPolicy, class AccessorPolicy>
        struct DataField final : public writeable_field {
            /// the data view 
            fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            /// @brief constructor with argument forwarding for the vector constructor
            template<class... VecArgs>
            DataField(fespan<T, LayoutPolicy, AccessorPolicy> fedata, VecArgs&&... vec_args)
            : fedata(fedata), field_names({std::forward<VecArgs>(vec_args)...}){}

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &out, FESpace<T, IDX, ndim> &fespace) const override
            {
                using Element = FiniteElement<T, IDX, ndim>;
                constexpr int field_width = 18;
                constexpr int precision = 10;
                constexpr int npoin = 30;
                if constexpr(ndim == 1){
                    // headers
                    out << fmt::format("{:>{}}", "x", field_width);
                    for(const std::string &s : field_names){
                        out << " " << fmt::format("{:>{}}", s, field_width);
                    }
                    out << std::endl;

                    // loop over the elements
                    for(Element &el : fespace.elements){
                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());

                        for(int ipoin = 0; ipoin < npoin; ++ipoin){
                            MATH::GEOMETRY::Point<T, ndim> refnode{-1.0 + 2.0 / (npoin - 1) * ipoin};
                            MATH::GEOMETRY::Point<T, ndim> physnode = el.transform(refnode);
                            out << fmt::format("{:{}.{}e}", physnode[0], field_width, precision);

                            for(IDX ifield = 0; ifield < field_names.size(); ++ifield){
                                el.eval_basis(refnode, basis_data.data());
                                T field_value = 0;
                                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                    field_value += fedata[el.elidx, idof, ifield] 
                                        * basis_data[idof];
                                }
                                out << " " << fmt::format("{:>{}.{}e}", field_value, field_width, precision);
                            }

                            out << std::endl;
                        }
                        // add an extra linebreak after each element 
                        // so that gnuplot can plot in line segments per element
                        out << std::endl;
                    }
                }
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<DataField<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };


        template< class LayoutPolicy, class AccessorPolicy>
        struct Endpoints final : public writeable_field {
            /// the data view 
            fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            /// @brief constructor with argument forwarding for the vector constructor
            template<class... VecArgs>
            Endpoints(fespan<T, LayoutPolicy, AccessorPolicy> fedata, VecArgs&&... vec_args)
            : fedata(fedata), field_names({std::forward<VecArgs>(vec_args)...}){}

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &out, FESpace<T, IDX, ndim> &fespace) const override
            {
                using Element = FiniteElement<T, IDX, ndim>;
                constexpr int field_width = 18;
                constexpr int precision = 10;
                constexpr int npoin = 2;
                if constexpr(ndim == 1){
                    // headers
                    out << fmt::format("{:>{}}", "x", field_width);
                    for(const std::string &s : field_names){
                        out << " " << fmt::format("{:>{}}", s, field_width);
                    }
                    out << std::endl;

                    // loop over the elements
                    for(Element &el : fespace.elements){
                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());

                        for(int ipoin = 0; ipoin < npoin; ++ipoin){
                            MATH::GEOMETRY::Point<T, ndim> refnode{-1.0 + 2.0 / (npoin - 1) * ipoin};
                            MATH::GEOMETRY::Point<T, ndim> physnode = el.transform(refnode);
                            out << fmt::format("{:{}.{}e}", physnode[0], field_width, precision);

                            for(IDX ifield = 0; ifield < field_names.size(); ++ifield){
                                el.eval_basis(refnode, basis_data.data());
                                T field_value = 0;
                                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                    field_value += fedata[el.elidx, idof, ifield] 
                                        * basis_data[idof];
                                }
                                out << " " << fmt::format("{:>{}.{}e}", field_value, field_width, precision);
                            }

                            out << std::endl;
                        }
                        // add an extra linebreak after each element 
                        // so that gnuplot can plot in line segments per element
                        out << std::endl;
                    }
                }
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<Endpoints<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };

        private:
        AbstractMesh<T, IDX, ndim> *meshptr;
        FESpace<T, IDX, ndim> *fespace_ptr;
        std::vector<std::unique_ptr<writeable_field>> fields;

        public:
        using value_type = T;

        std::string collection_name = "iceicle_data";
        std::filesystem::path data_directory;

        DatWriter(FESpace<T, IDX, ndim> &fespace)
        : fespace_ptr{&fespace}, meshptr{fespace.meshptr}, data_directory{std::filesystem::current_path()}{
            data_directory /= "iceicle_data";
        }

        DatWriter(const DatWriter<T, IDX, ndim>& other) 
            : meshptr(other.meshptr), fespace_ptr(other.fespace_ptr), fields{}, 
              collection_name(other.collection_name), data_directory(other.data_directory) 
        {
            for(const std::unique_ptr<writeable_field>& field : other.fields){
                fields.push_back(field->clone());
            }
        }

        DatWriter(DatWriter<T, IDX, ndim>&& other) = default;

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
            assert(fedata.nv() == sizeof...(field_names));

            // create the field handle and add it to the list
            auto field_ptr = std::make_unique<DataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, std::forward<FieldNameTs>(field_names)...);
            fields.push_back(std::move(field_ptr));
            auto field_ptr2 = std::make_unique<Endpoints<LayoutPolicy, AccessorPolicy>>(
                    fedata, std::forward<FieldNameTs>(field_names)...);
            fields.push_back(std::move(field_ptr2));
        }


         /**
         * @brief register a set of fields represented in an fespan 
         * @param fedata the global data view to write to files 
         * @param field_names the names for each field in fe_data
         */
        template< class LayoutPolicy, class AccessorPolicy>
        void register_fields(fespan<T, LayoutPolicy, AccessorPolicy> &fedata, std::vector<std::string> field_names){
            // make sure the size matches
            if(field_names.size() != fedata.nv()){
                // make automated names to ensure size is right
                field_names.clear();
                for(int iv = 0; iv < fedata.nv(); ++iv){
                    field_names.push_back("u" + std::to_string(iv));
                }
            }

            // create the field handle and add it to the list
            auto field_ptr = std::make_unique<DataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, field_names);
            fields.push_back(std::move(field_ptr));
            auto field_ptr2 = std::make_unique<Endpoints<LayoutPolicy, AccessorPolicy>>(
                    fedata, field_names);
            fields.push_back(std::move(field_ptr2));
        }

        void write_dat(int itime, T time){

            if(fespace_ptr == nullptr) {
                throw std::logic_error("fespace pointer not set");
            }


            std::filesystem::create_directories(data_directory);

            for(int i = 0; i < fields.size(); ++i){
                auto &field = *(fields[i]);
                std::filesystem::path field_path = data_directory;
                std::string name = (i % 2 == 0) ? "fieldset" : "endpoints";
                field_path /= (name + std::to_string(i / 2)
                        + "_rank" + std::to_string(mpi::mpi_world_rank())
                        + "_i" + std::to_string(itime)
                        + ".dat");

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
