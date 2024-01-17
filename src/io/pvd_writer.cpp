#include <iceicle/pvd_writer.hpp>
#include <ostream>
namespace ICEICLE::IO {

    namespace impl {

        void write(const XMLField &field, std::ostream &out){
            out << field.name << "=\"" << field.value << "\"";
        }
        
        void write_open(const XMLTag &tag, std::ostream &out){
            out << "<" << tag.name;
            for(const XMLField &f : tag.fields){
                out << " ";
                write(f, out);
            }
            out << ">" << std::endl;
        }

        void write_empty(const XMLTag &tag, std::ostream &out){
            out << "<" << tag.name;
            for(const XMLField &f : tag.fields){
                out << " ";
                write(f, out);
            }
            out << "/>" << std::endl;;
        }

        void write_close(const XMLTag &tag, std::ostream &out){
            out << "</" << tag.name << ">" << std::endl;
        }

        void write_vtu_header(std::ostream &out){
            
            XMLTag vtkfiletag = {.name="VTKFile", .fields = {
                {"type", "UnstructuredGrid"},
                {"version", "0.1"},
                {"byte_order", "LittleEndian"},
                {"header_type", "UInt64"}
            }};

            write_open(vtkfiletag, out);
        }

        void write_vtu_footer(std::ostream &out){
            out << "</VTKFile>" << std::endl;
        }
    }
}
