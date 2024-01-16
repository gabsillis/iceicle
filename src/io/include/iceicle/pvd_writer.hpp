/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <ostream>
#include <vector>
#include <string>
namespace ICEICLE::IO {

    template<typename T, typename IDX, int ndim>
    class PVDWriter{

    };

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
    }

}
