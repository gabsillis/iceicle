/// @brief Utilities for dealing with gmsh input files 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/mesh/mesh.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/string_utils.hpp"
#include <istream>
#include <sstream>
#include <string>
#include <map>

namespace iceicle {

    namespace impl::gmsh {
        enum class READER_STATE {
            TOP_LEVEL,
            HEADER,
            ENTITIES,
        };

        /// @brief the gmsh header information that contains file metadata
        struct Header {
            unsigned int version_major;
            unsigned int version_minor;
            bool ascii;
            size_t data_size;
        };

        inline constexpr
        auto read_header(std::string line) -> Header{
            unsigned int version_major;
            unsigned int version_minor;
            int ascii_arg;
            std::size_t data_size;
            sscanf(line.c_str(), "%d.%d %d %ld", &version_major, &version_minor, &ascii_arg, &data_size);
            return Header{version_major, version_minor, ascii_arg == 0, data_size};
        }

        /// @brief metadata about which entities are in the mesh
        struct EntitiesInfo {
            std::size_t npoints;
            std::size_t ncurves;
            std::size_t nsurfaces;
            std::size_t nvolumes;
        };

        ///@brief maps for entities to their corresponding tags
        /// The first tag will be used to define the boundary condition
        struct TagMaps {
            std::map<std::size_t, std::vector<std::size_t>> pt_map;
            std::map<std::size_t, std::vector<std::size_t>> curv_map;
            std::map<std::size_t, std::vector<std::size_t>> surf_map;
            std::map<std::size_t, std::vector<std::size_t>> vol_map;
        };

        inline
        auto read_entities(
            std::string line,
            std::size_t& line_no,
            std::istream& infile
        ) -> TagMaps {
            using namespace impl::gmsh;
            using namespace util;

            EntitiesInfo entities_info;
            TagMaps tag_maps{};

            // get the metadata
            sscanf(line.c_str(), "%ld %ld %ld %ld", &entities_info.npoints, &entities_info.ncurves, 
                    &entities_info.nsurfaces, &entities_info.nvolumes);

            // get the point tags
            for(int iline = 0; iline < entities_info.npoints && std::getline(infile, line); iline++, line_no++){
                int pt_tag;
                double x, y, z;
                std::size_t nphys_tag;
                std::istringstream linestream{line};
                linestream >> pt_tag >> x >> y >> z >> nphys_tag;
                for(int itag = 0; itag < nphys_tag; ++itag){
                    int tag;
                    linestream >> tag;
                    tag_maps.pt_map[pt_tag].push_back(tag);
                }
            }

            // get the curve tags (omit bounding points info)
            for(int iline = 0; iline < entities_info.ncurves && std::getline(infile, line); iline++, line_no++){
                int curv_tag;
                double xmin, xmax, ymin, ymax, zmin, zmax;
                std::size_t nphys_tag;
                std::istringstream linestream{line};
                linestream >> curv_tag >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax >> nphys_tag;
                for(int itag = 0; itag < nphys_tag; ++itag){
                    int tag;
                    linestream >> tag;
                    tag_maps.curv_map[curv_tag].push_back(tag);
                }
            }

            // get the surface tags (omit bounding curve info)
            for(int iline = 0; iline < entities_info.nsurfaces && std::getline(infile, line); iline++, line_no++){
                int surf_tag;
                double xmin, xmax, ymin, ymax, zmin, zmax;
                std::size_t nphys_tag;
                std::istringstream linestream{line};
                linestream >> surf_tag >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax >> nphys_tag;
                for(int itag = 0; itag < nphys_tag; ++itag){
                    int tag;
                    linestream >> tag;
                    tag_maps.surf_map[surf_tag].push_back(tag);
                }
            }

            // get the volume tags (omit bounding surface info)
            for(int iline = 0; iline < entities_info.nvolumes && std::getline(infile, line); iline++, line_no++){
                int vol_tag;
                double xmin, xmax, ymin, ymax, zmin, zmax;
                std::size_t nphys_tag;
                std::istringstream linestream{line};
                linestream >> vol_tag >> xmin >> ymin >> zmin >> xmax >> ymax >> zmax >> nphys_tag;
                for(int itag = 0; itag < nphys_tag; ++itag){
                    int tag;
                    linestream >> tag;
                    tag_maps.vol_map[vol_tag].push_back(tag);
                }
            }
            return tag_maps;
        }
    }

    template<class T, class IDX, int ndim>
    auto read_gmsh(std::istream& infile) -> AbstractMesh<T, IDX, ndim> {
        using namespace impl::gmsh;
        using namespace util;

        // setup for parsing
        std::size_t line_no = 0;
        READER_STATE state = READER_STATE::HEADER; 
        
        // data 
        AbstractMesh<T, IDX, ndim> mesh{};
        Header header;
        TagMaps tag_maps{};

        // parse line by line
        for(std::string line; std::getline(infile, line);){

            switch(state){
                case READER_STATE::TOP_LEVEL:
                    if(eq_icase(line, "$MeshFormat")){
                        state = READER_STATE::HEADER;
                    }
                    if(eq_icase(line, "$Entities")){
                        state = READER_STATE::ENTITIES;
                    }

                    break;

                // Read in the header
                case READER_STATE::HEADER:

                    if(eq_icase(line, "$EndMeshFormat")){
                        // return to top level 
                        state = READER_STATE::TOP_LEVEL;
                    } else {
                        header = read_header(line);

                        // check for errors
                        if(!header.ascii) {
                            AnomalyLog::log_anomaly(Anomaly{"Currently only supports ASCII gmsh file types", file_parse_tag{line_no}});
                        }
                        if(header.version_major < 4) {
                            AnomalyLog::log_anomaly(Anomaly{"must be version 4.1 or greater", file_parse_tag{line_no}});
                        } else if (header.version_minor < 1) {
                            AnomalyLog::log_anomaly(Anomaly{"must be version 4.1 or greater", file_parse_tag{line_no}});
                        }
                    }
                    break;

                // read in entities
                case READER_STATE::ENTITIES:
                    if(eq_icase(line, "$EndEntities")){
                        state = READER_STATE::TOP_LEVEL;
                    } else {
                        tag_maps = read_entities(line, line_no, infile);            
                    }
                    break;

                default:
                    
            }

            line_no++;
        }

    }
}
