/// @brief Utilities for dealing with gmsh input files 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/hypercube_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/string_utils.hpp"
#include <cstdio>
#include <istream>
#include <sstream>
#include <list>
#include <string>
#include <map>

namespace iceicle {

    namespace impl::gmsh {
        enum class READER_STATE {
            TOP_LEVEL,
            HEADER,
            ENTITIES,
            NODES,
            ELEMENTS
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

        template<class T, class IDX, int ndim>
        auto read_nodes(
            std::string line,
            std::size_t& line_no,
            std::istream& infile,
            AbstractMesh<T, IDX, ndim>& mesh
        ) -> void {
            
            // get the metadata 
            std::size_t nblocks, nnodes, min_nodetag, max_nodetag;
            sscanf(line.c_str(), "%ld %ld %ld %ld", &nblocks, &nnodes, &min_nodetag, &max_nodetag);
            mesh.nodes.resize(max_nodetag + 1);

            for(int iblock = 0; iblock < nblocks; ++iblock){
                std::getline(infile, line); 
                line_no++;
                std::istringstream linestream{line};

                // block metadata
                int entity_dim, entity_tag, parametric;
                std::size_t n_blocknodes;
                linestream >> entity_dim >> entity_tag >> parametric >> n_blocknodes;

                std::vector<IDX> node_idxs;
                node_idxs.reserve(n_blocknodes);
                for(IDX inode = 0; inode < n_blocknodes && std::getline(infile, line); ++inode, ++line_no){
                    std::istringstream linestream{line};
                    IDX node_tag;
                    linestream >> node_tag;
                    node_idxs.push_back(node_tag);
                }

                for(IDX inode = 0; inode < n_blocknodes && std::getline(infile, line); ++inode, ++line_no){
                    std::istringstream linestream{line};
                    for(int idim = 0; idim < ndim; ++idim){
                        linestream >> mesh.nodes[node_idxs[inode]][idim];
                    }
                }
            }
        }

        template<class T, class IDX, int ndim>
        auto read_linear_quads(
            std::size_t nelem,
            std::size_t& line_no,
            std::istream& infile,
            AbstractMesh<T, IDX, ndim>& mesh
        ) {
            using namespace util;
            if constexpr (ndim != 2){
                AnomalyLog::log_anomaly(Anomaly{"Quads elements only available in 2D mesh", file_parse_tag{line_no}});
                return;
            }
            std::string line;
            for(int ielem = 0; ielem < nelem && std::getline(infile, line); ++ielem, ++line_no){
                // ordering for tensor prod 0 3 1 2
                IDX inodes[4];
                IDX ielem_global;
                std::istringstream linestream{line};
                linestream >> ielem_global >> inodes[0] >> inodes[1] >> inodes[2] >> inodes[3];

                auto el = std::make_unique<HypercubeElement<T, IDX, ndim, 1>>();
                el->setNode(0, inodes[0]);
                el->setNode(1, inodes[3]);
                el->setNode(2, inodes[1]);
                el->setNode(3, inodes[2]);
                mesh.elements[ielem] = std::move(el);
            }
        }

        /// @brief get the number of nodes for a given gmsh element type
        constexpr inline 
        auto gmsh_nnode(int element_type) -> int {
            switch(element_type){
                case 1:
                    return 2;
                case 2:
                    return 3;
                case 3:
                    return 4;
                case 4:
                    return 4;
                case 5:
                    return 8;
                case 6:
                    return 6;
                case 7:
                    return 5;
                case 8:
                    return 3;
                case 9:
                    return 6;
                case 10:
                    return 9;
                case 11:
                    return 10;
                case 12:
                    return 27;
                case 13:
                    return 18;
                case 14:
                    return 14;
                case 15:
                    return 1;
                default:
                {
                    util::AnomalyLog::log_anomaly(util::Anomaly{
                            "number of nodes not defined for element type: " 
                            + std::to_string(element_type), util::general_anomaly_tag{}});
                    return 0;
                }
            }
        }

        struct bdy_face_parse {
            int entity_tag;
            int element_type;
            std::vector<std::size_t> nodes;
        };

        /// @brief given a boundary face info, create the face and add it to the element
        template<class T, class IDX, int ndim>
        auto create_boundary_faces(
            std::vector<bdy_face_parse>& parsed_info,
            std::map<int, std::tuple<BOUNDARY_CONDITIONS, int>>& bcmap,
            AbstractMesh<T, IDX, ndim>& mesh
        ) -> void {
            using namespace util;
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            // find the faces surrounding elements to reduce search space
            std::vector<std::vector<IDX>> faces_surr_el(mesh.nelem());
            for(IDX ifac = mesh.interiorFaceStart; ifac < mesh.interiorFaceEnd; ++ifac){
                faces_surr_el[mesh.faces[ifac]->elemL].push_back(ifac);
                faces_surr_el[mesh.faces[ifac]->elemR].push_back(ifac);
            }

            //indices of elements that don't have all their faces
            std::vector<IDX> elements_missing_faces{};
            for(IDX ielem = 0; ielem < mesh.nelem(); ++ielem){
                if(faces_surr_el[ielem].size() < mesh.elements[ielem]->n_faces()){
                    elements_missing_faces.push_back(ielem);
                }
            }

            // find the boundary face
            for(bdy_face_parse& fac_info: parsed_info){
                for(IDX ielem : elements_missing_faces){
                    bool found = false;
                    GeometricElement<T, IDX, ndim> *elptr = mesh.elements[ielem].get();
                    switch(fac_info.element_type){
                        // 2 node line
                        case 1:
                        {
                            if constexpr(ndim == 2){
                                if(elptr->domain_type() == DOMAIN_TYPE::HYPERCUBE){
                                    Tensor<IDX, 2> nodes{{(IDX) fac_info.nodes[0], (IDX) fac_info.nodes[1]}};
                                    int face_nr = elptr->get_face_nr(nodes.data());
                                    // get the face nodes again, but from element so they are in order for external normal
                                    elptr->get_face_nodes(face_nr, nodes.data());
                                    if(face_nr >= 0){
                                        found = true;
                                        using Face_t = HypercubeFace<T, IDX, ndim, 1>;
                                        std::tuple<BOUNDARY_CONDITIONS, int> bcinfo = bcmap[fac_info.entity_tag];
                                        auto face = std::make_unique<HypercubeFace<T, IDX, ndim, 1>>(
                                            ielem, ielem, nodes, face_nr, face_nr, 0, std::get<0>(bcinfo), std::get<1>(bcinfo));
                                        mesh.faces.push_back(std::move(face));
                                        faces_surr_el[ielem].push_back(mesh.faces.size() - 1);
                                    }

                                } else {
                                    AnomalyLog::log_anomaly(Anomaly{"unsupported el type", general_anomaly_tag{}});
                                }
                            } 
                        }
                        break;
                        default:
                            AnomalyLog::log_anomaly(Anomaly{"unsupported gmsh element type", general_anomaly_tag{}});
                        break;
                    }
                    if(found) break;
                }
            }

            mesh.bdyFaceStart = mesh.interiorFaceEnd;
            mesh.bdyFaceEnd = mesh.faces.size();
        }

        template<class T, class IDX, int ndim>
        auto read_elements(
            std::string line,
            std::size_t& line_no,
            std::istream& infile,
            std::map<int, std::tuple<BOUNDARY_CONDITIONS, int>>& bcmap,
            AbstractMesh<T, IDX, ndim>& mesh
        ) -> void {

            // get the metadata
            std::size_t nblocks, nelem, min_eltag, max_eltag;
            sscanf(line.c_str(), "%ld %ld %ld %ld", &nblocks, &nelem, &min_eltag, &max_eltag);
            mesh.elements.resize(max_eltag + 1);

            std::vector<bdy_face_parse> boundary_face_infos{};
            for(int iblock = 0; iblock < nblocks; ++iblock){
                std::getline(infile, line); 
                line_no++;
                std::istringstream linestream{line};

                // block metadata
                int entity_dim, entity_tag, element_type;
                std::size_t nelem_block;
                linestream >> entity_dim >> entity_tag >> element_type >> nelem_block;

                if (entity_dim == ndim-1){
                    // these are boundary face definitions
                    // read in the nodes and store with entity tag for later
                    for(int ifac = 0; ifac < nelem_block; ++ifac){
                        int nnode = gmsh_nnode(element_type);
                        std::getline(infile, line);
                        ++line_no;
                        std::istringstream linestream{line};
                        int element_tag;
                        linestream >> element_tag;
                        bdy_face_parse parsed_face{entity_tag, element_type, std::vector<std::size_t>(nnode)};
                        for(int inode = 0; inode < nnode; ++inode){
                            linestream >> parsed_face.nodes[inode];
                        }
                        boundary_face_infos.push_back(parsed_face);
                    }

                } else if (entity_dim == ndim){
                    // these are internal element definitions
                    // dispatch to read specific elements
                    switch(element_type){
                        case 3:
                            read_linear_quads(nelem_block, line_no, infile, mesh);
                        default:
                            util::AnomalyLog::log_anomaly(util::Anomaly{"unsupported element type", util::file_parse_tag{line_no}});

                    }
                }
            }

            // compress the element list for any gaps gmsh put in
            IDX last_el = mesh.elements.size() - 1;
            for(IDX iel = 0; iel < mesh.elements.size(); ++iel){
                if(!mesh.elements[iel]) {
                    // keep moving back from the end until we have a real element to swap in
                    while(mesh.elements[last_el] == nullptr) last_el--;
                    if(last_el > iel)
                        std::swap(mesh.elements[iel], mesh.elements[last_el]);
                }
            }
            if(mesh.elements[last_el]){
                mesh.elements.resize(last_el + 1);
            } else {
                mesh.elements.resize(last_el);
            }

            // generate interior faces
            find_interior_faces(mesh);
            mesh.interiorFaceStart = 0;
            mesh.interiorFaceEnd = mesh.faces.size();

            // generate boundary faces
            create_boundary_faces(boundary_face_infos, bcmap, mesh);
        }
    }

    /// @brief create a mesh from a gmsh file 
    /// @param infile the gmsh file stream to read
    /// @param bcmap maps integer physical tags to boundary condition information
    /// The first physical tag for each boundary entity will be used to determine the boundary condition 
    /// that physical tag should have an entry in the map:
    /// a tuple which contains the boundary condition type and integer boundary flag. 
    template<class T, class IDX, int ndim>
    auto read_gmsh(std::istream& infile, std::map<int, std::tuple<BOUNDARY_CONDITIONS, int>>& bcmap) -> AbstractMesh<T, IDX, ndim> {
        using namespace impl::gmsh;
        using namespace util;

        // setup for parsing
        std::size_t line_no = 0;
        READER_STATE state = READER_STATE::TOP_LEVEL; 
        
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
                    if(eq_icase(line, "$Nodes")){
                        state = READER_STATE::NODES;
                    }
                    if(eq_icase(line, "$Elements")){
                        state = READER_STATE::ELEMENTS;
                    }
                    // otherwise we skip line
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

                case READER_STATE::NODES:
                    if(eq_icase(line, "$EndNodes")){
                        state = READER_STATE::TOP_LEVEL;
                    } else {
                        read_nodes(line, line_no, infile, mesh);
                    }
                    break;

                case READER_STATE::ELEMENTS:
                    if(eq_icase(line, "$EndElements")){
                        state = READER_STATE::TOP_LEVEL;
                    } else {
                        read_elements(line, line_no, infile, bcmap, mesh);
                    }
                default:
                    break;
            }

            line_no++;
        }
        return mesh;
    }
}
