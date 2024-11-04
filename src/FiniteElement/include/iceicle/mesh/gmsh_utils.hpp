/// @brief Utilities for dealing with gmsh input files 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/hypercube_element.hpp"
#include "iceicle/geometry/transformations_table.hpp"
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
            mesh.coord.resize(max_nodetag + 1);

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
                        linestream >> mesh.coord[node_idxs[inode]][idim];
                    }
                }
            }
        }

        template<class IDX>
        [[nodiscard]] 
        auto read_linear_quads(
            std::size_t nelem,
            std::size_t& line_no,
            std::istream& infile
        ) -> std::vector<std::vector<IDX>>
        {
            using namespace util;
            std::vector<std::vector<IDX>> el_conn{};
            std::string line;
            for(int ielem = 0; ielem < nelem && std::getline(infile, line); ++ielem, ++line_no){
                // ordering for tensor prod 0 3 1 2
                IDX inodes[4];
                IDX ielem_global;
                std::istringstream linestream{line};
                linestream >> ielem_global >> inodes[0] >> inodes[1] >> inodes[2] >> inodes[3];

                el_conn.push_back( std::vector<IDX>{
                    inodes[0], inodes[3],
                    inodes[1], inodes[2]
                });
            }
            return el_conn;
        }


        template<class IDX>
        [[nodiscard]] 
        auto read_linear_tris(
            std::size_t nelem,
            std::size_t& line_no,
            std::istream& infile
        ) -> std::vector<std::vector<IDX>> 
        {
            std::vector<std::vector<IDX>> el_conn;
            std::string line;
            for(int ielem = 0; ielem < nelem && std::getline(infile, line); ++ielem, ++line_no){
                // ordering for tensor prod 0 3 1 2
                IDX inodes[3];
                IDX ielem_global;
                std::istringstream linestream{line};
                linestream >> ielem_global >> inodes[0] >> inodes[1] >> inodes[2];

                el_conn.push_back( std::vector<IDX>{
                    inodes[0], inodes[1],
                    inodes[2]
                });
            }
            return el_conn;
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
                if(faces_surr_el[ielem].size() < mesh.el_transformations[ielem]->nfac){
                    elements_missing_faces.push_back(ielem);
                }
            }

            // find the boundary face
            for(bdy_face_parse& fac_info: parsed_info){
                for(IDX ielem : elements_missing_faces){
                    bool found = false;
                    ElementTransformation<T, IDX, ndim> *trans = mesh.el_transformations[ielem];
                    std::span<IDX> el_nodes = mesh.get_el_nodes(ielem);
                    switch(fac_info.element_type){
                        // 2 node line
                        case 1:
                        {
                            if constexpr(ndim == 2){
                                auto[bctype, bcflag] = bcmap[fac_info.entity_tag];
                                std::vector<IDX> nodes{ (IDX) fac_info.nodes[0], (IDX) fac_info.nodes[1]};
                                auto binfo_opt = boundary_face_info(nodes, trans, el_nodes);
                                if(binfo_opt){ // we found the adjacent element yay ^.^
                                    found = true;
                                    auto [domain_type, face_nr] = binfo_opt.value();

                                    // get the face nodes again, but from element so they are in order for external normal
                                    nodes = trans->get_face_nodes(face_nr, el_nodes);
                                    auto face_opt = make_face<T, IDX, ndim>(
                                        domain_type, trans->domain_type, trans->domain_type, 
                                        1, ielem, ielem, nodes, face_nr, face_nr,
                                        0, bctype, bcflag);
                                    mesh.faces.push_back(std::move(face_opt.value()));
                                    faces_surr_el[ielem].push_back(mesh.faces.size() - 1);
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

            std::vector<bdy_face_parse> boundary_face_infos{};
            std::vector<std::vector<IDX>> el_conn_ragged;
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
                        case 2:
                        {
                            auto trans = transformation_table<T, IDX, ndim>.get_transform(DOMAIN_TYPE::SIMPLEX, 1);
                            auto conn = read_linear_tris<IDX>(nelem_block, line_no, infile);
                            el_conn_ragged.insert(el_conn_ragged.end(), conn.begin(), conn.end());
                            for(IDX i = 0; i < nelem_block; ++i)
                                mesh.el_transformations.push_back(trans);
                        }
                        break;
                        case 3:
                        {
                            auto trans = transformation_table<T, IDX, ndim>.get_transform(DOMAIN_TYPE::HYPERCUBE, 1);
                            auto conn = read_linear_quads<IDX>(nelem_block, line_no, infile);
                            el_conn_ragged.insert(el_conn_ragged.end(), conn.begin(), conn.end());
                            for(IDX i = 0; i < nelem_block; ++i)
                                mesh.el_transformations.push_back(trans);
                        }
                        break;
                        default:
                            util::AnomalyLog::log_anomaly(util::Anomaly{"unsupported element type", util::file_parse_tag{line_no}});

                    }
                }
            }

            std::cout << "Compressing element list" << std::endl;
            mesh.conn_el = util::crs<IDX, IDX>(el_conn_ragged);

            // generate interior faces
            std::cout << "Generating interior faces" << std::endl;
            find_interior_faces(mesh);
            mesh.interiorFaceStart = 0;
            mesh.interiorFaceEnd = mesh.faces.size();

            // generate boundary faces
            std::cout << "Generating boundary faces" << std::endl;
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

        std::cout << "Reading gmsh file... " << std::endl;

        // parse line by line
        for(std::string line; std::getline(infile, line);){

            switch(state){
                case READER_STATE::TOP_LEVEL:
                    if(eq_icase(line, "$MeshFormat")){
                        std::cout << "Reading format header" << std::endl;
                        state = READER_STATE::HEADER;
                    }
                    if(eq_icase(line, "$Entities")){
                        std::cout << "Reading entities" << std::endl;
                        state = READER_STATE::ENTITIES;
                    }
                    if(eq_icase(line, "$Nodes")){
                        std::cout << "Reading nodes" << std::endl;
                        state = READER_STATE::NODES;
                    }
                    if(eq_icase(line, "$Elements")){
                        std::cout << "Reading Elements" << std::endl;
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

        mesh.elsup = to_elsup(mesh.conn_el, mesh.n_nodes());
        { // build the element coordinates matrix
            mesh.coord_els = util::crs<MATH::GEOMETRY::Point<T, ndim>, IDX>{
                std::span{mesh.conn_el.cols(), mesh.conn_el.cols() + mesh.conn_el.nrow() + 1}};
            for(IDX iel = 0; iel < mesh.nelem(); ++iel){
                for(std::size_t icol = 0; icol < mesh.conn_el.rowsize(iel); ++icol){
                    mesh.coord_els[iel, icol] = mesh.coord[mesh.conn_el[iel, icol]];
                }
            }
        }

        // form face dof connectivity
        for(const auto& fac_ptr : mesh.faces){
            mesh.face_extended_conn.push_back(FaceGeoDofConnectivity{*fac_ptr, mesh.conn_el});
        }

        // print details for small meshes
        if(mesh.coord.size() < 100){
            mesh.printNodes(std::cout);
            mesh.printElements(std::cout);
            mesh.printFaces(std::cout);
        }

        return mesh;
    }
}
