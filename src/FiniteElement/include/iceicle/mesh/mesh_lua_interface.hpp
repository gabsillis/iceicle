#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/mesh/gmsh_utils.hpp"
#include "iceicle/string_utils.hpp"
#include <fstream>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/lua_utils.hpp>
#include <optional>
#include <sol/sol.hpp>

namespace iceicle {

    /**
     * @brief construct a uniform mesh from inputs provided in a lua table
     */
    template<class T, class IDX, int ndim>
    [[nodiscard]]
    auto lua_uniform_mesh(sol::table& mesh_table)
    -> std::optional<AbstractMesh<T, IDX, ndim>> {
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
        // boundary conditions
        Tensor<BOUNDARY_CONDITIONS, 2 * ndim> bctypes;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            std::string bcname = mesh_table["boundary_conditions"]["types"][iside + 1];
            bctypes[iside] = get_bc_from_name(bcname);
        }

        // boundary condition flags
        Tensor<int, 2 * ndim> bcflags;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            bcflags[iside] = mesh_table["boundary_conditions"]["flags"][iside + 1];
        }

        // optional geometry order input
        int geometry_order = mesh_table.get_or("geometry_order", 1);

        // bounding box
        Tensor<T, ndim> xmin;
        Tensor<T, ndim> xmax;
        sol::optional<sol::table> bounding_box_table = mesh_table["bounding_box"];
        if(bounding_box_table){
            // read nelem in each direction
            Tensor<IDX, ndim> nelem;
            sol::table nelem_table = mesh_table["nelem"];
            for(int idim = 0; idim < ndim; ++idim ){
                nelem[idim] = nelem_table[idim + 1]; // NOTE: Lua is 1-indexed
            }
            for(int idim = 0; idim < ndim; ++idim){
                xmin[idim] = bounding_box_table.value()["min"][idim + 1];
                xmax[idim] = bounding_box_table.value()["max"][idim + 1];
            }
            return std::optional{
                AbstractMesh<T, IDX, ndim>{xmin, xmax, nelem, geometry_order, bctypes, bcflags}};
        } else {
            // NOTE: nelem will be inferred
            sol::optional<sol::table> nodes_1d_table_opt = mesh_table["directional_nodes"];
            if(nodes_1d_table_opt) {
                sol::table nodes_1d_table = nodes_1d_table_opt.value();
                if(nodes_1d_table.size() != ndim){
                    util::AnomalyLog::log_anomaly("directional_nodes table does not have ndim tables of nodes");
                    return std::nullopt;
                }
                std::array<IDX, ndim> nelem;
                std::array<std::vector<T>, ndim> nodes_1d;
                for(int idim = 0; idim < ndim; ++idim) {
                    sol::optional<sol::table> nodes_dir_opt = nodes_1d_table[idim + 1];
                    if(!nodes_dir_opt.has_value()) {
                        util::AnomalyLog::log_anomaly("dimension " + std::to_string(idim + 1) + "nodes table not found" );
                        return std::nullopt;
                    }
                    sol::table nodes_dir = nodes_1d_table[idim + 1];
                    nelem[idim] = nodes_dir.size() - 1;
                    nodes_1d[idim].reserve(nelem[idim] + 1);
                    for(int inode = 0; inode < nelem[idim] + 1; ++inode){
                        nodes_1d[idim].push_back(nodes_dir[inode + 1]);
                    }
                }
                return std::optional{AbstractMesh<T, IDX, ndim>{nodes_1d, geometry_order, bctypes, bcflags}};
            } else {
                return std::nullopt;
            }
        }
    }

    /**
     * @brief construct a uniform mesh from inputs provided in a lua state 
     */
    template<class T, class IDX, int ndim>
    std::optional<AbstractMesh<T, IDX, ndim>> lua_uniform_mesh(sol::state &lua_state){
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
        sol::table mesh_table = lua_state["uniform_mesh"];
        return lua_uniform_mesh<T, IDX, ndim>(mesh_table);
    }

    /// @brief read in a manually created user defined mesh 
    /// this is meant for small test meshes 
    template<class T, class IDX, int ndim>
    [[nodiscard]] inline 
    auto lua_read_user_mesh(sol::table& mesh_table)
    -> std::optional<AbstractMesh<T, IDX, ndim>>
    {
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        // read the node coordinates
        NodeArray<T, ndim> coord;
        sol::table coord_table = mesh_table["coord"];

        for(IDX inode = 0; inode < coord_table.size(); ++inode){
            Point pt{};
            for(int idim = 0; idim < ndim; ++idim)
                pt[idim] = coord_table[inode + 1][idim + 1];
            coord.push_back(pt);
        }

        // read the elements
        sol::table el_table = mesh_table["elements"];
        std::vector<ElementTransformation<T, IDX, ndim>* > el_transforms{};
        std::vector< std::vector<IDX> > conn_el_ragged{};
        for(IDX ielem = 0; ielem < el_table.size(); ++ielem){
            std::string domain_name = el_table[ielem + 1]["domain_type"];
            DOMAIN_TYPE domain_type = parse_domain_type(domain_name);
            int order = el_table[ielem + 1]["order"];
            el_transforms.push_back(
                transformation_table<T, IDX, ndim>.get_transform(domain_type, order));
            std::vector<IDX> nodes;
            sol::table nodes_table = el_table[ielem + 1]["nodes"];
            for(IDX inode = 1; inode <= nodes_table.size(); ++inode){
                IDX node_idx = nodes_table[inode];
                nodes.push_back(node_idx);
            }
            conn_el_ragged.push_back(nodes);
        }
        util::crs<IDX, IDX> conn_el{conn_el_ragged};

        // read the boundary faces
        sol::table bound_table = mesh_table["boundary_faces"];
        std::vector<typename AbstractMesh<T, IDX, ndim>::boundary_face_desc> bdy_fac_infos;
        for(IDX ifac = 0; ifac < bound_table.size(); ++ifac){
            std::string bcname = bound_table[ifac + 1]["bc_type"];
            BOUNDARY_CONDITIONS bc_type = get_bc_from_name(bcname);
            int bcflag = bound_table[ifac + 1]["bc_flag"];
            std::vector<IDX> nodes;
            sol::table nodes_table = bound_table[ifac + 1]["nodes"];
            for(IDX inode = 1; inode <= nodes_table.size(); ++inode){
                IDX node_idx = nodes_table[inode];
                nodes.push_back(node_idx);
            }
            bdy_fac_infos.push_back(
                std::tuple{bc_type, bcflag, nodes});
        }

        return std::optional{AbstractMesh<T, IDX, ndim>(coord, conn_el, el_transforms, bdy_fac_infos)};
    }

    template<class T, class IDX, int ndim>
    [[nodiscard]] inline
    auto lua_read_mixed_mesh(sol::table& mesh_table)
    -> std::optional<AbstractMesh<T, IDX, ndim>>
    {
        using namespace iceicle::util;
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
        if constexpr (ndim == 2) {
            // read nelem in each direction
            Tensor<IDX, ndim> nelem;
            sol::table nelem_table = mesh_table["nelem"];
            for(int idim = 0; idim < ndim; ++idim ){
                nelem[idim] = nelem_table[idim + 1]; // NOTE: Lua is 1-indexed
            }

            // bounding box
            Tensor<T, ndim> xmin;
            Tensor<T, ndim> xmax;
            sol::table bounding_box_table = mesh_table["bounding_box"];
            for(int idim = 0; idim < ndim; ++idim){
                xmin[idim] = bounding_box_table["min"][idim + 1];
                xmax[idim] = bounding_box_table["max"][idim + 1];
            }

            // read the quad element ratio
            Tensor<T, ndim> quad_ratio;
            std::ranges::fill(quad_ratio, 0.0);
            sol::optional<sol::table> quad_ratio_opt = mesh_table["quad_ratio"];
            if(quad_ratio_opt){
                sol::table quad_ratio_arr = quad_ratio_opt.value();
                for(int idim = 0; idim < ndim; ++idim){
                    quad_ratio[idim] = quad_ratio_arr[idim + 1];
                }
            }

            // boundary conditions
            Tensor<BOUNDARY_CONDITIONS, 2 * ndim> bctypes;
            for(int iside = 0; iside < 2 * ndim; ++iside){
                std::string bcname = mesh_table["boundary_conditions"]["types"][iside + 1];
                bctypes[iside] = get_bc_from_name(bcname);
            }

            // boundary condition flags
            Tensor<int, 2 * ndim> bcflags;
            for(int iside = 0; iside < 2 * ndim; ++iside){
                bcflags[iside] = mesh_table["boundary_conditions"]["flags"][iside + 1];
            }

            return mixed_uniform_mesh<T, IDX>(nelem, xmin, xmax, quad_ratio, bctypes, bcflags);

        } else {
            return std::nullopt;
        }
    }

    /// @brief set up a mesh from a gmsh file 
    /// @param mesh_table the lua table to describe the gmsh file and mesh setup
    /// @return the mesh read from the input file
    template<class T, class IDX, int ndim>
    [[nodiscard]] inline 
    auto lua_read_gmsh(sol::table& mesh_table) -> std::optional<AbstractMesh<T, IDX, ndim>> {
        using namespace iceicle::util;
        
        // get the filename to read and open the file 
        sol::optional<std::string> filename = mesh_table["file"];
        if(filename) {
            std::ifstream infile{filename.value()};

            // get the boundary definitions
            std::map<int, std::tuple<BOUNDARY_CONDITIONS, int>> bcmap;
            sol::optional<sol::table> bctable_opt = mesh_table["bc_definitions"];
            if(bctable_opt){
                sol::table bctable = bctable_opt.value();
                for(int idef = 1; idef <= bctable.size(); ++idef){ // 1 indexed for lua
                    sol::table def = bctable[idef];
                    std::string bcname = def[1];
                    sol::optional<int> bcflag = def[2];
                    bcmap[idef] = std::tuple{get_bc_from_name(bcname), bcflag.value_or(0)};
                }
            }

            return read_gmsh<T, IDX, ndim>(infile, bcmap);
        } else {
            AnomalyLog::log_anomaly(Anomaly{"In the gmsh table \"file\" must be specified", general_anomaly_tag{}});
            return std::nullopt;
        }
    }

    template<class T, class IDX, int ndim>
    std::optional<AbstractMesh<T, IDX, ndim>> construct_mesh_from_config(sol::table config){
        using namespace util;

        // try gmsh
        sol::optional<sol::table> gmesh_table = config["gmsh"];
        if(gmesh_table){
            return lua_read_gmsh<T, IDX, ndim>(gmesh_table.value());
        } 

        // try uniform mesh
        sol::optional<sol::table> uniform_mesh_table = config["uniform_mesh"];
        if(uniform_mesh_table){
            return lua_uniform_mesh<T, IDX, ndim>(uniform_mesh_table.value());
        } 

        // try mixed-uniform
        sol::optional<sol::table> mixed_mesh_table = config["mixed_uniform_mesh"];
        if(mixed_mesh_table){
            return lua_read_mixed_mesh<T, IDX, ndim>(mixed_mesh_table.value());
        }

        // try user defined mesh
        sol::optional<sol::table> user_mesh_table = config["user_mesh"];
        if(user_mesh_table){
            return lua_read_user_mesh<T, IDX, ndim>(user_mesh_table.value());
        }

        // no valid mesh config found
        AnomalyLog::log_anomaly(Anomaly{"No recognized mesh configuration found", general_anomaly_tag{}});
        return std::nullopt;
    }

    /// @brief perturb the nodes of the mesh if applicable
    template<class T, class IDX, int ndim>
    auto perturb_mesh(sol::table& config, AbstractMesh<T, IDX, ndim>& mesh) -> void {
        using namespace util;
        sol::optional<std::string> perturb_fcn_name = config["mesh_perturbation"];
        if(perturb_fcn_name){
            std::vector<bool> fixed_nodes = flag_boundary_nodes(mesh);

            std::function< void(std::span<T, ndim>, std::span<T, ndim>) > perturb_fcn;
            if(eq_icase(perturb_fcn_name.value(), "taylor-green")){
                auto bounding_box = compute_bounding_box(mesh);
                perturb_fcn = PERTURBATION_FUNCTIONS::TaylorGreenVortex<T, ndim>{
                    .v0 = 0.5,
                    .xmin = bounding_box.xmin,
                    .xmax = bounding_box.xmax,
                    .L = 1
                };
            } else if(eq_icase(perturb_fcn_name.value(), "zig-zag")){
                if constexpr (ndim >= 2){
                    perturb_fcn = PERTURBATION_FUNCTIONS::ZigZag<T, ndim>{};
                } else {
                    AnomalyLog::log_anomaly(Anomaly{"zig-zag requires 2D or higher", general_anomaly_tag{}});
                }
            }

            perturb_nodes(mesh, perturb_fcn, fixed_nodes);
        }
    }

    template<class T, class IDX, int ndim>
    auto manual_mesh_management(sol::table config, AbstractMesh<T, IDX, ndim>&mesh) -> void {
        using namespace util;

        sol::optional<sol::table> mesh_management = config["mesh_management"];

        if constexpr (ndim == 2){
       
            if(mesh_management){ 
                sol::optional<sol::table> edge_flip_list = mesh_management.value()["edge_flips"];
                
                if(edge_flip_list) for(const auto& kv_pair : edge_flip_list.value()){
                    IDX ifac = kv_pair.second.as<IDX>();
                    edge_swap(mesh, ifac);
                }
            }
        }
        std::vector<IDX> invalid_faces;
        if (!validate_normals(mesh, invalid_faces)) {
            std::string msg = "invalid normals on the following faces: ";
            for(IDX ifac : invalid_faces){
                msg = msg + std::to_string(ifac) + ", ";
            }
            AnomalyLog::log_anomaly(msg);
        }
    }
}
