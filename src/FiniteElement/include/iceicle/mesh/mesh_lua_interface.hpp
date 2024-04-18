#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/anomaly_log.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/string_utils.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/lua_utils.hpp>
#include <optional>
#include <sol/sol.hpp>

namespace MESH {

    /**
     * @brief construct a uniform mesh from inputs provided in a lua state 
     */
    template<class T, class IDX, int ndim>
    AbstractMesh<T, IDX, ndim> lua_uniform_mesh(sol::state &lua_state){
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
        sol::table mesh_table = lua_state["uniform_mesh"];

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

        // boundary conditions
        Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            std::string bcname = mesh_table["boundary_conditions"]["types"][iside + 1];
            bctypes[iside] = ELEMENT::get_bc_from_name(bcname);
        }

        // boundary condition flags
        Tensor<int, 2 * ndim> bcflags;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            bcflags[iside] = mesh_table["boundary_conditions"]["flags"][iside + 1];
        }

        // optional geometry order input
        int geometry_order = 1;
        sol::optional<int> geo_order_input = lua_state["uniform_mesh"]["geometry_order"];
        if(geo_order_input){
            geometry_order = geo_order_input.value();
        }

        return AbstractMesh<T, IDX, ndim>{xmin, xmax, nelem, geometry_order, bctypes, bcflags};
    }


    /**
     * @brief construct a uniform mesh from inputs provided in a lua table
     */
    template<class T, class IDX, int ndim>
    [[nodiscard]] AbstractMesh<T, IDX, ndim> lua_uniform_mesh(sol::table& mesh_table){
        using namespace NUMTOOL::TENSOR::FIXED_SIZE;
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

        // boundary conditions
        Tensor<ELEMENT::BOUNDARY_CONDITIONS, 2 * ndim> bctypes;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            std::string bcname = mesh_table["boundary_conditions"]["types"][iside + 1];
            bctypes[iside] = ELEMENT::get_bc_from_name(bcname);
        }

        // boundary condition flags
        Tensor<int, 2 * ndim> bcflags;
        for(int iside = 0; iside < 2 * ndim; ++iside){
            bcflags[iside] = mesh_table["boundary_conditions"]["flags"][iside + 1];
        }

        // optional geometry order input
        int geometry_order = 1;
        sol::optional<int> geo_order_input = mesh_table["geometry_order"];
        if(geo_order_input){
            geometry_order = geo_order_input.value();
        }

        return AbstractMesh<T, IDX, ndim>{xmin, xmax, nelem, geometry_order, bctypes, bcflags};
    }
    template<class T, class IDX, int ndim>
    std::optional<AbstractMesh<T, IDX, ndim>> construct_mesh_from_config(sol::table& config){
        using namespace ICEICLE::UTIL;
        if(config["uniform_mesh"]){
            return lua_uniform_mesh<T, IDX, ndim>(config["uniform_mesh"]);
        } else {
            AnomalyLog::log_anomaly(Anomaly{"No recognized mesh configuration found", general_anomaly_tag{}});
            return std::nullopt;
        }
    }

    /// @brief perturb the nodes of the mesh if applicable
    template<class T, class IDX, int ndim>
    auto perturb_mesh(sol::table& config, AbstractMesh<T, IDX, ndim>& mesh) -> void {
        using namespace ICEICLE::UTIL;
        sol::optional<std::string> perturb_fcn_name = config["mesh_perturbation"];
        if(perturb_fcn_name){
            std::vector<bool> fixed_nodes = MESH::flag_boundary_nodes(mesh);

            std::function< void(std::span<T, ndim>, std::span<T, ndim>) > perturb_fcn;
            if(eq_icase(perturb_fcn_name.value(), "taylor-green")){
                auto bounding_box = compute_bounding_box(mesh);
                perturb_fcn = MESH::PERTURBATION_FUNCTIONS::TaylorGreenVortex<T, ndim>{
                    .v0 = 0.5,
                    .xmin = bounding_box.xmin,
                    .xmax = bounding_box.xmax,
                    .L = 1
                };
            } else if(eq_icase(perturb_fcn_name.value(), "zig-zag")){
                perturb_fcn = MESH::PERTURBATION_FUNCTIONS::ZigZag<T, ndim>{};
            }

            MESH::perturb_nodes(mesh, perturb_fcn, fixed_nodes);
        }
    }
}
