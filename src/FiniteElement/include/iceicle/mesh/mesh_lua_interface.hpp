#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/geometry/face.hpp"
#include <iceicle/mesh/mesh.hpp>
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
        // TODO: figure out why optional isn't working
//        std::optional<int> geo_order_input = lua_state["uniform_mesh"]["geometry_order"];
//        if(geo_order_input){
//            geometry_order = geo_order_input.value();
//        }

        return AbstractMesh<T, IDX, ndim>{xmin, xmax, nelem, geometry_order, bctypes, bcflags};
    }
}
