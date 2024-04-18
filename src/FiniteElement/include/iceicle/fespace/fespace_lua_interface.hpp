#pragma once
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/lua_utils.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <sol/sol.hpp>
namespace FE {

    template<class T, class IDX, int ndim>
    FESpace<T, IDX, ndim> lua_fespace(
        MESH::AbstractMesh<T, IDX, ndim> *meshptr,
        sol::table &tbl
    ) {

        // get the basis function type
        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype = FESPACE_ENUMS::LAGRANGE; // default
        sol::optional<std::string> basis_name = tbl["basis"];
        if(basis_name){
            if(ICEICLE::UTIL::eq_icase(basis_name.value(), "lagrange")){
                btype = FESPACE_ENUMS::LAGRANGE;
            }
        }

        // get the quadrature type
        FESPACE_ENUMS::FESPACE_QUADRATURE qtype = FESPACE_ENUMS::GAUSS_LEGENDRE;
        sol::optional<std::string> quadrature_name = tbl["quadrature"];
        if(quadrature_name){
            if(ICEICLE::UTIL::eq_icase(quadrature_name.value(), "gauss")){
                qtype = FESPACE_ENUMS::GAUSS_LEGENDRE;
            }
        }

        // get the basis polynomial order
        sol::optional<int> order = tbl["order"];
        auto seq = NUMTOOL::TMP::make_range_sequence<int, 0, BUILD_CONFIG::FESPACE_BUILD_PN>{};
        FESpace<T, IDX, ndim> fespace;
        return NUMTOOL::TMP::invoke_at_index(seq, order.value(),
            [&]<int order_comp>{
                return FESpace<T, IDX, ndim>{meshptr, btype, qtype, std::integral_constant<int, order_comp>{}};
            }
        );
    }

    template<class T, class IDX, int ndim>
    FESpace<T, IDX, ndim> lua_fespace(
        MESH::AbstractMesh<T, IDX, ndim> *meshptr,
        sol::state &lua
    ) {
        sol::table tbl = lua["fespace"];

        return lua_fespace(meshptr, tbl);
    }
}
