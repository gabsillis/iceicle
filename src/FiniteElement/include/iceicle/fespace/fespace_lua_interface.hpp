#pragma once
#include "Numtool/tmp_flow_control.hpp"
#include "iceicle/build_config.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <sol/sol.hpp>
#include <type_traits>
namespace iceicle {

    template<class T, class IDX, int ndim>
    FESpace<T, IDX, ndim> lua_fespace(
        AbstractMesh<T, IDX, ndim> *meshptr,
        sol::table &tbl
    ) {

        // get the basis function type
        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype = FESPACE_ENUMS::LAGRANGE; // default
        sol::optional<std::string> basis_name = tbl["basis"];
        if(basis_name){
            if(util::eq_icase(basis_name.value(), "lagrange")){
                btype = FESPACE_ENUMS::LAGRANGE;
            }
            if(util::eq_icase(basis_name.value(), "legendre")){
                btype = FESPACE_ENUMS::LEGENDRE;
            }
        }

        // get the quadrature type
        FESPACE_ENUMS::FESPACE_QUADRATURE qtype = FESPACE_ENUMS::GAUSS_LEGENDRE;
        sol::optional<std::string> quadrature_name = tbl["quadrature"];
        if(quadrature_name){
            if(util::eq_icase(quadrature_name.value(), "gauss")){
                qtype = FESPACE_ENUMS::GAUSS_LEGENDRE;
            }
        }

        // get the basis polynomial order
        sol::optional<int> order = tbl["order"];
        if(order){
            auto seq = NUMTOOL::TMP::make_range_sequence<int, 0, build_config::FESPACE_BUILD_PN>{};
            return NUMTOOL::TMP::invoke_at_index(seq, order.value(),
                [&]<int order_comp>{
                    return FESpace<T, IDX, ndim>{meshptr, btype, qtype, std::integral_constant<int, order_comp>{}};
                }
            );
        } else {
            return FESpace<T, IDX, ndim>{meshptr, btype, qtype, std::integral_constant<int, 0>{}};
        }
    }

    template<class T, class IDX, int ndim>
    FESpace<T, IDX, ndim> lua_fespace(
        AbstractMesh<T, IDX, ndim> *meshptr,
        sol::state &lua
    ) {
        sol::table tbl = lua["fespace"];

        return lua_fespace(meshptr, tbl);
    }
}
