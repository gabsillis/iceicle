/**
 * @brief procedures for forming residuals from arbitrary discretizations
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once 
#include "iceicle/fespace/fespace.hpp"
#include <span>
#include <exception>
#include <stdexcept>

namespace ICEICLE::SOLVERS {

    /**
     * @brief requires that the discretization specifies the number of 
     * vector components the solutions have 
     */
    template< class disc_class >
    concept specifies_ncomp = std::is_same_v<decltype(disc_class::nv_comp), int>;



    /**
     * @brief form the residual based on the fespace and discretization 
     * The residual is the function over the vector components for each degree of freedom
     * where getting a residual of 0 solves the discretized portion of the equation 
     * (i.e. for semi-discrete forms M du/dt = residual)
     *
     * @tparam T the floaating point type
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions 
     * @tparam  disc_class the discretization type 
     */
    template<
        class T, 
        class IDX,
        int ndim,
        template<class, class, int> class disc_class
    >
    void form_residual(
        FE::FESpace<T, IDX, ndim> &fespace,
        disc_class<T, IDX, ndim> disc,
        std::span<T> res_data
    )
    requires specifies_ncomp<disc_class<T, IDX, ndim>>
    {

        // TODO: machinery for Continuous Galerkin case 
        if(res_data.size() < fespace.dg_offsets.calculate_size_requirement(disc_class<T, IDX, ndim>::nv_comp)){
            throw std::out_of_range("The residual span given is not large enough to accomadate the DG residual");
        }



    }
}
