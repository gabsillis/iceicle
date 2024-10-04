/// @brief solver for linear forms 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/fe_function/fespan.hpp>
#include <iceicle/element_linear_solve.hpp>

namespace iceicle::solvers {

    /// @brief Solver for u = f 
    /// or the weak form <u, v> = <f, v>
    /// @tparam T the real number type
    /// @tparam IDX the indexing type 
    /// @tparam ndim the number of dimensions
    /// @tparam disc_type the discretization type that must provide 
    ///  domain_integral(const FiniteElement&, NodeArray&, elspan)
    ///  and static constxpr int nv_comp
    template<
        class T, 
        class IDX,
        int ndim,
        class disc_type
    >
    class LinearFormSolver {

        private:
        /// @brief number of equations
        static constexpr int neq = disc_type::nv_comp;

        public:
        /// @brief store a reference to the fespace being used 
        FESpace<T, IDX, ndim> &fespace;

        /// @brief store a reference to the discretization being solved
        disc_type &disc;

        /// @brief Constructor 
        /// @param fespace the Finite Element Space
        /// @param disc the discretization 
        LinearFormSolver(
            FESpace<T, IDX, ndim>& fespace,
            disc_type& disc
        ) : fespace{fespace}, disc{disc} {}

        /// @brief solve <u, v> = <f, v>
        /// @param u the fespan to represent the coefficients of u
        template<class LayoutPolicy, class AccessorPolicy>
        auto solve(
            fespan<T, LayoutPolicy, AccessorPolicy> u
        ) -> void {
            // TODO: generalize to CG and remove reliance on ElementLinearSolve
            std::vector<T> u_local_data(fespace.dg_map.max_el_size_reqirement(neq));
            std::vector<T> res_local_data(fespace.dg_map.max_el_size_reqirement(neq));
            std::for_each(fespace.elements.begin(), fespace.elements.end(), 
                [&](const FiniteElement<T, IDX, ndim> &el){
                    // form the element local views
                    // TODO: maybe instead of scatter from local view 
                    // we can directly create the view on the subset of u 
                    // for CG this might require a different compact Layout 
                    dofspan u_local{u_local_data.data(), u.create_element_layout(el.elidx)};
                    u_local = 0;

                    dofspan res_local{res_local_data.data(), u.create_element_layout(el.elidx)};
                    res_local = 0;

                    // project
                    disc.domain_integral(el, res_local);

                    // solve 
                    ElementLinearSolver<T, IDX, ndim, neq> solver{el};
                    solver.solve(u_local, res_local);

                    // scatter to global array 
                    // (note we use 0 as multiplier for current values in global array)
                    scatter_elspan(el.elidx, 1.0, u_local, 0.0, u);
                }
            );
    
        }
    };

    // deduction guide
    template<
        class T, 
        class IDX,
        int ndim,
        class disc_type
    > LinearFormSolver(
        FESpace<T, IDX, ndim>& fespace,
        disc_type& disc
    ) -> LinearFormSolver<T, IDX, ndim, disc_type>;
}
