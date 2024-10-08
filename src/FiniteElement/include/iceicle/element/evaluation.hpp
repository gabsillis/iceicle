/// @brief utility for storing the evauation of basis functions
/// and solutions
/// @author Gianni Absillis (gabsill@ncsu.edu)
#pragma once
#include "iceicle/quadrature/QuadratureRule.hpp"
#include "mdspan/mdspan.hpp"
#include <iceicle/basis/basis.hpp>

namespace iceicle {

    /// @brief an evaluation of basis functions and optionally the derivatives
    ///         Derivatives are wrt the reference domain coordinates
    ///         Takes care of data storage concerns
    ///
    /// @tparam real the real number type
    /// @tparam IDX the index type
    /// @tparam ndim the number of dimensions
    template<class real, int ndim>
    class BasisEvaluation {
        private:

        /// The 1D storage for basis functions
        std::vector<real> bi;

        /// The 1D storage for gradient of basis functions
        std::vector<real> grad_bi;

        /// The 1D storage for hessian of basis functions
        std::vector<real> hess_bi;

        public:

        // ============
        // = Typedefs =
        // ============

        /// === view type definitions ===

        /// view over the basis functions type
        using bi_span_t = std::span<real, std::dynamic_extent>;

        /// view of the gradient type
        using grad_span_t = std::mdspan<real, std::extents<int, std::dynamic_extent, ndim>>;

        /// view over the hessian type
        using hess_span_t = std::mdspan<real, std::extents<int, std::dynamic_extent, ndim, ndim>>;

        // === utility type definitions ===
        using Point = MATH::GEOMETRY::Point<real, ndim>;

        // ==============
        // = data views =
        // ==============

        /// view over the basis functions
        bi_span_t bi_span;

        /// view over the gradient of the basis functions
        grad_span_t grad_bi_span;

        /// view over the hessian of the basis functions 
        hess_span_t hess_bi_span;

        // ===============
        // = Constructor =
        // ===============
        
        // @brief Construct a Basis Evaluation 
        //
        // Evaluates basis functions at the point 
        // if gradient is enabled, evaluates gradient
        // if hessian is enabled, evaluates hessian
        //
        // @param el the finite element to evaluate basis functions on 
        // @param reference_domain_pt the point in the reference domain to evaluate at
        BasisEvaluation(
            const Basis<real, ndim>& basis,
            const Point& reference_domain_pt
        ) : bi(basis.nbasis()), 
            grad_bi( basis.nbasis() * ndim ), 
            hess_bi( basis.nbasis() * ndim * ndim ),
            bi_span{bi.data(), (std::size_t) basis.nbasis()},
            grad_bi_span{grad_bi.data(), basis.nbasis()},
            hess_bi_span{hess_bi.data(), basis.nbasis()}
        {
            // perform the evaluations
            basis.evalBasis(reference_domain_pt, bi.data());
            basis.evalGradBasis(reference_domain_pt, grad_bi.data());
            basis.evalHessBasis(reference_domain_pt, hess_bi.data());
        }
    };

    template<class T, int ndim>
    BasisEvaluation(
        const Basis<T, ndim>&,
        const MATH::GEOMETRY::Point<T, ndim>&
    )-> BasisEvaluation<T, ndim>;

    /// @brief evaluate basis functions and derivatives at all the quadrature points
    /// in a given quadrature rule 
    /// @param basis the basis functions 
    /// @param quadrule the quadrature rule 
    /// @return a vector of evauations for each quadrature point
    template<class T, class IDX, int ndim>
    inline constexpr
    auto quadrature_point_evaluations(
        const Basis<T, ndim>& basis,
        const QuadratureRule<T, IDX, ndim>& quadrule
    ) -> std::vector<BasisEvaluation<T, ndim>>
    {
        std::vector<BasisEvaluation<T, ndim>> evals{}; 
        evals.reserve(quadrule.npoints());
        for(int iqp = 0; iqp < quadrule.npoints(); ++iqp){
            evals.emplace_back(basis, quadrule.getPoint(iqp).abscisse);
        }
        return evals;
    }
}
