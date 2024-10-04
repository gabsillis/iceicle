/// @brief utility for storing the evauation of basis functions
/// and solutions
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include "mdspan/mdspan.hpp"
#include <iceicle/element/finite_element.hpp>

namespace iceicle {

    /// @brief an evaluation of basis functions and optionally the derivatives
    ///        Takes care of data storage concerns
    ///
    /// @tparam real the real number type
    /// @tparam IDX the index type
    /// @tparam ndim the number of dimensions
    /// @tparam use_gradient set to true to enable computationn and storage of gradients
    /// gradients of basis functions are computed wrt reference domain coordinates
    /// @tparam use_hessian set to true to enable computation and storage of hessians
    /// hessians of the basis functions are computed wrt reference domain coordinates
    template<class real, class IDX, int ndim, bool use_gradient = true, bool use_hessian = true>
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
        using bi_span_t = std::mdspan<real, std::extents<int, std::dynamic_extent>>;

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
            const FiniteElement<real, IDX, ndim>& el,
            const Point& reference_domain_pt
        ) : bi(el.nbasis()), 
            grad_bi( (use_gradient) ? el.nbasis() * ndim : 0 ), 
            hess_bi( (use_hessian) ? el.nbasis() * ndim * ndim : 0 ),
            bi_span{bi.data(), el.nbasis()},
            grad_bi_span{grad_bi.data(), el.nbasis()},
            hess_bi_span{hess_bi.data(), el.nbasis()}
        {
            // perform the evaluations
            el.evalBasis(reference_domain_pt, bi.data());
            if constexpr (use_gradient) {
                el.evalGradBasis(reference_domain_pt, grad_bi.data());
            }

            if constexpr (use_hessian) {
                el.evalHessBasis(reference_domain_pt, hess_bi.data());
            }
        }
    };
}
