/**
 * @file finite_element.hpp
 * @brief A finite element which can get quadrature, evaluate basis functions,
 * and perform transformations
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include <Numtool/matrix/dense_matrix.hpp>
#include <Numtool/matrixT.hpp>
#include <iceicle/basis/basis.hpp>
#include <iceicle/element/evaluation.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/quadrature/QuadratureRule.hpp>
#include <iceicle/linalg/linalg_utils.hpp>
#include <span>
#include <vector>

#include <mdspan/mdspan.hpp>

namespace iceicle {

/**
 * @brief Precomputed values of basis functions and other reference domain quantities 
 * at important points (such as quadrature points)
 */
template <typename T, typename IDX, int ndim> 
class FEEvaluation {
private:
  /** @brief the basis functions evaluated at each quadrature point */
  std::vector<std::vector<T>> data;

public:
  /** @brief the basis functions */
  const Basis<T, ndim> *basis;

  /** @brief the quadraature rule */
  const QuadratureRule<T, IDX, ndim> *quadrule;

  FEEvaluation() = default;

  FEEvaluation(Basis<T, ndim> *basisptr,
               QuadratureRule<T, IDX, ndim> *quadruleptr)
      : data{static_cast<std::size_t>(quadruleptr->npoints())}, basis(basisptr),
        quadrule(quadruleptr) {
    // call eval basis for each quadrature point and prestore
    for (int igauss = 0; igauss < quadrule->npoints(); ++igauss) {
      std::vector<T> &eval_vec = data[igauss];
      eval_vec.resize(basis->nbasis());
      basis->evalBasis(quadrule->getPoint(igauss).abscisse, eval_vec.data());
    }

    // TODO: gradient of basis (make a DenseMatrix class)
  }

  FEEvaluation(Basis<T, ndim> &basis,
               QuadratureRule<T, IDX, ndim> &quadrule)
      : FEEvaluation(&basis, &quadrule) {}

  /* @brief get the evaluations of the basis functions at the igaussth
   * quadrature pt */
  const std::vector<T> &operator[](int igauss) const { return data[igauss]; }
};

/**
 * @brief FiniteElement is a collection of components
 * to represent functions on a physical subdomain, that is mapped to a reference subdomain 
 * and perform calculations to query and integrate these functions
 *
 * WARNING: everything here is a pointer or reference type 
 * make sure the data referenced cannot be invalidated
 *
 * Consists of:
 * A ElementTransformation - to represent the transformation from reference to physical space 
 * 
 * Basis function set - finite element functions are a linear combination
 *                      of coefficients and basis functions
 *
 * Quadrature Rule    - for integrating functions on the finite element
 *
 * Evaluation         - precomputed values
 *
 * @tparam T the data type
 * @tparam IDX the index type
 * @tparam ndim the number of dimensions
 */
template <typename T, typename IDX, int ndim>
struct FiniteElement {
  
  using Point = MATH::GEOMETRY::Point<T, ndim>;
  using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
  using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

  /** @brief the geometric transformation */
  const ElementTransformation<T, IDX, ndim> *trans;

  /** @brief the basis functions */
  const Basis<T, ndim> *basis;

  /** @brief the quadraature rule */
  const QuadratureRule<T, IDX, ndim> *quadrule;

  /// @brief the basis evaluation type
  using Eval_t = BasisEvaluation<T, ndim>;

  /** @brief precomputed evaluations of the basis functions 
   * at the quadrature points 
   *
   * NOTE: this is stored by reference because many physical elements will share the same 
   * reference domain, basis functions, and quadrature rule 
   * therefore the evaluations will be the same
   *
   **/
  std::span<const Eval_t> qp_evals;

  /// @brief the node indices for the element
  const std::span<IDX> inodes;

  /// @brief the node coordinates for the element
  const std::span<Point> coord_el;

  /** @brief the element index in the mesh */
  const IDX elidx;

  // =============================
  // = Basis Function Operations =
  // =============================

  /* @brief get the number of basis functions */
  int nbasis() const { return basis->nbasis(); }

  /**
   * @brief evaluate the values of the basis functions at the given point
   * cost: cost of evaluating all basis functions at a point
   * @param [in] xi the point in the reference domain [size = ndim]
   * @param [out] Bi the values of the basis functions at the point [size =
   * nbasis]
   */
  void eval_basis(const T *xi, T *Bi) const { basis->evalBasis(xi, Bi); }

  /**
   * @brief get the basis function evaluated at the quadrature point 
   * @param iqp the quadrature point index 
   * @param ibasis the basis function index
   */
  T basis_qp(int iqp, int ibasis) const noexcept
  { return qp_evals[iqp].bi_span[ibasis]; }

  /**
   * @brief get the values of the basis functions at the given quadrature point
   * @param [in] quadrature_pt_idx the index of the quadrature point in [0,
   * ngauss())
   * @return the values of the basis functions at the quadrature point [size =
   * nbasis]
   */
  auto eval_basis_qp(int quadrature_pt_idx) const noexcept
  -> Eval_t::bi_span_t 
  { return qp_evals[quadrature_pt_idx].bi_span; }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to reference domain coordinates
   *
   * @param [in] xi  the point in the reference domain [size = ndim]
   * @param [out] dBidxj the values of the first derivatives of the basis
   * functions with respect to the reference domain at that point This is in the
   * form of a 1d pointer array that must be preallocated size must be nbasis *
   * ndim or larger
   *
   * @return an mdspan view of dBidxj for an easy interface
   *         \frac{dB_i}{d\xi_j} where i is ibasis
   *         takes a pointer to the first element of this data structure
   *         [size = [nbasis : i][ndim : j]]
   */
  auto eval_grad_basis(const T *xi, T *dBidxj) const {
    basis->evalGradBasis(xi, dBidxj);
    std::extents<int, std::dynamic_extent, ndim> extents(
        nbasis());
    std::mdspan gbasis{dBidxj, extents};
    static_assert(gbasis.extent(1) == ndim);
    return gbasis;
  }

  /**
   * @brief evaluate the first derivatives of the basis functions at a
   * quadrature point
   *
   * @param [in] quadrature_pt_idx the index of the quadrature point [0,
   * ngauss()]
   *
   * @return an mdspan view of dBidxj for an easy interface
   *         \frac{dB_i}{d\xi_j} where i is ibasis
   *         takes a pointer to the first element of this data structure
   *         [size = [nbasis : i][ndim : j]]
   */
  auto eval_grad_basis_qp(int quadrature_pt_idx) const noexcept
  -> Eval_t::grad_span_t 
  { return qp_evals[quadrature_pt_idx].grad_bi_span; }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to physical domain coordinates
   * @param [in] xi the point in the reference domain (uses Point class)
   * @param [in] J the jacobian of the element transformation 
   *               (optional: see overload)
   * @param [in] grad_bi view over the gradients of the basis functions 
   *               (optional: see overload)
   * @param [out] dBidxj the values of the first derivatives of the basis
   * functions with respect to the reference domain at that point This is in the
   * form of a 1d pointer array that must be preallocated size must be nbasis *
   * ndim or larger
   *
   * @return an mdspan view of dBidxj for an easy interface
   *         \frac{dB_i}{dx_j} where i is ibasis
   *         takes a pointer to the first element of this data structure
   *         [size = [nbasis : i][ndim : j]]
   */
  auto eval_phys_grad_basis(
    const Point &xi,
    const JacobianType &J,
    linalg::in_tensor auto grad_bi, 
    T *dBidxj
  ) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // pull nbasis to variable so its more likely to be put in register
    int ndof = nbasis();

    //  fill with zero
    std::fill_n(dBidxj, ndof * ndim, 0.0);

    // the inverse of J = adj(J) / det(J)
    auto adjJ = adjugate(J);
    auto detJ = determinant(J);
    detJ = (detJ == 0.0) ? 1.0 : detJ; // protect from div by zero

    std::extents<int, std::dynamic_extent, ndim> extents(
        ndof);
    std::mdspan gbasis{dBidxj, extents};
    // dBidxj =  Jadj_{jk} * dBidxk
    for (int i = 0; i < ndof; ++i) {
      for (int j = 0; j < ndim; ++j) {
        // gbasis[i, j] = 0.0;
        for (int k = 0; k < ndim; ++k) {
          gbasis[i, j] += grad_bi[i, k] * adjJ[k][j];
        }
      }
    }

    // multiply though by the determinant
    for (int i = 0; i < ndof * ndim; ++i) {
      dBidxj[i] /= detJ;
    }

    return gbasis;
  }

  /** Calculates the gradient wrt reference domain: \overload */
  auto eval_phys_grad_basis(
    const Point &xi,
    const JacobianType& J,
    T *dbidxj 
  ) const {
    // compute the basis functions in reference domain
    std::vector<T> dBi_data(ndim * nbasis());
    auto grad_bi = eval_grad_basis(xi, dBi_data.data());

    return eval_phys_grad_basis(xi, J, grad_bi, dbidxj);
  }

  /** Calculates the Jacobian and gradient wrt reference domain: \overload */
  auto eval_phys_grad_basis(
    const Point &xi,
    T *dbidxj 
  ) const {
    // compute the basis functions in reference domain
    std::vector<T> dBi_data(ndim * nbasis());
    auto grad_bi = eval_grad_basis(xi, dBi_data.data());

    // compute jacobian
    JacobianType J = trans->jacobian(coord_el, xi);

    return eval_phys_grad_basis(xi, J, grad_bi, dbidxj);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   * wrt the reference domain
   *
   * @param [in] xi the point in the reference domain to evaluate at
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial \xi_j \partial \xi_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_hess_basis(const Point &xi, T *basis_hessian_data) const {
    using namespace std::experimental;
    basis->evalHessBasis(xi, basis_hessian_data);
    extents<int, std::dynamic_extent, ndim, ndim> exts{nbasis()};
    mdspan hess_basis{basis_hessian_data, exts};
    return hess_basis;
  }

  /**
   * @brief get the hessian of the basis functions at a  quadrature point
   * wrt the referene domain 
   *
   * @param quadrature_pt_idx the quadrature point index 
   * @return a view over the hessian of the basis functions 
   */
  auto eval_hess_basis_qp(int quadrature_pt_idx) const noexcept 
  -> Eval_t::hess_span_t
  { return qp_evals[quadrature_pt_idx].hess_bi_span; }

  /**
   * @brief evaluate the second derivatives of the basis functions 
   * wrt the physical domain coordinates
   *
   * @param [in] xi the point in the reference domain to evaluate at
   * @param [in] coord the global node coordinates array
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial x_j \partial x_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_phys_hess_basis(
    const JacobianType& trans_jac,
    const HessianType& trans_hess,
    linalg::in_tensor auto phys_grad_basis,
    linalg::in_tensor auto ref_hess_basis,
    T *basis_hessian_data
  ) const {

    // pull nbasis to variable so its more likely to be put in register
    int ndof = nbasis();

    // shape of hessian multidimensional arrays
    std::extents<int, std::dynamic_extent, ndim, ndim> hess_exts{ndof};

    // rhs of hessian equation
    std::vector<T> rhs_data(ndim * ndim * ndof, 0.0);
    std::mdspan rhs{rhs_data.data(), hess_exts};

    // multidimensional array view of hessian
    std::mdspan hess_phys{basis_hessian_data, hess_exts};

    // fill with zeros
    std::fill_n(basis_hessian_data, ndim * ndim * ndof, 0.0);

    // get the adjugate and determinant of the transformation jacobian
    // the inverse of J = adj(J) / det(J)
    auto adjJ = adjugate(trans_jac);
    T detJ = determinant(trans_jac);
    detJ = (detJ == 0.0) ? 1.0 : detJ; // protect from div by zero
    T detJ2 = SQUARED(detJ);

    // loop variables (idof = degree of freedom index), (*d = dimension index)
    int idof, id, jd, kd, ld;

    // form the rhs of the physical hessian equation
    // copy over the reference coordinate hessian to rhs
    // adjust for gradient
    for (idof = 0; idof < ndof; ++idof) {
      for (id = 0; id < ndim; ++id) {
        for (jd = 0; jd < ndim; ++jd) {
          rhs[idof, id, jd] = ref_hess_basis[idof, id, jd];
          for (kd = 0; kd < ndim; ++kd) {
            rhs[idof, id, jd] -=
              trans_hess[kd][id][jd] * phys_grad_basis[idof, kd];
          }
        }
      }
    }

    // compute the hessians
    for (idof = 0; idof < ndof; ++idof) {
      // J transpose inverse times H twice
      for (id = 0; id < ndim; ++id) {
        for (jd = 0; jd < ndim; ++jd) {
          for (kd = 0; kd < ndim; ++kd) {
            for (ld = 0; ld < ndim; ++ld) {
              hess_phys[idof, id, jd] +=
                  adjJ[kd][jd] * adjJ[ld][id] * rhs[idof, ld, kd];
            }
          }
        }
      }
    }

    for (int i = 0; i < ndim * ndim * ndof; ++i) {
      basis_hessian_data[i] /= detJ2;
    }

    return hess_phys;
  }

  /// @brief \overload recomputes jacobian, hessian, physical gradient, and reference hessian
  auto eval_phys_hess_basis(
      const Point& refpt,
      T *hessian_data
  ) const {
    JacobianType jac = jacobian(refpt);
    HessianType hess = hessian(refpt);
    std::vector<T> grad_data(nbasis() * ndim);
    std::vector<T> hess_data(nbasis() * ndim * ndim);
    auto phys_grad_basis = eval_phys_grad_basis(refpt, jac, grad_data.data());
    auto ref_hess_basis = eval_hess_basis(refpt, hess_data.data());
    return eval_phys_hess_basis(jac, hess, phys_grad_basis, ref_hess_basis, hessian_data);
  }

  // =========================
  // = Quadrature Operations =
  // =========================

  /** @brief get the number of quadrature points in the quadrature rule */
  int nQP() const { return quadrule->npoints(); }

  /** @brief get the "QuadraturePoint" (contains point and weight) at the given
   * quadrature point index */
  const QuadraturePoint<T, ndim> getQP(int qp_idx) const {
    return (*quadrule)[qp_idx];
  }

  // ========================
  // = Geometric Operations =
  // ========================

  /**
   * @brief transform from the reference domain to the physical domain
   * @param [in] pt_ref the point in the refernce domain
   * @return pt_phys the point in the physical domain
   */
  inline Point transform( const Point &pt_ref ) const {
    return trans->transform(coord_el, pt_ref);
  }

  /// @brief calculate the Jacobian of the transformation at a given point 
  /// @param pt_ref the point in the reference domain 
  /// @return the jacobian of the transformation
  inline constexpr 
  auto jacobian( const Point& pt_ref ) const 
  -> JacobianType {
    return trans->jacobian(coord_el, pt_ref);
  }

  /// @brief calculate the Hessian of the transformation at a given point 
  /// @param pt_ref the point in the reference domain 
  /// @return the hessian of the transformation
  inline constexpr 
  auto hessian( const Point& pt_ref ) const 
  -> HessianType {
    return trans->hessian(coord_el, pt_ref);
  }

  inline constexpr
  auto centroid() const -> Point
  {
    return trans->centroid(coord_el);
  }
};

/**
 * @brief storage for use in PhysDomainEval 
 *
 * prevents re-creating the same std::vector's repeatedly
 */
template<class T, int ndim>
class PhysDomainEvalStorage {
  public:
  std::vector<T> gradient_storage;
  std::vector<T> hessian_storage;

  template<class IDX>
  PhysDomainEvalStorage(const FiniteElement<T, IDX, ndim>& el)
  : gradient_storage(el.nbasis() * ndim),
    hessian_storage(el.nbasis() * ndim * ndim)
  {}
};

template<class T, class IDX, int ndim>
PhysDomainEvalStorage(const FiniteElement<T, IDX, ndim>&) -> PhysDomainEvalStorage<T, ndim>;

/**
 * @brief given a finite element and a point in the reference domain 
 * This utility computes and stores physical domain dependent information
 * to reduce code duplication
 * 
 * - Transformation Jacobian
 * - Transformation Hessian
 *
 * - Basis function gradients wrt physical domain coordinates 
 * - Basis function hessians wrt physical domain coordinates 
 */
template<class T, int ndim>
struct PhysDomainEval {
  using Point = MATH::GEOMETRY::Point<T, ndim>;
  using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
  using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

  /// view of the gradient type
  using grad_span_t = std::mdspan<T, std::extents<int, std::dynamic_extent, ndim>>;

  /// view over the hessian type
  using hess_span_t = std::mdspan<T, std::extents<int, std::dynamic_extent, ndim, ndim>>;

  PhysDomainEvalStorage<T, ndim>& storage;

  /// @brief the jacobian of the element transformation 
  JacobianType jac;

  /// @brief the hessian of the element transformation
  HessianType hess;

  /// @brief the gradient of the basis functions wrt physical domain coordinates
  grad_span_t phys_grad_basis;

  /// @brief the hessian of the basis functions wrt physical domain coordinates 
  hess_span_t phys_hess_basis;

  /** 
   * @brief construct data arrays and compute the evaluation 
   *
   * @param el the FiniteElement to compute for 
   * @param pt the point in the reference domain 
   * @param ref_evals the evaluations of basis functions and derivatives
   * wrt the reference domain 
   * NOTE: this must be computed with pt to be correct
   */
  template<class IDX>
  PhysDomainEval(
      PhysDomainEvalStorage<T, ndim>& storage,
      const FiniteElement<T, IDX, ndim>& el, 
      const Point& pt,
      const BasisEvaluation<T, ndim>& ref_evals
  ) : storage{storage},
      jac{el.jacobian(pt)}, hess{el.hessian(pt)}, 
      phys_grad_basis{storage.gradient_storage.data(), el.nbasis()},
      phys_hess_basis{storage.hessian_storage.data(), el.nbasis()}
  {
    el.eval_phys_grad_basis(pt, jac, ref_evals.grad_bi_span,
        storage.gradient_storage.data());
    el.eval_phys_hess_basis(jac, hess, phys_grad_basis,
        ref_evals.hess_bi_span, storage.hessian_storage.data());
  }

  /**
   * @brief overload to construct data arrays and compute evaluation 
   * performs the reference domain basis functions on the fly 
   * this may be redundant -- prefer manually computing these to using this constructor
   *
   * @param el the FiniteElement to compute for 
   * @param pt the point in the reference domain 
   */
  template<class IDX>
  PhysDomainEval(
      PhysDomainEvalStorage<T, ndim>& storage,
      const FiniteElement<T, IDX, ndim>& el,
      const Point& pt
  ) : PhysDomainEval{storage, el, pt, BasisEvaluation<T, ndim>{el.basis, pt}}
  {}
};

template<class T, class IDX, int ndim>
PhysDomainEval(
    PhysDomainEvalStorage<T, ndim>&,
    const FiniteElement<T, IDX, ndim>&,
    const typename PhysDomainEval<T, ndim>::Point&,
    const BasisEvaluation<T, ndim>&) 
  -> PhysDomainEval<T, ndim>;
template<class T, class IDX, int ndim>
PhysDomainEval(
    PhysDomainEvalStorage<T, ndim>&,
    const FiniteElement<T, IDX, ndim>&,
    const typename PhysDomainEval<T, ndim>::Point&)
  -> PhysDomainEval<T, ndim>;

/**
 * @brief calculate the mass matrix for an element 
 * @param el the element to calculate the mass matrix for 
 * @param node_coords the global node coordinates array 
 * @return the mass matrix
 */
template<class T, class IDX, int ndim>
MATH::MATRIX::DenseMatrix<T> calculate_mass_matrix(
  const FiniteElement<T, IDX, ndim> &el
) {
  MATH::MATRIX::DenseMatrix<T> mass(el.nbasis(), el.nbasis());
  mass = 0;

  for(int ig = 0; ig < el.nQP(); ++ig){
    const QuadraturePoint<T, ndim> quadpt = el.getQP(ig);

    // calculate the jacobian determinant
    auto J = el.jacobian(quadpt.abscisse);
    T detJ = NUMTOOL::TENSOR::FIXED_SIZE::determinant(J);

    // integrate Bi * Bj
    int nbasis = el.nbasis();
    auto bi = el.eval_basis_qp(ig);
    for(int ibasis = 0; ibasis < nbasis; ++ibasis){
      for(int jbasis = 0; jbasis < nbasis; ++jbasis){
        mass[ibasis][jbasis] += bi[ibasis] * bi[jbasis] * quadpt.weight * detJ;
      }
    }
  }

  return mass;
}

} // namespace ELEMENT
