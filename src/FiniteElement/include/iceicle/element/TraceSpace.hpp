/**
 * @file TraceSpace.hpp
 * @brief Trace Space (element faces)
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "Numtool/point.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/element/finite_element.hpp>

#include <iceicle/geometry/face.hpp>

#include <cassert>
#include <mdspan/mdspan.hpp>

namespace iceicle {

/**
 * @brief TraceSpace represents a contiguous subsection of the Trace of finite elements 
 * The trace is the boundary of the finite element domain 
 * 
 * For conformal meshes, this is shared between two elements
 * dubbed the Left and Right elements (indexed respectively)
 * Convention for the left and right is that the normal vector 
 * points from the left element into the right element
 *
 * NOTE: if this is a boundary face, then the Left element is the interior element 
 * by convention. However, to satisfy reference invariants, the finite element 
 * for the right element is set to the left element as well
 *
 * The reference domain for the trace space or, in short, reference trace space
 * is a subset of \mathbb{R}^{d-1} where d = ndim is the dimensionality of the 
 * finite element domains
 *
 * The reference trace space is defined as the space oriented with the left element 
 * for operations on the right element this requires the orientation transformation
 * 
 * @tparam T the data type 
 * @tparam IDX the index type 
 * @tparam ndim the number of dimensions of the element domains \mathbb{R}^{d}
 */
template<typename T, typename IDX, int ndim>
class TraceSpace {

  // ===========
  // = Aliases =
  // ===========
private:
  // None
public:
  /// @brief the type of geometric faces 
  using FaceType = Face<T, IDX, ndim>;

  /// @brief the type of Geometric Elements (transformation definition)
  using GeoElementType = GeometricElement<T, IDX, ndim>;

  /// @brief the type of the FiniteElement
  using FEType = FiniteElement<T, IDX, ndim>;

  /// @brief the type of the quadrature in the reference trace space 
  using QuadratureType = QuadratureRule<T, IDX, ndim - 1>;

  /// @brief a point with domain dimensionality
  using DomainPoint = MATH::GEOMETRY::Point<T, ndim>;

  /// @brief a point with face dimensionality
  using FacePoint = MATH::GEOMETRY::Point<T, ndim - 1>;

  /// @brief the type of the basis functions in the trace space 
  using TraceBasis = Basis<T, ndim - 1>;

  /// @brief the type of evaluations for basis functions
  using Eval_t = BasisEvaluation<T, ndim>;

  // ================
  // = Data Members =
  // ================
private:
  // None
public:
  const FaceType *face;
  /// @brief the left finite element 
  /// (WARNING: if vector of elements changes this will probably need to be updated)
  const FEType &elL;
  /// @brief the right finite element
  const FEType &elR;
  /// @brief the Basis over the continuous trace space 
  const TraceBasis &trace_basis; 
  /// @brief the quadrature rule for integration on the trace
  const QuadratureType &quadrule;
  /// @brief the precomputed basis functions on the left
  std::span<const Eval_t> qp_evals_l;
  /// @brief the precomputed basis functions on the right 
  std::span<const Eval_t> qp_evals_r;
  /// @brief the index of this face in the container used to organize faces
  const IDX facidx;

  // ================
  // = Constructors =
  // ================
public:

  /**
   * @brief Constructor 
   * @param facptr pointer to the geometric face 
   * @param elLptr pointer to the left FiniteElement 
   * @param elRptr pointer to the right FiniteElement
   * @param quadruleptr pointer to the trace quadrature rule
   * @param trace_basisptr pointer to the basis function over the trace space
   * @param qp_evals_l the precomputed basis functions at quadrature points on the left
   * @param qp_evals_r the precomputed basis functions at quadrature points on the right 
   * @param facidx the index of this face in the container
   */
  TraceSpace(
      const FaceType *facptr,
      const FEType *elLptr,
      const FEType *elRptr,
      const TraceBasis *trace_basisptr,
      const QuadratureType *quadruleptr,
      std::span<const Eval_t> qp_evals_l,
      std::span<const Eval_t> qp_evals_r,
      IDX facidx
  ) : face(facptr), elL(*elLptr), elR(*elRptr), trace_basis(*trace_basisptr),
      quadrule(*quadruleptr), qp_evals_l(qp_evals_l), qp_evals_r(qp_evals_r),
      facidx(facidx) 
  {
    // TODO: can't do assertion because we call from make_bdy_trace_space
 //   assert((facptr->bctype == BOUNDARY_CONDITIONS::INTERIOR) 
 //       && "The given face is not an interior face->");
  }

  /**
   * @brief create a boundary face
   * special case named constructor
   * separated from regular constructors to encourage intentional use 
   *
   * NOTE: reuses elL as elR
   *
   * @param facptr pointer to the geometric face 
   * @param elLptr pointer to the interior FiniteElement 
   * @param quadruleptr pointer to the trace quadrature rule
   * @param qp_evals_ptr pointer to the quadrature evaluations
   * @param facidx the index of this face in the container
   */
  static constexpr TraceSpace<T, IDX, ndim> make_bdy_trace_space(
      const FaceType *facptr,
      const FEType *elLptr,
      const TraceBasis *trace_basisptr,
      const QuadratureType *quadruleptr,
      std::span<const Eval_t> qp_evals_l,
      std::span<const Eval_t> qp_evals_r,
      IDX facidx
  ) {
    assert((facptr->bctype != BOUNDARY_CONDITIONS::INTERIOR) 
        && "The given face is not a boundary face->");
    return TraceSpace(facptr, elLptr, elLptr, trace_basisptr, quadruleptr,
        qp_evals_l, qp_evals_r , facidx);
  }

  // =============================
  // = Basis Function Operations =
  // =============================
 
  /// @brief get the number of basis functions on the left element
  inline int nbasisL() const { return elL.nbasis(); }

  /// @brief get the number of basis functions on the right element 
  inline int nbasisR() const { return elR.nbasis(); }

  /// @brief get the number of basis functions in the continuous trace space
  inline int nbasis_trace() const { return trace_basis.nbasis(); }

  // === Function Value ===

  /**
   * @brief calculate the basis functions on the left element 
   * at a point in the reference trace space 
   * @param [in] s the location in the reference trace space 
   *        NOTE: this is the reference trace space from the 
   *        orientation of the left element 
   *
   * @param [out] Bi the values of the basis functions on the left 
   * (size = nbasis)
   */
  void eval_basis_l(const FacePoint &s, T *Bi) const {
    DomainPoint xi{};
    face->transform_xiL(s, xi);
    elL.eval_basis(xi, Bi);
  }

  /**
   * @brief calculate the basis functions on the right element 
   * at a point in the reference trace space 
   * @param [in] s the location in the reference trace space 
   *        NOTE: this is the reference trace space from the 
   *        orientation of the left element 
   *
   * @param [out] Bi the values of the basis functions on the right 
   * (size = nbasis)
   */
  void eval_basis_r(const FacePoint &s, T *Bi) const {
    DomainPoint xi{};
    face->transform_xiR(s, xi);
    elR.eval_basis(xi, Bi);
  }

  /**
   * @brief calculate the basis functions in the trace space 
   * @param [in] s the location in the reference trace space 
   * @param [out] Bi the value of the basis functions 
   */
  void eval_trace_basis(const FacePoint &s, T *Bi) const {
    trace_basis.evalBasis(s, Bi);
  }

  /**
   * @brief get the basis function evaluations at the given quadrature point 
   * @param [in] quadrature_idx the index of the quadrature point 
   * @param [out] Bi the values of the basis functions on the left 
   *              (size = nbasis)
   */
  auto eval_basis_l_qp(int quadrature_idx) const noexcept
  -> Eval_t::bi_span_t 
  { return qp_evals_l[quadrature_idx].bi_span; }

  /**
   * @brief get the basis function evaluations at the given quadrature point 
   * @param [in] quadrature_idx the index of the quadrature point 
   * @param [out] Bi the values of the basis functions on the right
   *              (size = nbasis)
   */
  auto eval_basis_r_qp(int quadrature_idx) const noexcept
  -> Eval_t::bi_span_t 
  { return qp_evals_r[quadrature_idx].bi_span; }

  /**
   * @brief get the basis function evaluations at the given quadrature point 
   * @param [in] quadrature_idx the index of the quadrature point 
   * @param [out] Bi the values of the basis functions on the trace space
   *              (size = nbasis_trace())
   */
  void eval_trace_basis_qp(int quadrature_idx, T *Bi) const {
    return eval_trace_basis(quadrule[quadrature_idx].abscisse, Bi);
  }

  // === First Derivatives ===

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to reference domain coordinates
   *        on the left element
   *
   * @param [in] s the point in the reference trace space 
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
  auto eval_grad_basis_l(const FacePoint &s, T *grad_data) const {
    DomainPoint xi{};
    face->transform_xiL(s, xi);
    return elL.eval_grad_basis(xi, grad_data);
  }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to reference domain coordinates
   *        on the right element
   *
   * @param [in] s the point in the reference trace space 
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
  auto eval_grad_basis_r(const FacePoint &s, T *grad_data) const {
    DomainPoint xi{};
    face->transform_xiR(s, xi);
    return elR.eval_grad_basis(xi, grad_data);
  }

  /**
   * @brief evaluate the first derivaives of the basis functions
   *        with respect to physical domain coordinates 
   *        on the left element 
   *  NOTE: this does not reuse transformation jacobian
   *
   * @param [in] s the point in the reference trace space 
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
  auto eval_phys_grad_basis_l(
      const FacePoint &s,
      T *grad_data
  ) const {
    DomainPoint xi{};
    face->transform_xiL(s, xi);
    return elL.eval_phys_grad_basis(xi, grad_data);
  }

  /**
   * @brief evaluate the first derivaives of the basis functions
   *        with respect to physical domain coordinates 
   *        on the right element 
   *  NOTE: this does not reuse transformation jacobian
   *
   * @param [in] s the point in the reference trace space 
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
  auto eval_phys_grad_basis_r(
      const FacePoint &s,
      T *grad_data
  ) const {
    DomainPoint xi{};
    face->transform_xiR(s, xi);
    return elR.eval_phys_grad_basis(xi, grad_data);
  }


  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to reference domain coordinates
   *        on the left element
   *
   * @param [in] qidx the quadrature point index
   *
   * @return an mdspan view of dBidxj for an easy interface
   *         \frac{dB_i}{d\xi_j} where i is ibasis
   *         takes a pointer to the first element of this data structure
   *         [size = [nbasis : i][ndim : j]]
   */
  auto eval_grad_basis_l_qp(int qidx) const noexcept
  -> Eval_t::grad_span_t 
  { return qp_evals_l[qidx].grad_bi_span; }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to reference domain coordinates
   *        on the right element
   *
   * @param [in] qidx the quadrature point index
   *
   * @return an mdspan view of dBidxj for an easy interface
   *         \frac{dB_i}{d\xi_j} where i is ibasis
   *         takes a pointer to the first element of this data structure
   *         [size = [nbasis : i][ndim : j]]
   */
  auto eval_grad_basis_r_qp(int qidx, T *grad_data) const noexcept 
  -> Eval_t::grad_span_t
  { return qp_evals_r[qidx].grad_bi_span; }

  /**
   * @brief evaluate the first derivaives of the basis functions
   *        with respect to physical domain coordinates 
   *        on the left element 
   *
   * @param [in] qidx the quadrature point index
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
  auto eval_phys_grad_basis_l_qp(
      int qidx,
      T *grad_data
  ) const {
    DomainPoint xi{};
    face->transform_xiL(quadrule[qidx].abscisse, xi);
    auto el_jac = elL.jacobian(xi);
    return elL.eval_phys_grad_basis(xi, el_jac, qp_evals_l[qidx].grad_bi_span, grad_data);
  }

  /**
   * @brief evaluate the first derivaives of the basis functions
   *        with respect to physical domain coordinates 
   *        on the right element 
   *
   * @param [in] qidx the quadrature point index
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
  auto eval_phys_grad_basis_r_qp(
      int qidx,
      T *grad_data
  ) const {
    DomainPoint xi{};
    face->transform_xiR(quadrule[qidx].abscisse, xi);
    auto el_jac = elR.jacobian(xi);
    return elR.eval_phys_grad_basis(xi, el_jac, qp_evals_r[qidx].grad_bi_span, grad_data);
  }

  // === Second Derivatives ===

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        with respect to reference domain coordinates 
   *        for the left element
   *
   * @param [in] s the point in the reference trace space
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial \xi_j \partial \xi_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_hess_basis_l(const FacePoint &s, T *hess_data) const {
    DomainPoint xi{};
    face->transform_xiL(s, xi);
    return elL.eval_hess_basis(xi, hess_data);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        with respect to reference domain coordinates 
   *        for the right element
   *
   * @param [in] s the point in the reference trace space
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial \xi_j \partial \xi_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_hess_basis_r(const FacePoint &s, T *hess_data) const {
    DomainPoint xi{};
    face->transform_xiR(s, xi);
    return elR.eval_hess_basis(xi, hess_data);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        wrt physical domain coordinates 
   *        for the left element
   *
   * @param [in] s the point in the reference trace space
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial x_j \partial x_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_phys_hess_basis_l(
      const FacePoint &s,
      T *hess_data
  ) const {
    DomainPoint xi{};
    face->transform_xiL(s, xi);
    return elL.eval_phys_hess_basis(xi, hess_data);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        wrt physical domain coordinates 
   *        for the right element
   *
   * @param [in] s the point in the reference trace space
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial x_j \partial x_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_phys_hess_basis_r(
      const FacePoint &s,
      T *hess_data
  ) const {
    DomainPoint xi{};
    face->transform_xiR(s, xi);
    return elR.eval_phys_hess_basis(xi, hess_data);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        with respect to reference domain coordinates 
   *        for the left element
   *
   * @param [in] qidx the quadrature point index
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_hess_basis_l_qp(int qidx) const noexcept 
  -> Eval_t::hess_span_t
  { return qp_evals_l[qidx].hess_bi; }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *        with respect to reference domain coordinates 
   *        for the right element
   *
   * @param [in] qidx the quadrature point index
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto eval_hess_basis_r_qp(int qidx, T *hess_data) const noexcept 
  -> Eval_t::hess_span_t
  { return qp_evals_r[qidx].hess_bi; }

  // =========================
  // = Quadrature Operations =
  // =========================

  /** @brief get the number of quadrature points in the quadrature rule */
  int nQP() const { return quadrule.npoints(); }

  /** 
   * @brief get the "QuadraturePoint" (contains point and weight) at the given
   * quadrature point index 
   **/
  inline constexpr const QuadraturePoint<T, ndim - 1> 
  getQP(int qp_idx) const
  {
    return quadrule[qp_idx];
  }

  // ========================
  // = Geometric Operations =
  // ========================

  /// @brief transform froom the face reference domain to the physical domain 
  /// @param s the face reference domain point 
  /// @param coord the node coordinates
  /// @return the physical domain coordinates
  inline constexpr
  DomainPoint transform( const FacePoint& s, NodeArray<T, ndim>& coord ) const
  {
    DomainPoint x{};
    face->transform(s, coord, x);
    return x;
  }

  /// @brief transform froom the face reference domain to the reference domain of the left element
  /// @param s the face reference domain point 
  /// @return the coordinates in the reference domain of the left element
  inline constexpr
  DomainPoint transform_xiL( const FacePoint& s) const
  {
    DomainPoint xiL{};
    face->transform_xiL(s, xiL);
    return xiL;
  }

  /// @brief transform froom the face reference domain to the reference domain of the right element
  /// @param s the face reference domain point 
  /// @return the coordinates in the reference domain of the right element
  inline constexpr
  DomainPoint transform_xiR( const FacePoint& s) const
  {
    DomainPoint xiR{};
    face->transform_xiR(s, xiR);
    return xiR;
  }
};

}
