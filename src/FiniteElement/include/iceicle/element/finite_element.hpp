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

  /** @brief precomputed evaluations of the basis functions 
   * at the quadrature points 
   *
   * NOTE: this is stored by reference because many physical elements will share the same 
   * reference domain, basis functions, and quadrature rule 
   * therefore the evaluations will be the same
   *
   **/
  const FEEvaluation<T, IDX, ndim> &qp_evals;

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
  void evalBasis(const T *xi, T *Bi) const { basis->evalBasis(xi, Bi); }

  /**
   * @brief get the values of the basis functions at the given quadrature point
   * cost: one memcpy of size nbasis
   * @param [in] quadrature_pt_idx the index of the quadrature point in [0,
   * ngauss()]
   * @param [out] Bi the values of the basis functions at the point [size =
   * nbasis]
   */
  void evalBasisQP(int quadrature_pt_idx, T *Bi) const {
    const std::vector<T> &eval = qp_evals[quadrature_pt_idx];
    std::copy(eval.begin(), eval.end(), Bi);
  }

  /**
   * @brief directly access the evaluation of the given basis function at the
   * given quadrature point cost: pointer dereference
   * @param [in] quadrature_pt_idx the index of the quadrature point in [0,
   * ngauss()]
   * @param [in] ibasis the index of the basis function
   * @return the evaluation of the basis function at the given quadrature point
   */
  inline T basisQP(int quadrature_pt_idx, int ibasis) const {
    return qp_evals[quadrature_pt_idx][ibasis];
  }

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
  auto evalGradBasis(const T *xi, T *dBidxj) const {
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
   * TODO: precompute evaluation to reduce cost to a memcpy
   * @param [in] quadrature_pt_idx the index of the quadrature point [0,
   * ngauss()]
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
  auto evalGradBasisQP(int quadrature_pt_idx, T *dBidxj) const {
    basis->evalGradBasis(quadrule[quadrature_pt_idx].abscisse, dBidxj);
    std::experimental::extents<int, std::dynamic_extent, ndim> extents(
        nbasis());
    std::experimental::mdspan gbasis{dBidxj, extents};
    static_assert(gbasis.extent(1) == ndim);
    return gbasis;
  }

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
  auto evalPhysGradBasis(
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
  auto evalPhysGradBasis(
    const Point &xi,
    const JacobianType& J,
    T *dbidxj 
  ) const {
    // compute the basis functions in reference domain
    std::vector<T> dBi_data(ndim * nbasis());
    auto grad_bi = evalGradBasis(xi, dBi_data.data());

    return evalPhysGradBasis(xi, J, grad_bi, dbidxj);
  }

  /** Calculates the Jacobian and gradient wrt reference domain: \overload */
  auto evalPhysGradBasis(
    const Point &xi,
    T *dbidxj 
  ) const {
    // compute the basis functions in reference domain
    std::vector<T> dBi_data(ndim * nbasis());
    auto grad_bi = evalGradBasis(xi, dBi_data.data());

    // compute jacobian
    JacobianType J = trans->jacobian(coord_el, xi);

    return evalPhysGradBasis(xi, J, grad_bi, dbidxj);
  }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to physical domain coordinates at the given quadrature
   * point
   *
   * @param [in] quadrature_pt_idx the quadrature point index
   * @param [in] transformation the transformation from the reference domain to
   * the physical domain (must be compatible with the geometric element)
   * @param [in] J the jacobian of the element transformation 
   *               (optional: see overload)
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
  auto evalPhysGradBasisQP(
      int quadrature_pt_idx,
      const JacobianType &J,
      T *dBidxj
  ) const {
    // TODO: prestore
    return evalPhysGradBasis(
        (*quadrule)[quadrature_pt_idx].abscisse, J, dBidxj);
  }
  
  /** Calculates the Jacobian: \overload */
  auto evalPhysGradBasisQP(
      int quadrature_pt_idx,
      T *dBidxj
  ) const {
    // TODO: prestore
    return evalPhysGradBasis( (*quadrule)[quadrature_pt_idx].abscisse, dBidxj);
  }

  /**
   * @brief evaluate the second derivatives of the basis functions
   *
   * @param [in] xi the point in the reference domain to evaluate at
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial \xi_j \partial \xi_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto evalHessBasis(const Point &xi, T *basis_hessian_data) const {
    using namespace std::experimental;
    basis->evalHessBasis(xi, basis_hessian_data);
    extents<int, std::dynamic_extent, ndim, ndim> exts{nbasis()};
    mdspan hess_basis{basis_hessian_data, exts};
    return hess_basis;
  }

  /**
   * @brief evaluate the second derivatives of the basis functions in the
   * physical domain
   *
   * @param [in] xi the point in the reference domain to evaluate at
   * @param [in] coord the global node coordinates array
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial x_j \partial x_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto evalPhysHessBasis(
    const Point &xi,
    T *basis_hessian_data
  ) const {
    using namespace std::experimental;

    // pull nbasis to variable so its more likely to be put in register
    int ndof = nbasis();

    // Get the Transformation Jacobian and Hessian
    auto trans_jac = trans->jacobian(coord_el, xi);
    auto trans_hess = trans->hessian(coord_el, xi);

    // Evaluate basis function derivatives and second derivatives

    // derivatives wrt Physical Coordinates
    std::vector<T> dBi_data(ndim * ndof, 0.0);
    auto dBi = evalPhysGradBasis(xi, dBi_data.data());

    // Hessian wrt reference coordinates
    std::vector<T> hess_Bi_data(ndim * ndim * ndof, 0.0);
    auto hessBi = evalHessBasis(xi, hess_Bi_data.data());

    // multidimensional array view of hessian
    extents<int, std::dynamic_extent, ndim, ndim> exts{ndof};
    mdspan hess_phys{basis_hessian_data, exts};

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
    // (put this result in hessBi overwriting the values)
    for (idof = 0; idof < ndof; ++idof) {
      for (id = 0; id < ndim; ++id) {
        for (jd = 0; jd < ndim; ++jd) {
          for (kd = 0; kd < ndim; ++kd) {
            hessBi[idof, id, jd] -= trans_hess[kd][id][jd] * dBi[idof, kd];
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
                  adjJ[kd][jd] * adjJ[ld][id] * hessBi[idof, ld, kd];
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

  /**
   * @brief evaluate the second derivatives of the basis functions in the
   * physical domain
   *
   * @param [in] iqp the quadrature point index
   * @param [in] coord the global node coordinates array
   * @param [out] basis_hessian_data pointer to the 1d array to store the
   * hessian data in ordered i, j, k in C style array for \frac{\partial^2
   * B_i}{\partial x_j \partial x_k}
   * @return a multidimensional array view of the basis_hessian_data
   */
  auto evalPhysHessBasisQP(
    int iqp,
    NodeArray<T, ndim> &coord,
    T *basis_hessian_data
  ) const {
    return evalPhysHessBasis((*quadrule)[iqp].abscisse, coord, basis_hessian_data);
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
    for(int ibasis = 0; ibasis < nbasis; ++ibasis){
      for(int jbasis = 0; jbasis < nbasis; ++jbasis){
        mass[ibasis][jbasis] += el.basisQP(ig, ibasis) * el.basisQP(ig, jbasis) * quadpt.weight * detJ;
      }
    }
  }

  return mass;
}

} // namespace ELEMENT
