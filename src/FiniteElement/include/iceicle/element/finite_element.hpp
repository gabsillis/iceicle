/**
 * @file finite_element.hpp
 * @brief A finite element which can get quadrature, evaluate basis functions,
 * and perform transformations
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include "Numtool/fixed_size_tensor.hpp"
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include <Numtool/matrix/dense_matrix.hpp>
#include <Numtool/matrixT.hpp>
#include <iceicle/basis/basis.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/quadrature/QuadratureRule.hpp>
#include <span>
#include <vector>

#include <mdspan/mdspan.hpp>

namespace ELEMENT {

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
  const BASIS::Basis<T, ndim> *basis;

  /** @brief the quadraature rule */
  const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule;

  FEEvaluation() = default;

  FEEvaluation(BASIS::Basis<T, ndim> *basisptr,
               QUADRATURE::QuadratureRule<T, IDX, ndim> *quadruleptr)
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

  FEEvaluation(BASIS::Basis<T, ndim> &basis,
               QUADRATURE::QuadratureRule<T, IDX, ndim> &quadrule)
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
 * Consists of:
 * A GeometricElement - to represent the transformation from reference to physical space 
 *                      and stores the node coordinates to represent the subdomain
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
class FiniteElement {

  using Point = MATH::GEOMETRY::Point<T, ndim>;
  using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;
  using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

public:
  /** @brief the geometric element (container for node indices and basic
   * geometric operations) */
  const GeometricElement<T, IDX, ndim> *geo_el;

  /** @brief the basis functions */
  const BASIS::Basis<T, ndim> *basis;

  /** @brief the quadraature rule */
  const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule;

  /** @brief precomputed evaluations of the basis functions 
   * at the quadrature points 
   *
   * NOTE: this is stored by reference because many physical elements will share the same 
   * reference domain, basis functions, and quadrature rule 
   * therefore the evaluations will be the same
   *
   **/
  const FEEvaluation<T, IDX, ndim> &qp_evals;

  /** @brief the element index in the mesh */
  const IDX elidx;

  /**
   * @brief construct a FiniteElement
   * @param geo_el_arg the geometric element
   * @param basis_arg the basis function
   * @param quadrule_arg the qudarature rule to use
   * @param qp_evals evaluations of the basis functions at the quadrature points
   * @param elidx_arg the element index in the mesh
   */
  FiniteElement(
    const GeometricElement<T, IDX, ndim> *geo_el_arg,
    const BASIS::Basis<T, ndim> *basis_arg,
    const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule_arg,
    const FEEvaluation<T, IDX, ndim> *qp_evals,
    IDX elidx_arg
  ) : geo_el(geo_el_arg), basis(basis_arg), quadrule(quadrule_arg),
      qp_evals(*qp_evals), elidx(elidx_arg) {}

  /**
   * @brief precompute any stored quantities
   * @param node_list the global node list which geo_el has indices into
   */
  void precompute(FE::NodalFEFunction<T, ndim> &node_list) {}

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
    std::experimental::extents<int, std::dynamic_extent, ndim> extents(
        nbasis());
    std::experimental::mdspan gbasis{dBidxj, extents};
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
   * @param [in] transformation the transformation from the reference domain to
   * the physical domain (must be compatible with the geometric element)
   * @param [in] node_list the list of global node coordinates
   * @param [in] J the jacobian of the element transformation 
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
    FE::NodalFEFunction<T, ndim> &node_list,
    const JacobianType &J,
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

    // Evaluate dBi in reference domain
    std::vector<T> dBi_data(ndim * ndof, 0.0);
    auto dBi = evalGradBasis(xi, dBi_data.data());

    std::experimental::extents<int, std::dynamic_extent, ndim> extents(
        ndof);
    std::experimental::mdspan gbasis{dBidxj, extents};
    // dBidxj =  Jadj_{jk} * dBidxk
    for (int i = 0; i < ndof; ++i) {
      for (int j = 0; j < ndim; ++j) {
        // gbasis[i, j] = 0.0;
        for (int k = 0; k < ndim; ++k) {
          gbasis[i, j] += dBi[i, k] * adjJ[k][j];
        }
      }
    }

    // multiply though by the determinant
    for (int i = 0; i < ndof * ndim; ++i) {
      dBidxj[i] /= detJ;
    }

    return gbasis;
  }

  /** Calculates the Jacobian: \overload */
  auto evalPhysGradBasis(
    const Point &xi,
    FE::NodalFEFunction<T, ndim> &node_list,
    T *dbidxj 
  ) const {
    JacobianType J = geo_el->Jacobian(node_list, xi);
    return evalPhysGradBasis(xi, node_list, J, dbidxj);
  }

  /**
   * @brief evaluate the first derivatives of the basis functions
   *        with respect to physical domain coordinates at the given quadrature
   * point
   *
   * @param [in] quadrature_pt_idx the quadrature point index
   * @param [in] transformation the transformation from the reference domain to
   * the physical domain (must be compatible with the geometric element)
   * @param [in] node_list the list of global node coordinates
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
      FE::NodalFEFunction<T, ndim> &node_list,
      const JacobianType &J,
      T *dBidxj
  ) const {
    // TODO: prestore
    return evalPhysGradBasis(
        (*quadrule)[quadrature_pt_idx].abscisse,
        node_list, J, dBidxj);
  }
  
  /** Calculates the Jacobian: \overload */
  auto evalPhysGradBasisQP(
      int quadrature_pt_idx,
      FE::NodalFEFunction<T, ndim> &node_list,
      T *dBidxj
  ) const {
    // TODO: prestore
    return evalPhysGradBasis(
        (*quadrule)[quadrature_pt_idx].abscisse,
        node_list, dBidxj);
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
    FE::NodalFEFunction<T, ndim> &coord,
    T *basis_hessian_data
  ) const {
    using namespace std::experimental;

    // pull nbasis to variable so its more likely to be put in register
    int ndof = nbasis();

    // Get the Transformation Jacobian and Hessian
    auto trans_jac = geo_el->Jacobian(coord, xi);
    auto trans_hess = geo_el->Hessian(coord, xi);

    // Evaluate basis function derivatives and second derivatives

    // derivatives wrt Physical Coordinates
    std::vector<T> dBi_data(ndim * ndof, 0.0);
    auto dBi = evalPhysGradBasis(xi, coord, dBi_data.data());

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
    FE::NodalFEFunction<T, ndim> &coord,
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
  const QUADRATURE::QuadraturePoint<T, ndim> getQP(int qp_idx) const {
    return (*quadrule)[qp_idx];
  }

  // ========================
  // = Geometric Operations =
  // ========================

  /**
   * @brief transform from the reference domain to the physical domain
   * @param [in] node_coords the coordinates of all the nodes
   * @param [in] pt_ref the point in the refernce domain
   * @param [out] pt_phys the point in the physical domain
   */
  inline void transform(FE::NodalFEFunction<T, ndim> &node_coords,
                        const Point &pt_ref, Point &pt_phys) const {
    return geo_el->transform(node_coords, pt_ref, pt_phys);
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
  const FiniteElement<T, IDX, ndim> &el,
  FE::NodalFEFunction<T, ndim> &node_coords
) {
  MATH::MATRIX::DenseMatrix<T> mass(el.nbasis(), el.nbasis());
  mass = 0;

  for(int ig = 0; ig < el.nQP(); ++ig){
    const QUADRATURE::QuadraturePoint<T, ndim> quadpt = el.getQP(ig);

    // calculate the jacobian determinant
    auto J = el.geo_el->Jacobian(node_coords, quadpt.abscisse);
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
