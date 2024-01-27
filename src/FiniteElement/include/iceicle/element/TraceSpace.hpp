/**
 * @file TraceSpace.hpp
 * @brief Trace Space (element faces)
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "Numtool/point.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/quadrature/QuadratureRule.hpp"
#include <iceicle/element/finite_element.hpp>

#include <iceicle/geometry/face.hpp>

#include <vector>
#include <cassert>
#include <mdspan/mdspan.hpp>

namespace ELEMENT {

/**
 * @brief Precomputed values of basis functions and other reference domain quantities
 */
template<typename T, typename IDX, int ndim>
class TraceEvaluation {
public:
  using FaceType = ELEMENT::Face<T, IDX, ndim>;
  using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;
  using FEType = ELEMENT::FiniteElement<T, IDX, ndim>;

private:

  /** @brief the 1d array of data to store */
  std::vector<T> data_;

public:
  TraceEvaluation(){}
  TraceEvaluation(const FEType &fe_l, const FEType &fe_r) {
  }
};

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
public:
  /// @brief the type of geometric faces 
  using FaceType = ELEMENT::Face<T, IDX, ndim>;

  /// @brief the type of Geometric Elements (transformation definition)
  using GeoElementType = ELEMENT::GeometricElement<T, IDX, ndim>;

  /// @brief the type of the FiniteElement
  using FEType = ELEMENT::FiniteElement<T, IDX, ndim>;

  /// @brief the type of the quadrature in the reference trace space 
  using QuadratureType = QUADRATURE::QuadratureRule<T, IDX, ndim - 1>;

  /// @brief a point with domain dimensionality
  using DomainPoint = MATH::GEOMETRY::Point<T, ndim>;

  /// @brief a point with face dimensionality
  using FacePoint = MATH::GEOMETRY::Point<T, ndim - 1>;

  // ================
  // = Data Members =
  // ================
private:
public:
  const FaceType &face;
  /// @brief the left finite element 
  const FEType &elL;
  /// @brief the right finite element
  const FEType &elR;
  /// @brief the quadrature rule for integration on the trace
  const QuadratureType &quadrule;
  /// @brief the precomputed quantities
  const TraceEvaluation<T, IDX, ndim> &qp_evals;
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
   * @param qp_evals_ptr pointer to the quadrature evaluations
   * @param facidx the index of this face in the container
   */
  TraceSpace(
      const FaceType *facptr,
      const FEType *elLptr,
      const FEType *elRptr,
      const QuadratureType *quadruleptr,
      const TraceEvaluation<T, IDX, ndim> *qp_evals_ptr,
      IDX facidx
  ) : face(*facptr), elL(*elLptr), elR(*elRptr), quadrule(*quadruleptr),
      qp_evals(*qp_evals_ptr), facidx(facidx) 
  {
    assert((facptr->bctype == ELEMENT::BOUNDARY_CONDITIONS::INTERIOR) 
        && "The given face is not an interior face.");
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
      const QuadratureType *quadruleptr,
      const TraceEvaluation<T, IDX, ndim> *qp_evals_ptr,
      IDX facidx
  ) {
    assert((facptr->bctype != ELEMENT::BOUNDARY_CONDITIONS::INTERIOR) 
        && "The given face is not a boundary face.");
    return TraceSpace(facptr, elLptr, elLptr, quadruleptr, qp_evals_ptr, facidx);
  }

  // =============================
  // = Basis Function Operations =
  // =============================
 
  /// @brief get the number of basis functions on the left element
  inline int nbasisL() const { return elL.nbasis(); }

  /// @brief get the number of basis functions on the right element 
  inline int nbasisR() const { return elR.nbasis(); }

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
  void evalBasisL(const FacePoint &s, T *Bi) const {
    DomainPoint xi{};
    face.transform_xiL(s, xi);
    elL.evalBasis(xi, Bi);
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
  void evalBasisR(const FacePoint &s, T *Bi) const {
    DomainPoint xi{};
    face.transform_xiR(s, xi);
    elR.evalBasis(xi, Bi);
  }

};


}
