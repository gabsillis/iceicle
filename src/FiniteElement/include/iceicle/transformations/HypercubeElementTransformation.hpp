#pragma once
#include "Numtool/point.hpp"
#include <Numtool/constexpr_math.hpp>
#include <Numtool/integer_utils.hpp>
#include <Numtool/polydefs/LagrangePoly.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <Numtool/matrix/dense_matrix.hpp>
#include <Numtool/fixed_size_tensor.hpp>
#include <array>
#include <cmath>
#include <iceicle/fe_function/nodal_fe_function.hpp>

#include <algorithm>
#include <sstream>
#include <string>

namespace ELEMENT::TRANSFORMATIONS {
// forward declaration for friend 
template<typename T, typename IDX, int ndim, int Pn>
class HypercubeTraceTransformation;

/**
 * @brief Transformations from the [-1, 1] hypercube
 * to an arbitrary order hypercube element
 *
 * NOTE: Convention: 
 * Nodes - refers to all points that define the geometry (includes inner lagrange nodes)
 * Vertices - refers to endpoints
 *
 * @tparam T the floating point type
 * @tparam IDX the index type for large lists
 * @tparam ndim the number of dimensions
 * @tparam Pn the polynomial order of the element
 */
template <typename T, typename IDX, int ndim, int Pn>
class HypercubeElementTransformation {
private:
  // === Aliases ===
  using Point = MATH::GEOMETRY::Point<T, ndim>;
  using PointView = MATH::GEOMETRY::PointView<T, ndim>;

  // === Constants ===
  static constexpr int nfac = 2 * ndim;
  static constexpr int nnode = MATH::power_T<Pn + 1, ndim>::value;
  static constexpr int nvert = MATH::power_T<2, ndim>::value;
  static constexpr int nfacevert = MATH::power_T<2, ndim - 1>::value;
  static constexpr int nfacenode = MATH::power_T<Pn + 1, ndim -1>::value;
  static constexpr NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, ndim> strides = []{
    NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, ndim> ret{};
    NUMTOOL::TMP::constexpr_for_range<0, ndim>([&]<int i>{
      ret[i] = MATH::power_T<Pn + 1, ndim - i - 1>::value;
    });
    return ret;
  }();

  // ==== Friends ===
  // this is the domain for a face in d+1 dimensions
  friend class HypercubeTraceTransformation<T, IDX, ndim + 1, Pn>;

public:
  int convert_indices_helper(int ijk[ndim]){
    int ret = 0;
    for(int idim = 0; idim < ndim; ++idim){
      ret += ijk[idim] * std::pow(Pn + 1, ndim - idim - 1);
    }
    return ret;
  }

  /// Basis function indices by dimension for each node
  static constexpr std::array<std::array<int, ndim>, nnode> ijk_poin = []() {
    std::array<std::array<int, ndim>, nnode> ret{};

    NUMTOOL::TMP::constexpr_for_range<0, ndim>([&ret]<int idim>() {
      // number of times to repeat the loop over basis functions
      const int nrepeat = MATH::power_T<Pn + 1, idim>::value;
      // the size that one loop through the basis function indices gives
      const int cyclesize = MATH::power_T<Pn + 1, ndim - idim>::value;
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            [irep, &ret]<int ibasis>() {
              const int nfill = NUMTOOL::TMP::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                const int offset = irep * cyclesize + start_offset;
                ret[offset + ifill][idim] = ibasis;
              }
            });
      }
    });

    return ret;
  }();

  /// Nodes in the reference domain
  static inline std::array<Point, nnode> xi_poin = [] {
    std::array<Point, nnode> ret{};

    NUMTOOL::TMP::constexpr_for_range<0, ndim>([&ret]<int idim>() {
      // number of times to repeat the loop over basis functions
      const int nrepeat = MATH::power_T<Pn + 1, idim>::value;
      // the size that one loop through the basis function indices gives
      const int cyclesize = MATH::power_T<Pn + 1, ndim - idim>::value;
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            [irep, &ret]<int ibasis>() {
              const T dx = 2.0 / (Pn);

              const int nfill = NUMTOOL::TMP::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                const int offset = irep * cyclesize + start_offset;
                ret[offset + ifill][idim] = -1.0 + dx * ibasis;
              }
            });
      }
    });

    return ret;

  }();

  /**
   * @brief Interpolation of a uniform set of Pn + 1 points from -1 to 1 
   */
  struct UniformLagrangeInterpolation {
    template<typename T1, std::size_t... sizes>
    using Tensor = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T1, sizes...>;

    /// compile-time precompute the evenly spaced interpolation points
    static constexpr Tensor<T, Pn + 1> xi_nodes = []{
      Tensor<T, Pn + 1> ret = {};
      if constexpr (Pn == 0) {
        // finite volume should recover cell center
        // for consistency
        ret[0] = 0.0; 
      } else {
        T dx = 2.0 / Pn;
        ret[0] = -1.0;
        for(int j = 1; j < Pn + 1; ++j){
          // better for numerics than j * dx
          ret[j] = ret[j - 1] + dx;
        }
      }
      return ret;
    }();

    /// compile-time precompute the lagrange polynomial denominators
    /// the barycentric weights
    static constexpr Tensor<T, Pn + 1> wj = []{
      Tensor<T, Pn + 1> ret;
      for(int j = 0; j < Pn + 1; ++j){
        ret[j] = 1.0;
        for(int k = 0; k < Pn + 1; ++k) if(k != j) {
          ret[j] *= (xi_nodes[j] - xi_nodes[k]);
        }
        // invert (this is a denominator)
        // NOTE: Berrut, Trefethen have an optimal way to compute this 
        // but we don't because its computed at compile time
        ret[j] = 1.0 / ret[j];
      }
      return ret;
    }();

    /**
     * @brief Evaluate every interpolating polynomial at the given point 
     * @param xi the point to evaluate at 
     * @return an array of all the evaluations
     */
    Tensor<T, Pn + 1> eval_all(T xi) const {
      Tensor<T, Pn + 1> Nj{};

      // finite volume case
      if constexpr (Pn == 0){
        Nj[0] = 1.0;
      }

      // run-time precompute the product of differences
      T lskip = 1; // this is the product skipping the node closest to xi
      int k; // this will be used to determine which node to skip
      for(k = 0; k < Pn; ++k){
        // make sure k+1 isn't the closest node to xi
        if( xi >= (xi_nodes[k] + xi_nodes[k+1])/2 ){
          lskip *= (xi - xi_nodes[k]);
        } else {
          break; // k is the closest node to xi
        }
      }
      for(int i = k + 1; i < Pn + 1; ++i){
        lskip *= (xi - xi_nodes[i]);
      }
      T lprod = lskip * (xi - xi_nodes[k]);

      // calculate Nj 
      int j;
      for(j = 0; j < k; ++j){
        Nj[j] = lprod * wj[j] / (xi - xi_nodes[j]);
      }
      Nj[k] = lskip * wj[k];
      for(++j; j < Pn + 1; ++j){
        Nj[j] = lprod * wj[j] / (xi - xi_nodes[j]);
      }

      return Nj;
    }

    /**
     * @brief Get the value and derivative of 
     * every interpolating polynomial at the given point 
     * @param xi the point to evaluate at 
     * @return an array of all the evaluations
     */
    void deriv_all(
        T xi,
        Tensor<T, Pn+1> &Nj,
        Tensor<T, Pn+1> &dNj
      ) const {

      // finite volume case
      if constexpr (Pn == 0){
        Nj[0] = 1.0;
        dNj[0] = 0.0;
      }

      // run-time precompute the product of differences
      T lskip = 1; // this is the product skipping the node closest to xi
      int k; // this will be used to determine which node to skip
      for(k = 0; k < Pn; ++k){
        // make sure k+1 isn't the closest node to xi
        if( xi >= (xi_nodes[k] + xi_nodes[k+1])/2 ){
          lskip *= (xi - xi_nodes[k]);
        } else {
          break; // k is the closest node to xi
        }
      }
      for(int i = k + 1; i < Pn + 1; ++i){
        lskip *= (xi - xi_nodes[i]);
      }
      T lprod = lskip * (xi - xi_nodes[k]);

      // calculate the sum of inverse differences
      // neglecting the skipped node
      // And calculate Nj in the same loops
      T s = 0.0;
      int j;
      for(j = 0; j < k; ++j){
        T inv_diff = 1.0 / (xi - xi_nodes[j]);
        s += inv_diff;
        Nj[j] = lprod * inv_diff * wj[j];
      }
      Nj[k] = lskip * wj[k];
      for(++j; j < Pn + 1; ++j){
        T inv_diff = 1.0 / (xi - xi_nodes[j]);
        s += inv_diff;
        Nj[j] = lprod * inv_diff * wj[j];
      }

      // run-time precompute the derivative of the l-product 
      T lprime = lprod * s + lskip;

      // evaluate the derivatives
      for(j = 0; j < k; ++j){
        // quotient rule
        dNj[j] = (lprime * wj[j] - Nj[j]) / (xi - xi_nodes[j]);
      }
      dNj[k] = s * Nj[k];
      for(++j; j < Pn + 1; ++j){
        // quotient rule
        dNj[j] = (lprime * wj[j] - Nj[j]) / (xi - xi_nodes[j]);
      }
    }
  };

  UniformLagrangeInterpolation interpolation_1d{};

  /**
   * @brief fill the array with shape functions at the given point
   * @param [in] xi the point in the reference domain to evaluate the basis at
   * @param [out] Bi the shape function evaluations
   */
  inline void fill_shp(const Point &xi, T *Bi) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // run-time precompute the lagrange polynomial evaluations 
    // for each coordinate
    Tensor<T, ndim, Pn + 1> lagrange_evals{};
    for(int idim = 0; idim < ndim; ++idim){
      lagrange_evals[idim] = interpolation_1d.eval_all(xi[idim]);
    }

    // for the first dimension (fencepost)
    NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
        [&]<int ibasis>(T xi_dim) {
          static constexpr int nfill = MATH::power_T<Pn + 1, ndim - 1>::value;
          T Bi_idim = lagrange_evals[0][ibasis];
          std::fill_n(Bi + nfill * ibasis, nfill, Bi_idim);
        },
        xi[0]);

    for (int idim = 1; idim < ndim; ++idim) {
      T xi_dim = xi[idim];

      // number of times to repeat the loop over basis functions
      int nrepeat = std::pow(Pn + 1, idim);
      // the size that one loop through the basis function indices gives
      const int cyclesize = std::pow(Pn + 1, ndim - idim);
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            [&]<int ibasis>(int idim, T xi_dim, T *Bi) {
              // evaluate the 1d basis function at the idimth coordinate
              T Bi_idim = lagrange_evals[idim][ibasis];
              const int nfill = std::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                Bi[start_offset + ifill] *= Bi_idim;
              }
            },
            idim, xi_dim, Bi + irep * cyclesize);
      }
    }
  }

  void fill_deriv_alt(const Point &xi, T dBidxj[nnode][ndim]) const {
    std::fill_n(*dBidxj, nnode * ndim, 1.0);
    for(int inode = 0; inode < nnode; ++inode){
      for(int idim = 0; idim < ndim; ++idim){
        for(int jdim = 0; jdim < ndim; ++jdim){
          if(idim == jdim){
            dBidxj[inode][jdim] *= POLYNOMIAL::dlagrange1d<T, Pn>(ijk_poin[inode][idim], xi[idim]);
          } else {
            dBidxj[inode][jdim] *= POLYNOMIAL::lagrange1d<T, Pn>(ijk_poin[inode][idim], xi[idim]);
          }
        }
      }
    }
  }

  /**
   * @brief fill the given 2d array with the derivatives of each basis function 
   * @param [in] xi the point in the reference domain to evaluate the derivative at 
   * @param [out] dBidxj the derivatives of the basis functions 
   */
  void fill_deriv(const Point &xi, NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, nnode, ndim> &dBidxj) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // run-time precompute the lagrange polynomial evaluations 
    // and derivatives for each coordinate
    Tensor<T, ndim, Pn + 1> lagrange_evals{};
    Tensor<T, ndim, Pn + 1> lagrange_derivs{};
    for(int idim = 0; idim < ndim; ++idim){
      interpolation_1d.deriv_all(
          xi[idim], lagrange_evals[idim], lagrange_derivs[idim]);
    }
    // fencepost the loop at idim = 0
    NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
      [&]<int ibasis>(const Point &xi) {
        static constexpr int nfill = MATH::power_T<Pn + 1, ndim - 1>::value;
        T Bi_idim = lagrange_evals[0][ibasis];
        T dBi_idim = lagrange_derivs[0][ibasis];
        for(int ifill = 0; ifill < nfill; ++ifill){
            dBidxj[nfill * ibasis + ifill][0] = dBi_idim;
            for(int jdim = 1; jdim < ndim; ++jdim){
                dBidxj[nfill * ibasis + ifill][jdim] = Bi_idim;
            }
        }
      },
      xi[0]);
    
    NUMTOOL::TMP::constexpr_for_range<1, ndim>(
        [&]<int idim>(const Point &xi){
            // number of times to repeat the loop over basis functions
            const int nrepeat = std::pow(Pn + 1, idim);
            // the size that one loop through the basis function indices gives 
            const int cyclesize = std::pow(Pn + 1, ndim - idim);

            for(int irep = 0; irep < nrepeat; ++irep) {

                NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
                    [&, irep]<int ibasis>(const Point &xi) {
                        T dBi_idim = lagrange_derivs[idim][ibasis];
                        const int nfill = std::pow(Pn + 1, ndim - idim - 1);

                        // offset for multiplying by this ibasis
                        const int start_offset = ibasis * nfill;

                        // multiply the next nfill by the current basis function
                        for (int ifill = 0; ifill < nfill; ++ifill) {
                            const int offset = irep * cyclesize + start_offset;
                            NUMTOOL::TMP::constexpr_for_range<0, ndim>([&]<int jdim>(){
                                if constexpr(jdim == idim){
                                    dBidxj[offset + ifill][jdim] *= dBi_idim;
                                } else {
                                    dBidxj[offset + ifill][jdim] *= lagrange_evals[idim][ibasis];
                                }
                            });
                        }
                    },
                    xi
                );
            }
        },
        xi
    );
  }

  /**
   * @brief evaluate the second derivatives of the given basis function 
   * @param [in] xi the point in the reference domain to evaluate at 
   * @param [in] ibasis the index of the basis function 
   * @param [out] Hessian the hessian of the basis function 
   */
  void dshp2(const T *xi, int ibasis, T Hessian[ndim][ndim]) const {
      std::fill_n(Hessian[0], ndim * ndim, 1.0);
      for(int ideriv = 0; ideriv < ndim; ++ideriv){
          for(int jderiv = ideriv; jderiv < ndim; ++jderiv){
              for(int idim = 0; idim < ndim; ++idim){
                  if(ideriv == jderiv){
                      Hessian[ideriv][jderiv] *= POLYNOMIAL::dNlagrange1d<T, Pn>(
                              ijk_poin[ibasis][idim], 2, xi[ideriv]);
                  } else {
                      Hessian[ideriv][jderiv] *= 
                          POLYNOMIAL::dlagrange1d<T, Pn>(
                                  ijk_poin[ibasis][idim], xi[ideriv])
                          * POLYNOMIAL::dlagrange1d<T, Pn>(
                                  ijk_poin[ibasis][idim], xi[jderiv]);
                  }
              }
          }
      }

      // copy symmetric part 
      for(int ideriv = 0; ideriv < ndim; ++ideriv){
          for(int jderiv = 0; jderiv < ideriv; ++jderiv){
              Hessian[ideriv][jderiv] = Hessian[jderiv][ideriv];
          }
      }
  }

  /**
   * @brief transform from the reference domain to the physcial domain
   * T(s): s -> x
   * @param [in] node_coords the coordinates of all the nodes
   * @param [in] node_indices the indices in node_coords that pretain to the
   * element
   * @param [in] xi the position in the refernce domain
   * @param [out] x the position in the physical domain
   */
  void transform(const FE::NodalFEFunction<T, ndim> &node_coords,
                 const IDX *node_indices, const Point &xi, Point &x) const {
    // clear output array
    std::fill_n(&(x[0]), ndim, 0.0);

    // Calculate all of the nodal basis functions at xi
    std::array<T, nnode> Bi;
    fill_shp(xi, Bi.data());

    // multiply node coordinates by basis function evaluations
    for (int inode = 0; inode < nnode; ++inode) {
      for (int idim = 0; idim < ndim; ++idim) {
        const auto &node = node_coords[node_indices[inode]];
        x[idim] += Bi[inode] * node[idim];
      }
    }
  }

  /**
    * @brief get the Jacobian matrix of the transformation
    * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
    * @param [in] node_coords the coordinates of all the nodes
    * @param [in] node_indices the indices in node_coords that pretain to this element in order
    * @param [in] xi the position in the reference domain at which to calculate the Jacobian
    * @return the Jacobian matrix
    */
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
      FE::NodalFEFunction<T, ndim> &node_coords,
      const IDX *node_indices,
      const Point &xi
  ) const {
      using namespace NUMTOOL::TENSOR::FIXED_SIZE;
      // Get a 1D pointer representation of the matrix head
      Tensor<T, ndim, ndim> J;
      J = 0;

      // compute Jacobian per basis function
      Tensor<T, nnode, ndim> dBidxj;
      fill_deriv(xi, dBidxj);

      for(int inode = 0; inode < nnode; ++inode){
          IDX global_inode = node_indices[inode];
          PointView node{node_coords[global_inode]};
          for(int idim = 0; idim < ndim; ++idim){
              for(int jdim = 0; jdim < ndim; ++jdim){
                  // add contribution to jacobian from this basis function
                  J[idim][jdim] += dBidxj[inode][jdim] * node[idim];
              }
          }
      }

      return J;
  }


  /**
    * @brief get the Hessian of the transformation
    * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
    *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
    * @param [in] node_coords the coordinates of all the nodes
    * @param [in] node_indices the indices in node_coords that pretain to this element in order
    * @param [in] xi the position in the reference domain at which to calculate the hessian
    * @param [out] the Hessian in tensor form indexed [k][i][j] as described above
    */
  void Hessian(
      FE::NodalFEFunction<T, ndim> &node_coords,
      const IDX *node_indices,
      const Point &xi,
      T hess[ndim][ndim][ndim]
  ) const {
    // Get a 1D pointer representation
    T *Hptr = hess[0][0];

    // fill with zeros
    std::fill_n(Hptr, ndim * ndim * ndim, 0.0);

    for(int inode = 0; inode < nnode; ++inode){
      // get view to the node coordinates from the node coordinate array
      IDX global_inode = node_indices[inode];
      PointView node{node_coords[global_inode]};

      for(int kdim = 0; kdim < ndim; ++kdim){ // k corresponds to xi 
        T node_hessian[ndim][ndim];
        dshp2(xi, inode, node_hessian);
        for(int idim = 0; idim < ndim; ++idim){
          for(int jdim = idim; jdim < ndim; ++jdim){
            hess[kdim][idim][jdim] += node_hessian[idim][jdim] * node[kdim];
          }
        }
      }
    }

    // finish filling symmetric part
    for(int kdim = 0; kdim < ndim; ++kdim){
      for(int idim = 0; idim < ndim; ++idim){
        for(int jdim = 0; jdim < idim; ++jdim){
          hess[kdim][idim][jdim] = hess[kdim][jdim][idim];
        }
      }
    }
  }

  /** @brief get the number of nodes that define the transformation */
  constexpr int n_nodes() const { return nnode; }

  /** @brief the number of element vertices */
  constexpr int n_vert() const { return nvert; }

  /** @brief the number of vertices on the given face */
  constexpr int n_facevert(int faceNr) { return nfacevert;} 

  /**
    * @brief get a pointer to the array of Lagrange points in the reference domain
    * @return the Lagrange points in the reference domain
    */
  const Point *reference_nodes() const { return xi_poin.data(); }

  /**
   * @brief get the global vertex array from the global node indices 
   * @param [in] nodes_el the global indices of the element nodes 
   * @param [out] vert_el the global indices of the element vertices 
   */
  void get_element_vert(const IDX nodes_el[nnode], IDX vert_el[nvert]){
    int ivert = 0;

    std::function<void(int, int, int)> getvertex = [&](int istart1, int istart2, int idim)-> void {
      if(idim == 2){
        vert_el[ivert++] = nodes_el[istart1];
        vert_el[ivert++] = nodes_el[istart1 + Pn];

        vert_el[ivert++] = nodes_el[istart2];
        vert_el[ivert++] = nodes_el[istart2 + Pn];
      } else {
        const int block = std::pow(Pn + 1, ndim - idim - 1);
        getvertex(istart1, istart1 + Pn * block, idim + 1);
        getvertex(istart2, istart2 + Pn * block, idim + 1);

      }
    };
    getvertex(0, Pn * std::pow(Pn + 1, ndim - 1), 0);
  }

  // ==================
  // = Face Utilities =
  // ==================
  private:
  static constexpr int n_trace = 2 * ndim;
  static constexpr int trace_ndim = ndim - 1;
  /**
   * @brief sometimes the sign of the trace coordinate 
   * needs to be flipped to ensure positive normals 
   * The convention taken is the flip the sign of the first trace dimension 
   * this stores that sign 
   */
  inline static NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, n_trace> first_dim_sign = []{
    NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, n_trace> ret{};

    for(int itrace = 0; itrace < n_trace; ++itrace){
      if constexpr (ndim == 1){
        // 0 dimensional faces would lead to indexing issues
        // set to 1 and break
        ret[itrace] = 1.0;
        break;
      }

      int trace_coord = itrace % ndim;
      // true if this is a negative face
      bool is_negative_xi = itrace / ndim == 0;

      // build the indices for the levi_civita tensor 
      // first the trace coord (normal direction)
      // then the unit basis vector indices in lexicographic order
      std::size_t lc_indices[ndim];
      lc_indices[0] = trace_coord;
      for(int idim = 0; idim < trace_coord; ++idim)
        { lc_indices[idim + 1] = idim; }
      for(int idim = trace_coord; idim < trace_ndim; ++idim )
        { lc_indices[idim + 1] = idim + 1; }

      // get the levi_civita tensor, then chose sign based on normal direction 
      T lc = NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<T, ndim>.list_index(lc_indices);
      if(is_negative_xi) lc = -lc;

      ret[itrace] = lc;
    }

    return ret;
  }();

  public:

  /** @brief get the number of nodes on a given face */
  int n_nodes_face(int faceNr){ return nfacenode; }

  /**
   * @brief get the global node indices of the nodes in a given face 
   * in order such that a ndim-1 dimensional element transformation 
   * matches the TraceTransformation 
   *
   * NOTE: This is done by taking the sign flip for the first trace coordinate 
   * and determining an anchor vertex that would make the ndim-1 element 
   * transformation remain internal
   */
  void get_face_nodes(
      int faceNr,
      const IDX nodes_el[nnode],
      IDX face_nodes[nfacenode]
  ) const {
    int trace_coord = faceNr % ndim;
    // true if this is a negative face
    bool is_negative_xi = faceNr / ndim == 0;

    // the first dimension of the trace
    int trace_first_dim = (trace_coord == 0) ? 1 : 0;
    
    // the node that corresponds to the origin of the TraceTransformation
    IDX anchor_vertex = (first_dim_sign[faceNr] < 0) ? (Pn) * strides[trace_first_dim]: 0;

    IDX face_node_idx = 0;
   
    // function to fill the nodes recursively
    std::function<void(int, int)> fill_nodes = [&](
        int el_node_idx, int idim
    ) {
      int last_dim = (trace_coord == ndim - 1) ? ndim - 2 : ndim - 1;
      if(idim == last_dim){
        for(int j = 0; j < Pn + 1; ++j){
          // base case: assign the global node
          int lidx = el_node_idx + j * strides[last_dim];
          face_nodes[face_node_idx++] = nodes_el[lidx];
        }
      } else {
        if(idim == trace_coord) {
          // pass through 
          fill_nodes(el_node_idx, idim + 1);
        } else {
          for(int j = 0; j < Pn + 1; ++j){
            // move by the stride and recurse
            fill_nodes(el_node_idx + j * strides[idim], idim + 1);
          }
        }
      }
    };

    // handle the first dimension seperately because of possible direction difference
    int idim = trace_first_dim;
    for(int j = 0; j < Pn + 1; ++j){
      // travel in the direction of the first_dim_sign
      int jstride = j * strides[trace_first_dim] * first_dim_sign[faceNr];
      fill_nodes(anchor_vertex + jstride, trace_first_dim + 1);
    }
  }

  /**
   * @brief get the global vertex array corresponding to the given face number 
   * WARNING: this does not take orientation into account
   *
   * @param [in] faceNr the face number 
   * @param [in] nodes_el the global element node array 
   * @param [out] vert_fac the global face vertex array 
   */
  void get_face_vert(
    int faceNr,
    const IDX nodes_el[nnode],
    IDX vert_fac[nfacevert]
  ){
    int face_coord = faceNr % ndim;

    bool is_negative_xi = (faceNr / ndim == 0);
   
    auto next_ijk = [&](int ijk[ndim]){
      for(int idim = ndim - 1; idim >= 0; --idim) if(idim != face_coord)
      {
        if(ijk[idim] == 0){
          ijk[idim] = Pn;
          break; // don't have to continue to next dimension
        } else {
          ijk[idim] = 0;
          // carry over to next dimension (don't break)
        }
      }
    };

    int ijk[ndim] = {0};
    ijk[face_coord] = (is_negative_xi) ? 0 : Pn;
    vert_fac[0] = convert_indices_helper(ijk);
    for(int i = 1; i < nfacevert; ++i){
      next_ijk(ijk);
      vert_fac[i] = convert_indices_helper(ijk);
    }
  }

  /**
   * @brief get the face number from the global indices of element and face vertices 
   * @param [in] nodes_el the element node indices
   * @param [in] vert_jfac the face vertices to find the face number for
   * NOTE: vertices, not nodes 
   *
   * @return the face number or -1 if not found
   */
  int get_face_nr(const IDX nodes_el[nnode], const IDX *vert_jfac){
   
    for(int ifac = 0; ifac < nfac; ++ifac){
      bool all_found = true; // assume true until mismatch
      for(int ivert = 0; ivert < nfacevert; ++ivert){
        IDX vert_ifac[nfacevert];
        get_face_vert(ifac, nodes_el, vert_ifac);
        bool found = false;
        // try to find ivert
        for(int jvert = 0; jvert < nfacevert; ++jvert){
          if(vert_ifac[ivert] == vert_jfac[jvert]){
            found = true;
          }
        }

        if(!found){
          all_found = false;
          continue;
        }
      }

      if(all_found) return ifac;
    }

    return -1;
  }

  /** @brief print the 1d lagrange basis function indices for each dimension for
   * each node */
  std::string print_ijk_poin() {
    using namespace std;
    std::ostringstream ijk_string;
    for (int inode = 0; inode < nnode; ++inode) {
      ijk_string << "[";
      for (int idim = 0; idim < ndim; ++idim) {
        ijk_string << " " << ijk_poin[inode][idim];
      }
      ijk_string << " ]\n";
    }
    return ijk_string.str();
  }
};


/**
  * @brief transformation from the global reference trace space to the 
  *        reference trace space of the right element
  *        (the left element trace is defined to be equivalent to the global orientation)
  * @tparam T the floating point type
  * @tparam IDX the indexing type for large lists
  * @tparam ndim the number of dimensions (for the element)
  */
template<typename T, typename IDX, int ndim>
class HypercubeTraceOrientTransformation {
  /// number of dimensions for the trace space 
  static constexpr int trace_ndim = ndim - 1;
  /// number of vertices in a face
  static constexpr int nvert_tr = MATH::power_T<2, ndim>::value;

  /// upper bound (not inclusive) of the sign codes
  static constexpr int sign_code_bound = MATH::power_T<2, trace_ndim>::value;
 
  using TracePointView = MATH::GEOMETRY::PointView<T, trace_ndim>;

  private:

  template<int idim> inline int idim_mask(){ return (1 << idim); }

  /**
   * @brief copy the sign at the specified dimension 
   * @param [in] idim the dimension index (the bit index into sign_code)
   * @param [in] mag the magnitude to apply the sign to 
   * @param [in] sign code a set of bits
   *             (if sign_code at bit idim == 0 then that dimension is positive)
   *             (if sign_code at bit idim == 1 then that dimension is negative)
   */
  inline constexpr int copysign_idim(int idim, int mag, int sign_code){

    int is_negative = sign_code & (1 << idim);
    // TODO: remove branch
    return (is_negative) ? std::copysign(mag, -1) : std::copysign(mag, 1);
  }

  std::vector<std::vector<int>> axis_permutations;


  public:

  HypercubeTraceOrientTransformation(){
    std::vector<int> first_perm{};
    for(int i = 0; i < trace_ndim; ++i) first_perm.push_back(i);

    axis_permutations.push_back(first_perm);
    axis_permutations.push_back(first_perm);
    int iperm = 1;
    while(std::next_permutation(axis_permutations[iperm].begin(), axis_permutations[iperm].end())){
        // make a copy of this permutation to get the next one
        axis_permutations.push_back(axis_permutations[iperm]);
        ++iperm;
    }
  }
  
  /** 
    * @brief get the orientation of the right element given 
    * the vertices for the left and right element 
    * 
    * NOTE: the vertices are in the same order as they get generated
    *       in the element transformataion
    * NOTE: the orientation for the left element is always 0 
    * NOTE: the permutation index can be found by integer division by sign_code_bound
    * NOTE: the sign code is the modulus with sign_code_bound
    *
    * @param verticesL the vertices for the left element (not all nodes)
    * @param verticesR the vertices for the right element 
    * @return the orientation for the right element or -1 if the nodes don't match
    */
  int getOrientation(
      IDX verticesL[nvert_tr],
      IDX verticesR[nvert_tr]
  ) const {
    // binary represinting sign of each dimension 
    // \in [0, 2^trace_ndim)
    int sign_code = 0;
    // the direction in the left element that each axis corresponds to 
    std::vector<int> left_directions(trace_ndim);

    // loop over the axes for the right element 
    int gidx0 = verticesR[0]; // always start from origin
    NUMTOOL::TMP::constexpr_for_range<0, trace_ndim>([&]<int idim> {

      int gidx1 = verticesR[MATH::power_T<2, idim>::value];

      // find the local indices on verticesL that have the same global indices 
      int lidx0 = -1, lidx1 = -1; 

      for(int ivert = 0; ivert < nvert_tr; ++ivert){
        if(verticesL[ivert] == gidx0) lidx0 = ivert;
        else if(verticesL[ivert] == gidx1) lidx1 = ivert;

        // short circuit if both found 
        if(lidx0 >= 0 && lidx1 >= 0) break;
      }

      // check if not found 
      if(lidx0 < 0 || lidx1 < 0) return -1;

      // set the sign code if flipped 
      // (positive direction always has increasing indices)
      if(lidx1 < lidx0) sign_code |= idim_mask<idim>();

      // get the dimensional index 
      // through indexing tomfoolery
      left_directions[idim] = trace_ndim - MATH::log2(std::abs(lidx1 - lidx0)) - 1;
    });

    // get the permutation index in the axis directions
    // with a binary search
    auto lower = std::lower_bound(axis_permutations.begin(), axis_permutations.end(), left_directions);
    int iperm = std::distance(axis_permutations.begin(), lower);

    // generate an orientation from the sign code and left directions 
    return iperm * sign_code_bound + sign_code;
  }

  
  /**
    * @brief transform from the reference trace space
    *  to the right reference trace space
    *
    *  The reference trace space oriented with the left element space is 
    *  defined to be the same orientation as the general reference trace space
    *  so only the right element orientation needs to be considered
    *
    * @param [in] orientationR the orientation of the right element
    * @param [in] s the position in the reference trace space
    * @param [out] the posotion in the local reference trace space for the right element
    */
  void transform(
      int orientationR,
      const TracePointView &s,
      TracePointView sR
  ) const {

    int iperm = orientationR / sign_code_bound;
    int sign_code = orientationR % sign_code_bound;

    for(int idim = 0; idim < trace_ndim; ++idim){
      sR[idim] = copysign_idim(
          idim, 
          s[axis_permutations[iperm][idim]],
          sign_code
      );
    }
  }
};

template<typename T, typename IDX, int ndim, int Pn>
class HypercubeTraceTransformation {

  static constexpr int trace_ndim = (ndim - 1 < 0) ? 0 : ndim - 1;
  static constexpr int n_trace = ndim * 2;
  using TracePointView = MATH::GEOMETRY::PointView<T, trace_ndim>;
  using ElPointView = MATH::GEOMETRY::PointView<T, ndim>;

  inline static HypercubeElementTransformation<T, IDX, trace_ndim, Pn> trace_domain_trans{};

  /**
   * @brief sometimes the sign of the trace coordinate 
   * needs to be flipped to ensure positive normals 
   * The convention taken is the flip the sign of the first trace dimension 
   * this stores that sign 
   */
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, n_trace> first_dim_sign;

  public: 

  HypercubeTraceTransformation(){
    for(int itrace = 0; itrace < n_trace; ++itrace){
      if constexpr (ndim == 1){
        // 0 dimensional faces would lead to indexing issues
        // set to 1 and break
        first_dim_sign[itrace] = 1.0;
        break;
      }

      int trace_coord = itrace % ndim;
      // true if this is a negative face
      bool is_negative_xi = itrace / ndim == 0;

      // build the indices for the levi_civita tensor 
      // first the trace coord (normal direction)
      // then the unit basis vector indices in lexicographic order
      std::size_t lc_indices[ndim];
      lc_indices[0] = trace_coord;
      for(int idim = 0; idim < trace_coord; ++idim)
        { lc_indices[idim + 1] = idim; }
      for(int idim = trace_coord; idim < trace_ndim; ++idim )
        { lc_indices[idim + 1] = idim + 1; }

      // get the levi_civita tensor, then chose sign based on normal direction 
      T lc = NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<T, ndim>.list_index(lc_indices);
      if(is_negative_xi) lc = -lc;

      first_dim_sign[itrace] = lc;
    }
  }

  /**
   * @brief transform from the trace space reference domain to the 
   * reference element domain 
   *
   * @param [in] node_indices the global node indices for the element 
   * @param [in] faceNr the trace number 
   * @param [in] s the location in the reference trace domain 
   * @param [out] xi the position in the reference element domain 
   */
  void transform(
    IDX *node_indices,
    int traceNr,
    const TracePointView &s,
    ElPointView xi
  ) const {

    int trace_coord = traceNr % ndim;
    // true if this face is on the negative xi side for trace_coord
    bool is_negative_xi = traceNr / ndim  == 0;
    xi[trace_coord] = is_negative_xi ? -1.0 : 1.0;

    // copy over the other coordinates
    for(int idim = 0; idim < trace_coord; ++idim)
      { xi[idim] = s[idim]; }
    for(int idim = trace_coord + 1; idim < ndim; ++idim )
      { xi[idim] = s[idim - 1]; }

    // correct sign to ensure outward normal 
    if(trace_coord == 0) xi[1] *= first_dim_sign[traceNr];
    else xi[0] *= first_dim_sign[traceNr];
  }

  /**
   * @brief get the Jacobian dx ds 
   * @param coord the global node coordinates 
   * @param face_node_indices the global node indices of the nodes on the face 
   * in order so that the orientation matches the transform function 
   * @param traceNr the face number 
   * @param s the point in the reference trace space 
   */
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, trace_ndim> Jacobian(
      FE::NodalFEFunction<T, ndim> &coord,
      IDX *face_node_indices,
      int traceNr,
      const MATH::GEOMETRY::Point<T, trace_ndim> &s
  ) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // initialize the jacobian
    Tensor<T, ndim, trace_ndim> J;
    J = 0;

    // Get the gradient per basis function
    Tensor<T, trace_domain_trans.nnode, trace_ndim> dBidxj;
    trace_domain_trans.fill_deriv(s, dBidxj);

    // add contributions from each node 
    for(int inode = 0; inode < trace_domain_trans.nnode; ++inode){
        IDX global_inode = face_node_indices[inode];
        ElPointView node{coord[global_inode]};
        for(int idim = 0; idim < ndim; ++idim){
            for(int jdim = 0; jdim < trace_ndim; ++jdim){
                // add contribution to jacobian from this basis function
                J[idim][jdim] += dBidxj[inode][jdim] * node[idim];
            }
        }
    }

    return J;
  }

  /**
   * @brief Given the Jacobian dx dxi 
   * get the Jacobian dx ds 
   * @param node_indices the global node index array 
   * @param traceNr the face number 
   * @param s the point in the reference trace space 
   * @param elJacobian dx dxi 
   * @return dx ds Jacobian 
   */
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, trace_ndim> Jacobian(
      IDX *node_indices,
      int traceNr,
      const TracePointView &s,
      const NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> &elJacobian
  ) const {
    // TODO: column major storage would make permutations quicker

    NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, trace_ndim> J;
    J = 0;

    // basically apply the transformation to the jacobian columns
    int trace_coord = traceNr % ndim;
    bool is_negative_xi = traceNr / ndim  == 0;
    for(int jdim = 0; jdim < trace_coord; ++jdim){
      for(int idim = 0; idim < ndim; ++idim)
        { J[idim][jdim] = elJacobian[idim][jdim]; }
    }

    for(int jdim = trace_coord + 1; jdim < ndim; ++jdim){
      for(int idim = 0; idim < ndim; ++idim)
        { J[idim][jdim - 1] = elJacobian[idim][jdim]; }
    }

    // correct sign with levi_civita
    // if negative xi, flip by negative levi_civita 
    if constexpr(ndim > 1){
      std::size_t lc_indices[ndim];
      lc_indices[0] = trace_coord;
      for(int idim = 0; idim < trace_ndim; ++idim){
        if(idim < trace_coord){
          lc_indices[idim + 1] = idim;
        } else {
          lc_indices[idim + 1] = idim + 1;
        }
      }

      T lc = NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<T, ndim>.list_index(lc_indices);
      if(is_negative_xi) lc = -lc;
      for(int idim = 0; idim < ndim; ++idim){
      // NOTE: this matches up to the first coord 
      // corrected in transformation because it is already sliced out
        J[idim][0] *= lc; 
      }
    }
    return J;
  }
};

} // namespace ELEMENT::TRANSFORMATIONS
