#pragma once
#include <Numtool/constexpr_math.hpp>
#include <Numtool/integer_utils.hpp>
#include <Numtool/polydefs/LagrangePoly.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <Numtool/matrix/dense_matrix.hpp>
#include <array>
#include <iceicle/fe_function/nodal_fe_function.hpp>

#include <algorithm>
#include <sstream>
#include <string>

namespace ELEMENT::TRANSFORMATIONS {
/**
 * @brief Transformations from the [-1, 1] hypercube
 * to an arbitrary order hypercube element
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
  static constexpr int nfacept = MATH::power_T<2, ndim - 1>::value;
  static constexpr int nnode = MATH::power_T<Pn + 1, ndim>::value;

public:
  /// Basis function indices by dimension for each node
  static constexpr std::array<std::array<int, ndim>, nnode> ijk_poin = []() {
    std::array<std::array<int, ndim>, nnode> ret{};

    NUMTOOL::TMP::constexpr_for_range<0, ndim>([&ret]<int idim>() {
      // number of times to repeat the loop over basis functions
      const int nrepeat = MATH::power_T<Pn + 1, idim>::value;
      // the size that one loop through the basis function indices gives
      const int blocksize = MATH::power_T<Pn + 1, ndim - idim>::value;
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            [irep, &ret]<int ibasis>() {
              const int nfill = NUMTOOL::TMP::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                const int offset = irep * blocksize + start_offset;
                ret[offset + ifill][idim] = ibasis;
              }
            });
      }
    });

    return ret;
  }();

  /// Nodes in the reference domain
  static inline std::array<Point, nnode> xi_poin = [] {
    // Generate the vertices
    std::array<Point, nnode> ret{};

    NUMTOOL::TMP::constexpr_for_range<0, ndim>([&ret]<int idim>() {
      // number of times to repeat the loop over basis functions
      const int nrepeat = MATH::power_T<Pn + 1, idim>::value;
      // the size that one loop through the basis function indices gives
      const int blocksize = MATH::power_T<Pn + 1, ndim - idim>::value;
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            [irep, &ret]<int ibasis>() {
              const T dx = 2.0 / (Pn);

              const int nfill = NUMTOOL::TMP::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                const int offset = irep * blocksize + start_offset;
                ret[offset + ifill][idim] = -1.0 + dx * ibasis;
              }
            });
      }
    });

    return ret;

  }();


  /**
   * @brief fill the array with shape functions at the given point
   * @param [in] xi the point in the reference domain to evaluate the basis at
   * @param [out] Bi the shape function evaluations
   */
  void fill_shp(const Point &xi, T *Bi) const {
    NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
        [&Bi]<int ibasis>(T xi_dim) {
          static constexpr int nfill = MATH::power_T<Pn + 1, ndim - 1>::value;
          T Bi_idim = POLYNOMIAL::lagrange1d<T, Pn, ibasis>(xi_dim);
          std::fill_n(Bi + nfill * ibasis, nfill, Bi_idim);
        },
        xi[0]);

    for (int idim = 1; idim < ndim; ++idim) {
      T xi_dim = xi[idim];

      // number of times to repeat the loop over basis functions
      int nrepeat = std::pow(Pn + 1, idim);
      // the size that one loop through the basis function indices gives
      const int blocksize = std::pow(Pn + 1, ndim - idim);
      for (int irep = 0; irep < nrepeat; ++irep) {
        NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
            []<int ibasis>(int idim, T xi_dim, T *Bi) {
              // evaluate the 1d basis function at the idimth coordinate
              T Bi_idim = POLYNOMIAL::lagrange1d<T, Pn, ibasis>(xi_dim);
              const int nfill = std::pow(Pn + 1, ndim - idim - 1);

              // offset for multiplying by this ibasis
              const int start_offset = ibasis * nfill;

              // multiply the next nfill by the current basis function
              for (int ifill = 0; ifill < nfill; ++ifill) {
                Bi[start_offset + ifill] *= Bi_idim;
              }
            },
            idim, xi_dim, Bi + irep * blocksize);
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
  void fill_deriv(const Point &xi, T **dBidxj) const {

      // fencepost the loop at idim = 0
      NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
        [&dBidxj]<int ibasis>(const Point &xi) {
          static constexpr int nfill = MATH::power_T<Pn + 1, ndim - 1>::value;
          T Bi_idim = POLYNOMIAL::lagrange1d<T, Pn, ibasis>(xi[0]);
          T dBi_idim = POLYNOMIAL::dlagrange1d<T, Pn, ibasis>(xi[0]);
          for(int ifill = 0; ifill < nfill; ++ifill){
              dBidxj[nfill * ibasis + ifill][0] = dBi_idim;
              for(int jdim = 1; jdim < ndim; ++jdim){
                  dBidxj[nfill * ibasis + ifill][jdim] = Bi_idim;
              }
          }
        },
        xi[0]);
      
      NUMTOOL::TMP::constexpr_for_range<1, ndim>(
          [&dBidxj]<int idim>(const Point &xi){
              // number of times to repeat the loop over basis functions
              const int nrepeat = std::pow(Pn + 1, idim);
              // the size that one loop through the basis function indices gives 
              const int blocksize = std::pow(Pn + 1, ndim - idim);

              for(int irep = 0; irep < nrepeat; ++irep) {

                  NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
                      [&, irep]<int ibasis>(const Point &xi) {
                          T dBi_idim = POLYNOMIAL::dlagrange1d<T, Pn, ibasis>(xi[idim]);
                          const int nfill = std::pow(Pn + 1, ndim - idim - 1);

                          // offset for multiplying by this ibasis
                          const int start_offset = ibasis * nfill;

                          // multiply the next nfill by the current basis function
                          for (int ifill = 0; ifill < nfill; ++ifill) {
                              const int offset = irep * blocksize + start_offset;
                              NUMTOOL::TMP::constexpr_for_range<0, ndim>([&]<int jdim>(){
                                  if constexpr(jdim == idim){
                                      dBidxj[offset + ifill][jdim] *= dBi_idim;
                                  } else {
                                      dBidxj[offset + ifill][jdim] *= POLYNOMIAL::lagrange1d<T, Pn, ibasis>(xi[idim]);
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
    * @param [out] the jacobian matrix
    */
  void Jacobian(
      FE::NodalFEFunction<T, ndim> &node_coords,
      const IDX *node_indices,
      const Point &xi,
      T J[ndim][ndim]
  ) const {
      // Get a 1D pointer representation of the matrix head
      T *Jptr = J[0];

      // fill with zeros
      std::fill_n(Jptr, ndim * ndim, 0.0);

      // compute Jacobian per basis function
      MATH::MATRIX::DenseMatrixSetWidth<T, ndim> dBidxj(nnode);
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
  constexpr int n_nodes() { return nnode; }

  /**
    * @brief get a pointer to the array of Lagrange points in the reference domain
    * @return the Lagrange points in the reference domain
    */
  const Point *reference_nodes() const { return xi_poin.data(); }

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

} // namespace ELEMENT::TRANSFORMATIONS
