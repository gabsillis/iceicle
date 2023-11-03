#pragma once
#include <Numtool/integer_utils.hpp>
#include <Numtool/polydefs/LagrangePoly.hpp>
#include <Numtool/tmp_flow_control.hpp>
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
  static constexpr std::array<std::array<int, ndim>, nnode> ijk_poin = [](){
    std::array<std::array<int, ndim>, nnode> ret{};

    NUMTOOL::TMP::constexpr_for_range<0, ndim>( [&ret]<int idim>(){
      // number of times to repeat the loop over basis functions
      const int nrepeat = MATH::power_T<Pn + 1, idim>::value;
      // the size that one loop through the basis function indices gives
      const int blocksize = MATH::power_T<Pn + 1, ndim - idim>::value;
      for(int irep = 0; irep < nrepeat; ++irep){
        NUMTOOL::TMP::constexpr_for_range<0, Pn+1>([irep, &ret]<int ibasis>(){
          const int nfill = std::pow(Pn + 1, ndim - idim - 1);

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
  static constexpr std::array<T, (Pn + 1) *ndim> reference_nodes = [] {
    // Generate the vertices
  }();

  /**
   * @brief fill the array with shape functions at the given point
   * @param [in] xi the point in the reference domain to evaluate the basis at
   * @param [out] Bi the shape function evaluations
   */
  void fill_shp(const Point &xi, std::array<T, nnode> &Bi) {
    NUMTOOL::TMP::constexpr_for_range<0, Pn + 1>(
        [&Bi]<int ibasis>(T xi_dim) {
          static constexpr int nfill = MATH::power_T<Pn + 1, ndim - 1>::value;
          T Bi_idim = POLYNOMIAL::lagrange1d<T, Pn, ibasis>(xi_dim);
          std::fill_n(Bi.begin() + nfill * ibasis, nfill, Bi_idim);
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
            idim, xi_dim, Bi.begin() + irep * blocksize);
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
                 const IDX *node_indices, const Point &xi, Point &x) {
    // clear output array
    std::fill_n(&(x[0]), ndim, 0.0);

    // Calculate all of the nodal basis functions at xi
    std::array<T, nnode> Bi;
    fill_shp(xi, Bi);

    // multiply node coordinates by basis function evaluations
    for (int inode = 0; inode < nnode; ++inode) {
      for (int idim = 0; idim < ndim; ++idim) {
        const auto &node = node_coords[node_indices[inode]];
        x[idim] += Bi[inode] * node[idim];
      }
    }
  }

  /** @brief get the number of nodes that define the transformation */
  constexpr int n_nodes() {
    return nnode;
  }

  /** @brief print the 1d lagrange basis function indices for each dimension for each node */
  std::string print_ijk_poin(){
    using namespace std;
    std::ostringstream ijk_string;
    for(int inode = 0; inode < nnode; ++inode){
      ijk_string << "[";
      for(int idim = 0; idim < ndim; ++idim){
        ijk_string << " " << ijk_poin[inode][idim];
      }
      ijk_string << " ]\n";
    }
    return ijk_string.str();
  }
};

} // namespace ELEMENT::TRANSFORMATIONS
