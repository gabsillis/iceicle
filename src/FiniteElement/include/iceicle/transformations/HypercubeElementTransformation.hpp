#pragma once
#include "Numtool/point.hpp"
#include "iceicle/basis/lagrange_1d.hpp"
#include "iceicle/basis/tensor_product.hpp"
#include <Numtool/constexpr_math.hpp>
#include <Numtool/integer_utils.hpp>
#include <Numtool/polydefs/LagrangePoly.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <Numtool/matrix/dense_matrix.hpp>
#include <Numtool/fixed_size_tensor.hpp>
#include <array>
#include <cmath>
#include <iceicle/fe_function/nodal_fe_function.hpp>
#include <mdspan/mdspan.hpp>
#include <algorithm>

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

  // === Nodal Basis ===
public:
  BASIS::UniformLagrangeInterpolation<T, Pn> interpolation_1d{};
  using BasisType = decltype(interpolation_1d);
  BASIS::QTypeProduct<T, ndim, BasisType::nbasis> tensor_prod{};
  using TensorProdType = decltype(tensor_prod);

private:
  // === Constants ===
  static constexpr int nfac = 2 * ndim;
  static constexpr int nnode = TensorProdType::nvalues;
  static constexpr int nvert = MATH::power_T<2, ndim>::value;
  static constexpr int nfacevert = MATH::power_T<2, ndim - 1>::value;
  static constexpr int nfacenode = MATH::power_T<Pn + 1, ndim -1>::value;
  static constexpr NUMTOOL::TENSOR::FIXED_SIZE::Tensor<int, ndim> strides = TensorProdType::strides; 

  // ==== Friends ===
  // this is the domain for a face in d+1 dimensions
  friend class HypercubeTraceTransformation<T, IDX, ndim + 1, Pn>;

public:


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
    tensor_prod.fill_shp(interpolation_1d, xi, Bi.data());

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
      const FE::NodalFEFunction<T, ndim> &node_coords,
      const IDX *node_indices,
      const Point &xi
  ) const {
      using namespace NUMTOOL::TENSOR::FIXED_SIZE;
      // Get a 1D pointer representation of the matrix head
      Tensor<T, ndim, ndim> J;
      J = 0;

      // compute Jacobian per basis function
      Tensor<T, nnode, ndim> dBidxj;
      tensor_prod.fill_deriv(interpolation_1d, xi, dBidxj);

      for(int inode = 0; inode < nnode; ++inode){
          IDX global_inode = node_indices[inode];
          const auto &node = node_coords[global_inode];
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
    * @return the Hessian in tensor form indexed [k][i][j] as described above
    */
  NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim> Hessian(
      const FE::NodalFEFunction<T, ndim> &node_coords,
      const IDX *node_indices,
      const Point &xi
  ) const noexcept {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    Tensor<T, ndim, ndim, ndim> hess;
    // Zero initialize
    hess = 0;

    // Get the hessian at each node 
    std::vector<T> nodal_hessian_data(nnode * ndim * ndim);
    auto nodal_hessian = tensor_prod.fill_hess(interpolation_1d, xi, nodal_hessian_data.data());

    for(int inode = 0; inode < nnode; ++inode){
      // get view to the node coordinates from the node coordinate array
      IDX global_inode = node_indices[inode];
      const auto & node = node_coords[global_inode];

      for(int kdim = 0; kdim < ndim; ++kdim){ // k corresponds to xi 
        for(int idim = 0; idim < ndim; ++idim){
          for(int jdim = idim; jdim < ndim; ++jdim){
            hess[kdim][idim][jdim] += nodal_hessian[inode, idim, jdim] * node[kdim];
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

    return hess;
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

  // ====================
  // = Domain Utilities =
  // ====================

  private:
  auto get_vertex_helper(const IDX nodes_el[nnode], IDX vert_el[nvert], int istart1, int istart2, int idim, int ivert) -> int {
      if(idim == 2){
        vert_el[ivert++] = nodes_el[istart1];
        vert_el[ivert++] = nodes_el[istart1 + Pn];

        vert_el[ivert++] = nodes_el[istart2];
        vert_el[ivert++] = nodes_el[istart2 + Pn];
        return ivert;
      } else {
        const int block = std::pow(Pn + 1, ndim - idim - 1);
        ivert = get_vertex_helper(nodes_el, vert_el, istart1, istart1 + Pn * block, idim + 1, ivert);
        ivert = get_vertex_helper(nodes_el, vert_el, istart2, istart2 + Pn * block, idim + 1, ivert);
        return ivert;
      }
  }

  public:
  /**
   * @brief get the global vertex array from the global node indices 
   * @param [in] nodes_el the global indices of the element nodes 
   * @param [out] vert_el the global indices of the element vertices 
   */
  void get_element_vert(const IDX nodes_el[nnode], IDX vert_el[nvert]){
    (void) get_vertex_helper(nodes_el, vert_el, 0, Pn * std::pow(Pn + 1, ndim - 1), 0, 0);
  }

  /** @brief rotates the node indices 
   * 90 degrees ccw about the x axis 
   * WARNING: only defined for 3D 
   */
  void rotate_x(IDX gnodes[nnode]) requires (ndim == 3) {

    IDX gnodes_old[nnode];
    std::copy_n(gnodes, nnode, gnodes_old);

    for(int i = 0; i < Pn + 1; ++i){
      for(int j = 0; j < Pn + 1; ++j){
        for(int k =0; k < Pn + 1; ++k){
          int ijk_old[ndim] = {i, j, k};
          int ijk_new[ndim] = {i, Pn - k, j};
          gnodes[TensorProdType::convert_ijk(ijk_new)] = 
            gnodes_old[TensorProdType::convert_ijk(ijk_old)];
        }
      }
    }
  }

  /** @brief rotates the node indices 
   * 90 degrees ccw about the y axis 
   * WARNING: only defined for 3D 
   */
  void rotate_y(IDX gnodes[nnode]) requires (ndim == 3) {

    IDX gnodes_old[nnode];
    std::copy_n(gnodes, nnode, gnodes_old);

    for(int i = 0; i < Pn + 1; ++i){
      for(int j = 0; j < Pn + 1; ++j){
        for(int k = 0; k < Pn + 1; ++k){
          int ijk_old[ndim] = {i, j, k};
          int ijk_new[ndim] = {k, j, Pn - i};
          gnodes[TensorProdType::convert_ijk(ijk_new)] = 
            gnodes_old[TensorProdType::convert_ijk(ijk_old)];
        }
      }
    }
  }

  /** @brief rotates the node indices 
   * 90 degrees ccw about the z axis 
   * WARNING: only defined for 3D 
   */
  void rotate_z(IDX gnodes[nnode]) requires(ndim == 3) {

    IDX gnodes_old[nnode];
    std::copy_n(gnodes, nnode, gnodes_old);

    for(int i = 0; i < Pn + 1; ++i){
      for(int j = 0; j < Pn + 1; ++j){
        for(int k =0; k < Pn + 1; ++k){
          int ijk_old[ndim] = {i, j, k};
          int ijk_new[ndim] = {Pn - j, i, k};
          gnodes[TensorProdType::convert_ijk(ijk_new)] = 
            gnodes_old[TensorProdType::convert_ijk(ijk_old)];
        }
      }
    }
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
    // IDX anchor_vertex = (first_dim_sign[faceNr] < 0) ? (Pn) * strides[trace_first_dim]: 0;

    IDX face_node_idx = 0;

    auto next_ijk = [&](int ijk[ndim]){
      for(int idim = ndim - 1; idim >= 0; --idim) if(idim != trace_coord) {
        if(idim == trace_first_dim && first_dim_sign[faceNr] < 0){
          // reversed order
          if(ijk[idim] == 0){
            ijk[idim] = Pn;
          } else {
            ijk[idim]--;
            return true;
          }
        } else {
          
          if(ijk[idim] == Pn){
            ijk[idim] = 0;
          } else {
            ijk[idim]++;
            return true;
          }
        }
      }
      return false;
    };

    int ijk[ndim] = {0};
    if(first_dim_sign[faceNr] < 0){
      // anchor point needs to be at Pn 
      ijk[trace_first_dim] = Pn;
    }
    // move to the correct face between positive and negative side 
    ijk[trace_coord] = (is_negative_xi) ? 0 : Pn;

    int inode = 0;
    do {
      face_nodes[inode] = nodes_el[TensorProdType::convert_ijk(ijk)];
      inode++;
    } while(next_ijk(ijk));
  }

  /**
   * @brief get the global vertex array corresponding to the given face number 
   * in order such that a ndim-1 dimensional element transformation 
   * matches the TraceTransformation 
   *
   * NOTE: This is done by taking the sign flip for the first trace coordinate 
   * and determining an anchor vertex that would make the ndim-1 element 
   * transformation remain internal
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

    // the first dimension of the trace
    int trace_first_dim = (face_coord == 0) ? 1 : 0;
   
    auto next_ijk = [&](int ijk[ndim]){
      for(int idim = ndim - 1; idim >= 0; --idim) if(idim != face_coord)
      {
        if(idim == trace_first_dim && first_dim_sign[faceNr] < 0){
          // reversed order 
          if(ijk[idim] == Pn){
            ijk[idim] = 0;
            break; // don't have to continue to next dimension
          } else {
            ijk[idim] = Pn;
          }
        } else {
          // normal order
          if(ijk[idim] == 0){
            ijk[idim] = Pn;
            break; // don't have to continue to next dimension
          } else {
            ijk[idim] = 0;
            // carry over to next dimension (don't break)
          }
        }
      }
    };

    int ijk[ndim] = {0};
    if(first_dim_sign[faceNr] < 0){
      // anchor point needs to be at Pn 
      ijk[trace_first_dim] = Pn;
    }
    ijk[face_coord] = (is_negative_xi) ? 0 : Pn;
    vert_fac[0] = nodes_el[TensorProdType::convert_ijk(ijk)];
    for(int i = 1; i < nfacevert; ++i){
      next_ijk(ijk);
      vert_fac[i] = nodes_el[TensorProdType::convert_ijk(ijk)];
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
      IDX vert_ifac[nfacevert];
      get_face_vert(ifac, nodes_el, vert_ifac);
      for(int ivert = 0; ivert < nfacevert; ++ivert){
        bool found = false;
        // try to find ivert
        for(int jvert = 0; jvert < nfacevert; ++jvert){
          if(vert_ifac[ivert] == vert_jfac[jvert]){
            found = true;
            break;
          }
        }

        if(!found){
          all_found = false;
          break;
        }
      }

      if(all_found) return ifac;
    }

    return -1;
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
  static constexpr int nvert_tr = MATH::power_T<2, trace_ndim>::value;

  /// upper bound (not inclusive) of the sign codes
  static constexpr int sign_code_bound = MATH::power_T<2, trace_ndim>::value;
 
  using TracePointView = MATH::GEOMETRY::PointView<T, trace_ndim>;

  private:

  template<int idim> inline int idim_mask() const { return (1 << idim); }

  /**
   * @brief multiply by the sign at the given dimension
   * @param [in] idim the dimension index (the bit index into sign_code)
   * @param [in] mag the magnitude to apply the sign to 
   * @param [in] sign code a set of bits
   *             (if sign_code at bit idim == 0 then that dimension is positive)
   *             (if sign_code at bit idim == 1 then that dimension is negative)
   */
  inline constexpr T copysign_idim(int idim, T mag, int sign_code) const {

    int is_negative = sign_code & (1 << idim);
    // TODO: remove branch
    return (is_negative) ? mag * -1 : mag;
  }

  std::vector<std::vector<int>> axis_permutations;


  public:

  HypercubeTraceOrientTransformation(){
    std::vector<int> first_perm{};
    for(int i = 0; i < trace_ndim; ++i) first_perm.push_back(i);

    // need both of these for std::next_permutation
    // the second one (iperm=1) gets immediately permuted
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
    int errcode = 0;
    NUMTOOL::TMP::constexpr_for_range<0, trace_ndim>([&]<int idim> {

      // NOTE: trace_ndim - idim - 1 accounts for the fact that ordering is highest
      // dimension first
      int gidx1 = verticesR[MATH::power_T<2, trace_ndim - idim - 1>::value];

      // find the local indices on verticesL that have the same global indices 
      int lidx0 = -1, lidx1 = -1; 

      for(int ivert = 0; ivert < nvert_tr; ++ivert){
        if(verticesL[ivert] == gidx0) lidx0 = ivert;
        else if(verticesL[ivert] == gidx1) lidx1 = ivert;

        // short circuit if both found 
        if(lidx0 >= 0 && lidx1 >= 0) break;
      }

      // check if not found 
      if(lidx0 < 0 || lidx1 < 0) {
        errcode = -1;
        return;
      }

      // set the sign code if flipped 
      // (positive direction always has increasing indices)
      if(lidx1 < lidx0) sign_code |= idim_mask<idim>();

      // get the dimensional index 
      // through indexing tomfoolery
      left_directions[idim] = trace_ndim - MATH::log2(std::abs(lidx1 - lidx0)) - 1;
    });

    if(errcode != 0) return errcode;

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
      const T *s,
      TracePointView sR
  ) const {

    int iperm = orientationR / sign_code_bound;
    int sign_code = orientationR % sign_code_bound;

    for(int idim = 0; idim < trace_ndim; ++idim){
      // TODO: rename copysign_idim to better reflect multiply action
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

  static constexpr int n_nodes = trace_domain_trans.n_nodes();

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
      auto lc_tensor = NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<T, ndim>;
      T lc = lc_tensor.list_index(lc_indices);
      if(is_negative_xi) lc = -lc;

      first_dim_sign[itrace] = lc;
    }
  }

  /**
   * @brief transform from the trace space reference domain to the 
   * reference element domain 
   *
   * @param [in] node_indices the global node indices for the trace space 
   * @param [in] faceNr the trace number 
   * @param [in] s the location in the reference trace domain 
   * @param [out] xi the position in the reference element domain 
   */
  void transform(
    const IDX *node_indices,
    int traceNr,
    const T *s,
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

  void transform_physical(
      const IDX *node_indices,
      int traceNr,
      const MATH::GEOMETRY::Point<T, ndim - 1> &s,
      FE::NodalFEFunction<T, ndim> &coord,
      ElPointView x
  ) const {
    // zero fill 
    std::fill_n(x.data(), ndim, 0.0);
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    Tensor<T, n_nodes> Bi{};
    trace_domain_trans.tensor_prod.fill_shp(
        trace_domain_trans.interpolation_1d, s, Bi.data());

    for(int inode = 0; inode < n_nodes; ++inode) {
      for(int idim = 0; idim < ndim; ++idim) {
        const auto &node = coord[node_indices[inode]];
        x[idim] += Bi[inode] * node[idim];
      }
    }
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
      const IDX *face_node_indices,
      int traceNr,
      const MATH::GEOMETRY::Point<T, trace_ndim> &s
  ) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;

    // initialize the jacobian
    Tensor<T, ndim, trace_ndim> J;
    J = 0;

    // Get the gradient per basis function
    Tensor<T, trace_domain_trans.nnode, trace_ndim> dBidxj;
    trace_domain_trans.tensor_prod.fill_deriv(
        trace_domain_trans.interpolation_1d, s, dBidxj);

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

template<typename T, typename IDX, int Pn>
class HypercubeTraceTransformation<T, IDX, 1, Pn>{
  static constexpr int ndim = 1;
  static constexpr int trace_ndim = 0;
  static constexpr int n_trace = 2;

  using ElPointView = MATH::GEOMETRY::PointView<T, ndim>;
  using TracePointView = MATH::GEOMETRY::PointView<T, trace_ndim>;
  public:

  static constexpr int n_nodes = 1;
  /**
   * @brief transform from the trace space reference domain to the 
   * reference element domain 
   *
   * @param [in] node_indices the global node indices for the face 
   * @param [in] faceNr the trace number 
   * @param [in] s the location in the reference trace domain 
   * @param [out] xi the position in the reference element domain 
   */
  void transform(
    const IDX *node_indices,
    int traceNr,
    const T *s,
    ElPointView xi
  ) const {
    if(traceNr == 0){
      xi[0] = -1.0;
    } else {
      xi[0] = 1.0;
    }
  }

  void transform_physical(
      const IDX *node_indices,
      int traceNr,
      const MATH::GEOMETRY::Point<T, ndim - 1> &s,
      FE::NodalFEFunction<T, ndim> &coord,
      ElPointView x
  ) const {
    x[0] = coord[node_indices[0]][0];
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
      const IDX *face_node_indices,
      int traceNr,
      const MATH::GEOMETRY::Point<T, trace_ndim> &s
  ) const {
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    Tensor<T, ndim, trace_ndim> J{};
    if(traceNr == 0){
      // negative normal
      J[0][0] = -1.0;
      return J;
    } else {
      J[0][0] = 1.0;
      return J;
    }
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
    using namespace NUMTOOL::TENSOR::FIXED_SIZE;
    Tensor<T, ndim, trace_ndim> J{};
    if(traceNr == 0){
      // negative normal
      J[0][0] = -1.0;
      return J;
    } else {
      J[0][0] = 1.0;
      return J;
    }
  }
};

} // namespace ELEMENT::TRANSFORMATIONS
