#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/point.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iceicle/bitset.hpp>

namespace iceicle {
 
  /// @brief namespace for polytope implementation inspired by FEMPAR
  /// Badia et al 2018
  namespace polytope {
    
    /// @brief code for a topology
    /// NOTE: indexing operator[] goes from right to left so this will match codes from FEMPAR
    ///
    /// @tparam ndim the number of dimensions
    template<std::size_t ndim>
    using tcode = bitset<ndim>;

    /// @brief code for an extrusion 
    /// An extrusion represents a marker of if the given dimension of the topology is extruded
    /// in the geometry being considered 
    /// an extrusion of all ones is the full domain of the topology 
    /// all other extrusions refer to facets
    /// @tparam ndim the number of dimensions
    template<std::size_t ndim>
    using ecode = bitset<ndim>;

    template<std::size_t ndim>
    static constexpr ecode<ndim> full_extrusion = ~ecode<ndim>{0};

    /// @brief code for a vertex 
    /// @tparam ndim the number of dimensions
    template<std::size_t ndim>
    using vcode = bitset<ndim>;

    /// @brief code for a prismatic extrusion
    static constexpr bool prism_ext = 1;

    /// @brief code for a simplical extrusion 
    static constexpr bool simpl_ext = 0;

    // ======================
    // = geo_code utilities =
    // ======================

    /// @brief convert a vertex code to a point in space 
    /// @tparam T the floating point type 
    /// @tparam ndim the number of dimensions
    template<class T, std::size_t ndim>
    constexpr 
    auto vcode_to_point(vcode<ndim> v) noexcept -> MATH::GEOMETRY::Point<T, (int) ndim>
    {
      MATH::GEOMETRY::Point<T, (int) ndim> pt{};
      for(int idim = 0; idim < ndim; ++idim){
        if(v[idim] == 0){
          pt[idim] = static_cast<T>(0.0);
        } else {
          pt[idim] = static_cast<T>(1.0);
        }
      }
    }

    /// @brief get the number of dimensions of given code
    template<std::size_t ndim>
    constexpr
    auto get_ndim(bitset<ndim> code) -> int {
      return ndim;
    }

    template<class T>
    concept geo_code = requires(T code) {
      get_ndim(code);
    };

    // ============
    // = Vertices =
    // ============

    /// @brief get the number of vertices for the given polytope 
    /// @param t the code that defines the polytope domain 
    /// @return the number of vertices
    template<std::size_t ndim>
    constexpr 
    auto n_vert(tcode<ndim> t) noexcept -> std::size_t 
    {
      // special case (we will represent 0 dimensional objects with a single vertex)
      if constexpr(ndim == 0) { return 1; }
      else {
        std::size_t nvert = 2;
        for(auto idim = 1; idim < ndim; ++idim){
          //prism extrusions double the number of vertices 
          //while simplical extrusions converge to a single vertex at alpha = 1
          if(t[idim] == prism_ext) nvert *= 2;
          else nvert += 1;
        }
        return nvert;
      }
    }

    template<geo_code auto t>
    using vertex_list = std::array<vcode<get_ndim(t)>, n_vert(t)>;

    /// @brief generate the list of vertices for a given topology 
    /// @tparam the bitcode for the topology
    /// @return a list of bitcodes for each vertex 
    /// The bitcodes correspond to the given coordinate being either 0.0 or 1.0;
    template<geo_code auto t>
    constexpr
    auto gen_vert() noexcept -> vertex_list<t>
    {
      static constexpr int ndim = get_ndim(t);
      vertex_list<t> vertices;
      std::ranges::fill(vertices, vcode<ndim>{0});

      std::size_t nvert_current = 1;
      for(int idim = 0; idim < ndim; ++idim){
        if(t[idim] == simpl_ext){
          // simplex extrusion domain comes to a single point
          vertices[nvert_current] = vcode<ndim>{static_cast<unsigned long long>(std::pow(2, idim))};
          ++nvert_current;
        } else {
          // prismatic extrusion extrudes all the vertices of the current domain
          std::copy_n(vertices.begin(), nvert_current, vertices.begin() + nvert_current);
          for(int ivert = nvert_current; ivert < 2 * nvert_current; ++ivert){
            vertices[ivert][idim] = 1;
          }
          nvert_current *= 2;
        }
      }
      return vertices;
    }

    // ==========
    // = Facets =
    // ==========

    /// @brief get the number of facets of the given extrusion of the topology 
    template<std::size_t ndim>
    constexpr
    auto n_facets(tcode<ndim> t, ecode<ndim> e) -> std::size_t {
      if (e.to_ullong() == 0) return 0;
      else {
        std::size_t nvert = 1;
        for(int idim = 0; idim < ndim; ++idim){
          if(e[idim] == 1){
            if(t[idim] == prism_ext) nvert *= 2;
            else nvert += 1;
          }
        }
        return nvert;
      }
    }

    /// @brief an extrusion is said to have even parity if the sign for the 
    /// wedge product of basis vectors that form the orientation definition of the extrusion 
    ///
    /// e.g (e, v) = (110, 110)
    /// the orientation is defined by jhat /\ khat (y and z bits are set to 1 in e)
    /// the y coordinate of v == 1, therefore the j direciton of extrusion is negative (to be interior)
    /// the z coordinate of v == 1, therefore the k direction of extrusion is negative 
    /// -jhat /\ -khat = + (jhat /\ khat) therefore the parity is true
    ///
    /// is positive (returns true)
    /// if this is negative returns false
    template<std::size_t ndim>
    constexpr 
    auto extrusion_parity(ecode<ndim> e, vcode<ndim> v) -> bool 
    { return ( (e.count() - (e & ~v).count()) % 2 ) == 0; }

    /// @brief get the parity as defined above in extrusion_parity() 
    /// for the hodge dual of a given extrusion from a vertex
    template<std::size_t ndim>
    constexpr
    auto hodge_extrusion_parity(ecode<ndim> e, vcode<ndim> v) -> bool 
    {
      //TODO: probably faster way to directly compute levi civita based on e

      // set up the levi civita tensor for the hodge dual of the extrusion
      std::array<std::size_t, ndim> lc_indices{};
      int iindex = 0;

      // all dimensions in order where e == 1
      for(int idim = 0; idim < ndim; ++idim){
        if(e[idim] == 1) {
          lc_indices[iindex] = idim;
          iindex++;
        }
      }

      // all dimension in order where e == 0
      for(int idim = 0; idim < ndim; ++idim){
        if(e[idim] == 0) {
          lc_indices[iindex] = idim;
          iindex++;
        }
      }

      return extrusion_parity(e, v) ^
        (NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<int, ndim>.list_index(lc_indices.data()) == 1);
    }

    // ==================
    // = Tensor Product =
    // ==================

    namespace impl {

      /// @param t the topology code 
      /// @param e the extrusion code 
      /// @param nbasis_1d the number of basis functions in 1d 
      /// @param idim the 0-indexed dimension number
      static constexpr 
      auto get_n_node_recursive(geo_code auto t, geo_code auto e,
          std::size_t nbasis_1d, std::size_t idim) noexcept -> std::size_t 
      {
        // NOTE: idim is the 1-indexed dimension 
        // so when using for index we subtract 1
        if(e[idim] == 0) return get_n_node_recursive(t, e, nbasis_1d, idim - 1);
        else if(nbasis_1d == 1) return 1;
        else if(idim == 0){
          return nbasis_1d;
        } else {
          if(t[idim] == prism_ext)
            return nbasis_1d * get_n_node_recursive(t, e, nbasis_1d, idim - 1);
          else {
            std::size_t nnode = 1;
            for(int ibasis = 2; ibasis <= nbasis_1d; ++ibasis)
              nnode += get_n_node_recursive(t, e, ibasis, idim - 1);
            return nnode;
          }
        }
      }
    }

    /// @brief return the cumulative number of nodes of the extrusion e of topology t 
    /// up to dimension idim 
    /// @param t the topology 
    /// @param nbasis_1d the number of basis functions in 1 dimension
    static constexpr
    auto get_n_node(geo_code auto t, geo_code auto e,
        std::size_t nbasis_1d) noexcept -> std::size_t 
    {
      // static_assert(get_ndim(t) == get_ndim(e), "t and e must be same dimension.");
      // NOTE: use ndim - 1 for the index of the last dimension
      return impl::get_n_node_recursive(t, e, nbasis_1d, get_ndim(t) - 1);
    }

    namespace impl {

      /// @brief extend the given array by adding the next coordinate dimension
      template<class T, std::size_t ndim>
      constexpr
      auto extend_array(T value, std::array<T, ndim> arr) -> std::array<T, ndim + 1>
      {
        std::array<T, ndim + 1> ret{};
        for(std::size_t i = 0; i < ndim; ++i) ret[i] = arr[i];
        ret[ndim] = value;
        return ret;
      }

      template<
        class T,
        geo_code auto t,
        geo_code auto e,
        geo_code auto v
      >
      constexpr 
      auto facet_nodes(std::size_t nshp_1d, T domain_max) noexcept
      -> std::vector<std::array<T, get_ndim(t)>>
      {
        static constexpr std::size_t ndim = get_ndim(t);
        // the dimension index we are currently focusing on
        // for the recursive algorithm
        static constexpr std::size_t idim = ndim - 1; 

        // denominator for number of partitions protected for divide by zero
        T npartition_den = (nshp_1d > 1) ? nshp_1d - 1 : 1;

        if constexpr (ndim > 1) {

          std::vector<std::array<T, ndim>> nodes{};
          // geometric information of slices
          constexpr tcode<ndim - 1> t_slice{t.to_ullong()};
          constexpr vcode<ndim - 1> v_slice{v.to_ullong()};
          constexpr ecode<ndim - 1> e_slice{e.to_ullong()};


          if(e[idim] == 0) {

            // get the nodes from a slice
            auto slice_nodes = facet_nodes<T, t_slice, e_slice, v_slice>(nshp_1d, domain_max);

            // place the point in domain depending on v
            T xi = (v[idim] == 0) ? 0 : domain_max;

            // append coordinate to nodes from slice
            for(auto node : slice_nodes) nodes.push_back(extend_array(xi, node));
          } else {
            for(std::size_t ishp = 0; ishp < nshp_1d; ++ishp){

              // size of slice depends on direction if topology is simplex
              std::size_t slice_nshp = (t[idim] == simpl_ext) ? 
                ( (v[idim] == 0) ? nshp_1d - ishp : ishp )
                : nshp_1d;
              T slice_domain_max = domain_max * (slice_nshp - 1) / npartition_den;
              auto slice_nodes = facet_nodes<T, t_slice, e_slice, v_slice>(slice_nshp, slice_domain_max);

              // get the coordinate for this dimension 
              T xi = (v[idim] == 0) ? domain_max * (ishp / npartition_den) 
                : domain_max * ( (nshp_1d - ishp - 1) / npartition_den);
              for(auto node : slice_nodes) nodes.push_back(extend_array(xi, node));
            }
          }
          return nodes;
        } else if constexpr (ndim == 1){
          if(e[0] == 0) {
            std::array<T, 1> node{(v[idim] == 0) ? 0.0 : domain_max};
            return std::vector<std::array<T, 1>>{node};
          } else {
            std::vector<std::array<T, 1>> nodes{};
            for(std::size_t ishp = 0; ishp < nshp_1d; ++ishp){
              // get the coordinate for this dimension 
              T xi = (v[idim] == 0) ? domain_max * (ishp / npartition_den) 
                : domain_max * ( (nshp_1d - ishp - 1) / npartition_den);
              nodes.push_back(std::array<T, 1>{xi});
            }
            return nodes;
          }
        } else {
          return std::vector<std::array<T, ndim>>{};
        }

      }

      template<
        class index_type,
        index_type ndim,
        index_type size_1d,
        geo_code auto t,
        geo_code auto e,
        geo_code auto v
      >
      constexpr 
      auto multi_index_set_recursive(size_t idim) noexcept 
      {
        std::array< std::array< index_type, ndim>, get_n_nodes(t, e, size_1d)> nodes{};
        if(e[idim] == 0){
          // this dimension is not extruded (set to vertex value for all nodes)
          index_type basis_fill = (v[idim] == 0) ? 0 : size_1d - 1;
          for(index_type inode = 0; inode < get_n_node(t, e, size_1d); ++inode)
            nodes[inode][idim] = basis_fill;
        } else {
          // this dimension is extruded now we find out what kind of extrusion
          if(t[idim] == prism_ext) {
            index_type nfill = get_n_node_recursive(t, e, size_1d, idim - 1);
            for(index_type ibasis = 0; ibasis < size_1d; ++ibasis) {
              for(index_type ifill = 0; ifill < nfill; ++ifill)
                nodes[nfill * ibasis + ifill][idim] = ibasis;
            }
          } else {
            for(index_type ibasis = 0; ibasis < size_1d; ++ibasis) {
              index_type nfill = get_n_node_recursive(t, e, ibasis + 1, idim - 1);
              for(index_type ifill = 0; ifill < nfill; ++ifill)
                nodes[nfill * ibasis + ifill][idim] = ibasis;
            }

          }
        }
      }
    }

      template<
        class T,
        geo_code auto t,
        geo_code auto e,
        geo_code auto v,
        std::size_t nshp_1d
      >
      constexpr 
      auto facet_nodes() noexcept
      -> std::array< std::array< T, get_ndim(t) >, get_n_node(t, e, nshp_1d) > 
      { 
        auto nodes_vec = impl::facet_nodes<T, t, e, v>(nshp_1d, 1.0);
        std::array< std::array< T, get_ndim(t) >, get_n_node(t, e, nshp_1d) > nodes;
        std::ranges::copy(nodes_vec, nodes.begin());
        return nodes;
      }

    /// @brief generate the multi-index set for 
    /// @param t the given topology 
    /// @param e with the given extrusion 
    /// @param v the given vertex to extrude from
    template<
      class index_type,
      index_type ndim,
      index_type size_1d,
      geo_code auto t,
      geo_code auto e,
      geo_code auto v
    >
    constexpr
    auto multi_index_set() noexcept -> std::array< std::array<index_type, ndim>, get_n_node(t, e, size_1d)>
    {
      
    }

    /// @brief the tensor product for the given topology 
    ///
    /// @tparam T the real number type 
    /// @param t the topology of the domain 
    /// @param nbasis_1d the number of basis functions in one dimension
    template<class T, geo_code auto t, int nbasis_1d>
    struct TopologyTensorProd {

      static constexpr std::size_t ndim = get_ndim(t);

      /// @brief the total number of entries generated by the tensor product
      static constexpr std::size_t nvalues = get_n_node(t, full_extrusion<ndim>, nbasis_1d);
    };
  }
}
