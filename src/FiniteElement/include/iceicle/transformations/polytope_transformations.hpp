#pragma once
#include "Numtool/fixed_size_tensor.hpp"
#include "Numtool/point.hpp"
#include "iceicle/linalg/linalg_utils.hpp"
#include "iceicle/basis/tensor_product.hpp"
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

    /// @brief given a topology and extrusion code -- get the number of vertices 
    /// @param t the topology code 
    /// @param e the extrusion code
    template<std::size_t ndim>
    [[nodiscard]] inline constexpr 
    auto n_vert(tcode<ndim> t, ecode<ndim> e)
    -> std::size_t {
      std::size_t nvert = 1;
      for(int idim = 0; idim < ndim; ++idim){
        if(e[idim] != 0){
          if(t[idim] == prism_ext) nvert *= 2;
          else nvert += 1;
        }
      }
      return nvert;
    }

    // @brief get the topology of a given extrusion e of a topology t 
    // @tparam t the toplogy 
    // @tparam e the extrusion 
    // @return the toplogy of the e extrusion of t
    template<geo_code auto t, geo_code auto e>
    [[nodiscard]] inline constexpr
    auto extrusion_topology()
    -> tcode<e.count()> 
    {
      std::size_t jdim = 0;
      tcode<e.count()> ext_t{};
      for(std::size_t idim = 0; idim < get_ndim(t); ++idim)
        if(e[idim] == 0) ext_t[jdim++] = t[idim];
      return ext_t;
    }

    [[nodiscard]] inline constexpr
    auto validate_facet(geo_code auto t, geo_code auto e, geo_code auto v)
    -> bool {
      // prevent ambiguous facet definitions
      int ndim = get_ndim(t);
      for(int idim = 0; idim < ndim; ++idim){
        
      }
      return true;
    }

    template<geo_code auto t, geo_code auto e, geo_code auto v>
    [[nodiscard]]
    auto facet_vertices()
    -> std::array<vcode<get_ndim(t)>, n_vert(t, e)>
    {
      static constexpr int ndim = get_ndim(t);
      std::array<vcode<get_ndim(t)>, n_vert(t, e)> vertices;
      vertices[0] = v;
      std::size_t inode = 1;
      for(int idim = 0; idim < ndim; ++idim){
        if (e[idim] != 0){
          if (t[idim] == simpl_ext){
            vertices[inode] = vertices[inode - 1];
            for(int jdim = 0; jdim < ndim; ++jdim)
              vertices[inode][jdim] = 0;
            vertices[inode].flip(idim);
            ++inode;
          } else {
            std::copy_n(vertices.begin(), inode, &vertices[inode]);
            std::size_t end = 2 * inode;
            for(; inode < end; ++inode ){
              vertices[inode].flip(idim);
            }
          }
        }
      }
      return vertices;
    }

    template<geo_code auto t, geo_code auto e>
    struct facet {
      static constexpr std::size_t ndim = e.count();
      static constexpr tcode<ndim> extruson_t = extrusion_toplogy(t, e);
    };
  }
}
