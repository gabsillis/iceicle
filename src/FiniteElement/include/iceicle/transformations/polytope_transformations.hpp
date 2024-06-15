#pragma once
#include "Numtool/point.hpp"
#include <algorithm>
#include <cmath>
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

    /// @brief code for a vertex 
    /// @tparam ndim the number of dimensions
    template<std::size_t ndim>
    using vcode = bitset<ndim>;

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

    /// @brief code for a prismatic extrusion
    static constexpr bool prism_ext = 1;

    /// @brief code for a simplical extrusion 
    static constexpr bool simpl_ext = 0;

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

    // ==========
    // = Facets =
    // ==========

    /// @brief get the number of facets of the given extrusion of the topology 
    template<int ndim>
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
  }
}
