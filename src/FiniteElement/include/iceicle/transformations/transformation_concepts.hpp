
#pragma once
#include "iceicle/tmp_utils.hpp"
#include <type_traits>
namespace iceicle::transformations {

  // =======================================
  // = Element Transformation Requirements =
  // =======================================

  /// @brief The element transformation TransT has a method called get_face_vert() which takes:
  /// - [in] the face number,
  /// - [in] the element node indices as an array of TransT::index_type, 
  /// - [out] the node indices to an array of size n_facevert() 
  /// that correspond to the vertices of the face 
  ///
  /// NOTE: These vertices must be in the same order as if get_element_vert() 
  /// was called on the transformation corresponding to the face
  template<class TransT>
  concept has_get_face_vert =
    requires(
        std::remove_reference_t<TransT> trans,
        int face_number,
        const tmp::index_type_of<TransT>* nodes_el,
        tmp::index_type_of<TransT>* vert_fac
        ) {
      {trans.n_facevert(face_number)} -> std::same_as<int>;
      {trans.get_face_vert(face_number, nodes_el, vert_fac)};
    };

  /// @brief The element transformation TransT has a method called get_face_nr()
  /// Given index_type = transT::index_type
  /// - [in] index_type* nodes_el the node indices of the element 
  /// - [in] index_type* vert_fac the indices of the vertices of the face
  /// - returns the face number as an integer or -1 if not found
  template<class TransT>
  concept has_get_face_nr = 
    requires(
        std::remove_reference_t<TransT> trans,
        const tmp::index_type_of<TransT>* nodes_el,
        const tmp::index_type_of<TransT>* vert_fac
    ) {
      {trans.get_face_nr(nodes_el, vert_fac)} -> std::same_as<int>;
    };
}
