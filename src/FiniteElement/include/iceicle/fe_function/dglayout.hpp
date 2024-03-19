/**
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief dglayout represents the memory layout of
 * a DG representation of a vector-valued fe_function
 */
#pragma once
#include <iceicle/fe_function/layout_enums.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/basis/basis.hpp>
#include <iceicle/element/finite_element.hpp>

#include <stdexcept>
#include <type_traits>
#include <algorithm>
namespace FE {

    /** 
     * @brief represents the map from the pair (ielem, idof) -> to a global dof index
     * NOTE: this is independent of vector components;
     */
    template< class IndexType = std::size_t >
    class dg_dof_map {
    public:

        // =====================
        // = Integral Typedefs =
        // =====================

        using index_type = IndexType;
        using size_type = std::make_unsigned_t<index_type>;

    private:
        std::size_t calculate_max_dof_size(std::vector<std::size_t> &offsets_arg){
            std::size_t max_dof_sz = 0;
            for(std::size_t i = 1; i < offsets_arg.size(); ++i){
                max_dof_sz = std::max(max_dof_sz , offsets_arg[i] - offsets_arg[i - 1]);
            }
            return max_dof_sz;
        }

        /// @brief offsets of the start of each element 
        ///        the dofs for each element are in the range 
        ///        [ offsets[ielem], offsets[ielem + 1] )
        std::vector<std::size_t> offsets;

        /// @brief the max size in number of degrees of freedom for an element
        std::size_t max_dof_size;

    public:

        // ================
        // = Constructors =
        // ================

        /** @brief default constructor */
        constexpr dg_dof_map() noexcept : offsets{0}, max_dof_size{0} {}

        /** @brief construct with uniform basis function */
        template<typename T, int ndim>
        constexpr dg_dof_map(
            index_type nelem,
            const BASIS::Basis<T, ndim> &basis
        ) noexcept : offsets(nelem + 1) {
            for(int i = 0; i <= nelem; ++i){
                offsets[i] = i * basis.nbasis();
            }
            max_dof_size = basis.nbasis();
        }

        /** @brief construct with a list of FiniteElements */ 
        template<class T, int ndim>
        constexpr dg_dof_map(
            const std::vector<ELEMENT::FiniteElement<T, index_type, ndim> > &elements
        ) noexcept : offsets(elements.size() + 1) {
            offsets[0] = 0;
            for(index_type i = 0; i < elements.size(); ++i){
                offsets[i + 1] = offsets[i] + elements[i].nbasis();
            }

            max_dof_size = calculate_max_dof_size(offsets);
        }

        // === Default nothrow copy and nothrow move semantics ===
        constexpr dg_dof_map(const dg_dof_map<index_type>& other) noexcept = default;
        constexpr dg_dof_map(dg_dof_map<index_type>&& other) noexcept = default;

        constexpr dg_dof_map& operator=(const dg_dof_map<index_type>& other) noexcept = default;
        constexpr dg_dof_map& operator=(dg_dof_map<index_type>&& other) noexcept = default;

        // =============
        // = Accessors =
        // =============

        /** 
         * @brief Convert element index and local degree of freedom index 
         * to the global degree of freedom index 
         * @param ielem the element index 
         * @param idof the local degree of freedom index
         */
        constexpr index_type operator[](index_type ielem, index_type idof) 
            const noexcept { return offsets[ielem] + idof; }

        /** get reference to the last offset which is equivalent to the size of the gdof index space */ 
        constexpr auto back() const noexcept { return offsets.back(); }

        // ===========
        // = Utility =
        // ===========

        /** @brief get the size requirement for all degrees of freedom given
         * the number of vector components per dof 
         * @param nv_comp th number of vector components per dof 
         * @return the size requirement
         */
        constexpr size_type calculate_size_requirement( index_type nv_comp ) const noexcept {
            return offsets.back() * nv_comp;
        }

        /**
         * @brief calculate the largest size requirement for a single element 
         * @param nv_comp the number of vector components per dof 
         * @return the maximum size requirement 
         */
        constexpr size_type max_el_size_reqirement( index_type nv_comp ) const noexcept {
            return nv_comp * max_dof_size;
        }

        /**
         * @brief get the number of degrees of freedom at the given element index 
         * @param elidx the index of the element to get the ndofs for 
         * @return the number of degrees of freedom 
         */
        [[nodiscard]] constexpr size_type ndof_el( index_type elidx ) const noexcept {
            return offsets[elidx + 1] - offsets[elidx];
        }

        /** @brief get the number of elements represented in the map */
        [[nodiscard]] constexpr size_type nelem() const noexcept { return offsets.size() - 1; }

        /** @brief get the size of the global degree of freedom index space represented by this map */
        [[nodiscard]] constexpr size_type gdof_size() const noexcept { return static_cast<size_type>(offsets.back()); }
    };

    // Deduction Guides
    template<class T, class IDX, int ndim>
    dg_dof_map(IDX, const BASIS::Basis<T, ndim>&) -> dg_dof_map<IDX>;
    template<class T, class IDX, int ndim>
    dg_dof_map(const std::vector<ELEMENT::FiniteElement<T, IDX, ndim> > &) -> dg_dof_map<IDX>;
}
