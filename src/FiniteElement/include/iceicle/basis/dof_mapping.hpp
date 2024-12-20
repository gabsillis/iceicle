#pragma once

#include "iceicle/crs.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/mpi_type.hpp"
#include <mpi.h>
namespace iceicle {

    /// @brief indical mapping of the parallel partitioning of elements
    template< class IDX >
    struct ElementPartitioning {
        /// @brief the index pairs for rank and process local index 
        /// synchronized over all ranks
        std::vector< p_index<IDX> > el_index_pairs;

        /// @brief the parallel global indices for each process-local element index
        /// this is individual per rank
        std::vector< IDX > el_p_idxs;
    };

    namespace impl {
        template< class IDX >
        static util::crs<IDX, IDX> empty_crs{};
    }

    /// @brief a map between parallel degrees of freedom 
    /// and process local degrees of freedom
    template< class IDX >
    class pdof_map {

        /// @brief the index pairs of rank and process local index 
        /// synchronized over all ranks
        std::vector< p_index<IDX> > index_map;

        /// @brief for each local degree of freedom, the parallel index 
        std::vector< IDX > p_indices;
    };

    /// @brief a map of degrees of freedom
    ///
    /// NOTE: unspecialized implementation is for maps with some 
    /// level of conformity 
    ///
    /// @tparam IDX the index type 
    /// @tparam ndim the number of dimensions 
    /// @tparam conformity the conformity code 
    ///         This is equal to the index in the De Rham exact sequence
    ///         H1 is 0, L2 is ndim
    template< class IDX, int ndim, int conformity >
    struct dof_map {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;

        // ==============
        // = Properties =
        // ==============

        /**
         * @brief consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same
         * This is not true for h1 conformity because there are shared dofs
         */
        inline static constexpr bool local_dof_contiguous() noexcept 
        { return false; }

        // ================
        // = Data Members =
        // ================

        /// the connectivity of the degrees of freedom for the elements
        /// for example, mesh nodes connectivity
        util::crs<IDX, IDX> dof_connectivity;

        // ================
        // = Constructors =
        // ================

        constexpr 
        dof_map() : dof_connectivity{impl::empty_crs<IDX>} {}

        /// @brief construct from dof connectivity 
        /// takes universal reference to things that can construct dof connectivity
        constexpr 
        dof_map(auto&& dof_connectivity) 
        : dof_connectivity{dof_connectivity} {}

        // === Default nothrow copy and nothrow move semantics ===
        constexpr dof_map(const dof_map<IDX, ndim, conformity>& other) noexcept 
        : dof_connectivity{other.dof_connectivity} {}
        constexpr dof_map(dof_map<IDX, ndim, conformity>&& other) noexcept 
        : dof_connectivity{other.dof_connectivity} {}

        constexpr dof_map& operator=(const dof_map<IDX, ndim, conformity>& other) noexcept {
            dof_connectivity = other.dof_connectivity;
            return *this;
        };
        constexpr dof_map& operator=(dof_map<IDX, ndim, conformity>&& other) noexcept {
            dof_connectivity = other.dof_connectivity;
            return *this;
        }

        // =============
        // = Accessors =
        // =============

        /** 
         * @brief Convert element index and local degree of freedom index 
         * to the global degree of freedom index 
         * @param ielem the element index 
         * @param idof the local degree of freedom index
         */
        [[nodiscard]] inline constexpr 
        auto operator[](index_type ielem, index_type idof) const noexcept
        -> index_type
        { return dof_connectivity[ielem, idof]; }

        /** 
         * @brief Convert element index and local degree of freedom index 
         * to the global degree of freedom index 
         * @param ielem the element index 
         * @param idof the local degree of freedom index
         */
        [[nodiscard]] inline constexpr
        auto operator[](index_type ielem, index_type idof) noexcept
        -> index_type&
        { return dof_connectivity[ielem, idof]; }

        // ===========
        // = Utility =
        // ===========

        /** @brief get the size requirement for all degrees of freedom given
         * the number of vector components per dof 
         * @param nv_comp th number of vector components per dof 
         * @return the size requirement
         */
        constexpr size_type calculate_size_requirement( index_type nv_comp ) const noexcept {
            return size() * nv_comp;
        }

        /**
         * @brief calculate the largest size requirement for a single element 
         * @param nv_comp the number of vector components per dof 
         * @return the maximum size requirement 
         */
        constexpr size_type max_el_size_requirement( index_type nv_comp ) const noexcept {
            size_type max = 0;
            for(int irow = 0; irow < dof_connectivity.nrow(); ++irow){
                max = std::max(dof_connectivity.rowsize(irow) * nv_comp, max);
            }
            return max;
        }

        /**
         * @brief get the number of degrees of freedom at the given element index 
         * @param iel the index of the element to get the ndofs for 
         * @return the number of degrees of freedom 
         */
        [[nodiscard]] constexpr size_type ndof_el( index_type iel ) const noexcept {
            return dof_connectivity.rowsize(iel);
        }

        /// @brief get a span over the degrees of freedom for a given element index
        /// @param iel the index of the element to get the dofs for 
        /// @return a span over the dofs
        [[nodiscard]] inline constexpr 
        auto rowspan( index_type iel ) const noexcept 
        -> std::span<const index_type> {
            return dof_connectivity.rowspan(iel);
        }

        /// @brief get a span over the degrees of freedom for a given element index
        /// @param iel the index of the element to get the dofs for 
        /// @return a span over the dofs
        [[nodiscard]] inline constexpr 
        auto rowspan( index_type iel ) noexcept 
        -> std::span<index_type> {
            return dof_connectivity.rowspan(iel);
        }

        /** @brief get the number of elements represented in the map */
        [[nodiscard]] constexpr size_type nelem() const noexcept 
        { return dof_connectivity.nrow(); }

        /** @brief get the size of the global degree of freedom index space represented by this map */
        [[nodiscard]] constexpr size_type size() const noexcept 
        { return static_cast<size_type>(dof_connectivity.nnz()); }
    };

    /// @brief apply the renumbering of elements to a dof_map 
    /// return a new dof map that preserves the same dofs with the new element numbers
    template< class IDX, int ndim, int conformity >
    [[nodiscard]] inline constexpr 
    auto apply_el_renumbering (
        const dof_map< IDX, ndim, conformity >& dofs,
        const std::vector<IDX>& el_renumbering 
    ) noexcept -> dof_map< IDX, ndim, conformity> {
        if constexpr( conformity == l2_conformity(ndim) ) {

        } else {
            IDX nelem = dofs.dof_connectivity.nrow();
            std::vector<IDX> newcols(nelem + 1);
            newcols[0] = 0;
            for(IDX ielem = 0; ielem < nelem; ++ielem) {
                newcols[ielem + 1] = newcols[ielem] 
                    + dofs.dof_connectivity.rowsize(el_renumbering[ielem]);
            }
            util::crs new_connectivity{newcols};
            for(IDX ielem = 0; ielem < new_connectivity.nrow(); ++ielem){
                for(int icol = 0; icol < new_connectivity.rowsize(ielem); ++icol)
                    newcols[ielem, icol] = 
                        dofs.dof_connectivity[el_renumbering[ielem], icol];
            }
            return dof_map< IDX, ndim, conformity >(std::move(new_connectivity));
        }
    }

    template< class IDX, int ndim, int conformity >
    [[nodiscard]] inline constexpr
    auto partition_dofs(
        ElementPartitioning<IDX> el_part,
        const mpi::on_rank< dof_map< IDX, ndim, conformity > >& global_dofs
    ) noexcept -> std::pair< dof_map< IDX, ndim, conformity >, pdof_map<IDX> >
    {
        MPI_Status status;
        int nrank = mpi::mpi_world_size();
        int myrank = mpi::mpi_world_rank();
        std::vector< std::vector < IDX > > all_pdofs(nrank,
                std::vector<IDX>{});
        std::vector<IDX> my_pdofs;

        IDX nelem;
        dof_map<IDX, ndim, conformity> gdofs;

        // communicate the global dofs map
        if(global_dofs.has_value()){
            const auto& global_dofs_val = global_dofs.value();
            nelem = global_dofs_val.nelem();
#ifdef ICEICLE_USE_MPI 
            MPI_Bcast(&nelem, 1, mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
#endif
            gdofs = global_dofs.value();
            MPI_Bcast(gdofs.dof_connectivity.data(), gdofs.dof_connectivity.nnz(),
                    mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
        } else {
#ifdef ICEICLE_USE_MPI 
            MPI_Bcast(&nelem, 1, mpi_get_type<IDX>(),
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            std::vector<IDX>cols(nelem + 1);
            MPI_Bcast(&cols, nelem + 1, mpi_get_type<IDX>(),
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            util::crs<IDX, IDX> gdofs_crs{cols};
            MPI_Bcast(gdofs_crs.data(), gdofs_crs.nnz(), mpi_get_type<IDX>(), 
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            gdofs = dof_map<IDX, ndim, conformity>{gdofs_crs};
#endif
        }

        // build the dof list
        for(IDX iel = 0; iel < nelem; ++iel){
            int el_rank = el_part.el_index_pairs[iel].rank;
            if(el_rank == myrank) for(IDX pdof : gdofs.rowspan(iel)){
                my_pdofs.push_back(pdof);
            }
        }

        // sort and remove duplicates
        std::ranges::sort(my_pdofs);
        auto unique_subrange = std::ranges::unique(my_pdofs);
        my_pdofs.erase(unique_subrange.begin(), unique_subrange.end());
       
        // create a temporary inverse mapping
        std::vector<IDX> inv_pdofs(gdofs.size(), -1);
        for(int ldof = 0; ldof < my_pdofs.size(); ++ldof){
            inv_pdofs[my_pdofs[ldof]] = ldof;
        }

        // setup the cols array for number of process local elements
        std::vector ldof_cols{el_part.el_p_idxs.size() + 1};
        ldof_cols[0] = 0;
        for(IDX iel = 0; iel < nelem; ++iel){
            int el_rank = el_part.el_index_pairs[iel].rank;
            if(el_rank == myrank){
                IDX iel_local = el_part.el_index_pairs[iel].index;
                // NOTE: only putting the size in for now 
                // need another pass to accumulate
                ldof_cols[iel_local + 1] = gdofs.ndof_el(iel);
            }
        }
        for(int i = 1; i < ldof_cols.size(); ++i)
            ldof_cols[i] = ldof_cols[i - 1] + ldof_cols[i];
        util::crs<IDX, IDX> ldof_crs{ldof_cols};

        /// fill with inverse mapping
        for(IDX iel_local = 0; iel_local < ldof_cols.size(); ++iel_local){
            IDX iel_global = el_part.el_p_idxs[iel_local];
            for(int idof = 0; idof < ldof_crs.rowsize(iel_local); ++idof){
                ldof_crs[iel_local, idof] = inv_pdofs[gdofs[iel_global, idof]];
            }
        }

        // TODO: finish index_map for pdof_map and 
        // determining which rank owns duplicated
    }
}
