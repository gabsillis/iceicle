#pragma once

#include "fmt/base.h"
#include "iceicle/crs.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include <type_traits>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
namespace iceicle {

    namespace impl {
        template< class IDX >
        static util::crs<IDX, IDX> empty_crs{};
    }

    /// @brief a map between parallel indices
    /// and process local indices
    ///
    /// Parallel indices satisfy the following:
    /// - a parallel index (or pindex) is in the range 
    /// (0, size)
    ///
    /// local indices represent indices on the local process
    ///
    /// index_map[pindex] -> {owning process mpi rank, lindex}
    ///
    /// p_indices[lindex] -> pindex 
    ///
    /// "owned" pindices are the indices that are marked as belonging to this mpi_rank 
    /// lindices can map to pindices owned by other processes in index sets with conformity 
    /// The owned pindices for a given mpi rank are in the range
    /// [ owned_offsets[mpi_rank], owned_offsets[mpi_rank + 1] )
    ///
    /// This is so matrices can be constructed in block format per process
    ///
    /// If the indices are completely disjoint between processes (such as element indices),
    /// then p_indices = owned_offsets[my_mpi_rank], owned_offsets[my_mpi_rank] + 1, ...
    ///                  owned_offsets[my_mpi_rank + 1] - 1
    template< class IDX >
    struct pindex_map {
        using index_type = IDX;
        using size_type = std::make_unsigned_t<IDX>;

        /// @brief the index pairs of rank and process local index 
        /// synchronized over all ranks
        std::vector< p_index<IDX> > index_map;

        /// @brief for each local degree of freedom, the parallel index 
        std::vector< IDX > p_indices;

        /// @brief the offsets of "owned" pindex ranges
        std::vector< IDX > owned_offsets;

        /// @brief get the size of the pindex space
        [[nodiscard]] inline constexpr 
        auto size() const -> size_type 
        { return index_map.size(); }

        /// @brief get the number of local indices 
        auto n_lindex() const -> size_type 
        { return p_indices.size(); }

        /// @brief the size of the range of pindices that are owned by the given rank
        [[nodiscard]] inline constexpr 
        auto owned_range_size(int irank) const -> size_type 
        { return owned_offsets[irank + 1] - owned_offsets[irank]; }

        friend std::ostream& operator<< (std::ostream&out, const pindex_map<IDX> pidx_map) {
            if(mpi::mpi_world_rank() == 0){
                out << fmt::format("index_map:\n");
                out << fmt::format("pindex | rank | lindex\n");
                out << fmt::format("-------+------+-------\n");
                for(IDX pidx = 0; pidx < pidx_map.size(); ++pidx){
                    out << fmt::format(" {:5d} | {:4d} | {:5d}\n",
                            pidx, pidx_map.index_map[pidx].rank, pidx_map.index_map[pidx].index);
                }

                out << fmt::format("\n");
                out << fmt::format("offsets:\n");
                out << fmt::format("{}", pidx_map.owned_offsets);
                out << fmt::format("\n");
            }

            for(int irank = 0; irank < mpi::mpi_world_size(); ++irank) {
                if(irank == mpi::mpi_world_rank()){
                    out << fmt::format(" rank{} p_indices:\n", irank);
                    out << fmt::format(" lindex | pindex \n");
                    out << fmt::format("--------+--------\n");
                    for(IDX lindex = 0; lindex < pidx_map.p_indices.size(); ++lindex){
                        out << fmt::format(" {:6d} | {:6d} \n",
                                lindex, pidx_map.p_indices[lindex]);
                    }
                }
                mpi::mpi_sync();
            }
            return out;
        }
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

        /// the total number of degrees of freedom represented
        size_type ndof;

        /// the connectivity of the degrees of freedom for the elements
        /// for example, mesh nodes connectivity
        util::crs<IDX, IDX> dof_connectivity;

        // ================
        // = Constructors =
        // ================

        constexpr 
        dof_map() : ndof{0}, dof_connectivity{impl::empty_crs<IDX>} {}

        // === Default nothrow copy and nothrow move semantics ===
        constexpr dof_map(const dof_map<IDX, ndim, conformity>& other) noexcept = default;
        constexpr dof_map(dof_map<IDX, ndim, conformity>&& other) noexcept = default;
        constexpr dof_map& operator=(const dof_map<IDX, ndim, conformity>& other) noexcept 
            = default;
        constexpr dof_map& operator=(dof_map<IDX, ndim, conformity>&& other) noexcept 
            = default;

        /// @brief construct from dof connectivity 
        /// takes universal reference to things that can construct dof connectivity
        constexpr 
        dof_map(std::integral auto ndof, auto&& dof_connectivity) 
        requires(std::constructible_from<util::crs<IDX, IDX>, decltype(dof_connectivity)>)
        : ndof{(size_type) ndof}, dof_connectivity{dof_connectivity} {}

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
        { return ndof; }
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
            // TODO: 
        } else {
            IDX nelem = dofs.dof_connectivity.nrow();
            std::vector<IDX> newcols(nelem + 1);
            newcols[0] = 0;
            for(IDX ielem = 0; ielem < nelem; ++ielem) {
                newcols[ielem + 1] = newcols[ielem] 
                    + dofs.dof_connectivity.rowsize(el_renumbering[ielem]);
            }
            util::crs<IDX, IDX> new_connectivity{std::span<const IDX>{newcols}};
            for(IDX ielem = 0; ielem < new_connectivity.nrow(); ++ielem){
                for(int icol = 0; icol < new_connectivity.rowsize(ielem); ++icol)
                    new_connectivity[ielem, icol] = 
                        dofs.dof_connectivity[el_renumbering[ielem], icol];
            }
            return dof_map< IDX, ndim, conformity >(dofs.size(), std::move(new_connectivity));
        }
    }

//     template< class IDX, int ndim, int conformity >
//     [[nodiscard]] inline constexpr
//     auto apply_dof_renumbering(
//         const dof_map< IDX, ndim, conformity >& dofs,
//         const std::vector<IDX>& dof_old_to_new
//     ) noexcept -> dof_map< IDX, ndim, conformity > {
//         if constexpr( conformity == l2_conformity(ndim) ) {
// 
//         } else {
//             util::crs new_connectivity{dofs.dof_connectivity};
//             for(IDX ielem = 0; ielem < new_connectivity.nrow(); ++ielem){
//                 for(int icol = 0; icol < new_connectivity.rowsize(ielem); ++icol) {
//                     new_connectivity[ielem, icol] = dof_old_to_new[new_connectivity[ielem, icol]];
//                 }
//             }
//             return dof_map< IDX, ndim, conformity >(std::move(new_connectivity));
//         }
//     }

    ///
    /// @return a tuple of 
    ///  - The new dof map of process local dofs (unique per process)
    ///  - The pdof map that maps local dofs, ranks, and renumbered (new) pdofs 
    ///  - the new_to_old renumbering vector 
    ///    where renumbering[new_pdof] = old_pdof 
    template< class IDX, int ndim, int conformity >
    [[nodiscard]] inline constexpr
    auto partition_dofs(
        const pindex_map<IDX>& el_part,
        const mpi::on_rank< dof_map< IDX, ndim, conformity > >& global_dofs
    ) noexcept -> std::tuple< dof_map< IDX, ndim, conformity >, pindex_map<IDX>, std::vector<IDX> >
    {
#ifdef ICEICLE_USE_MPI 
        MPI_Status status;
#endif
        int nrank = mpi::mpi_world_size();
        int myrank = mpi::mpi_world_rank();
        std::vector< std::vector < IDX > > all_pdofs(nrank,
                std::vector<IDX>{});
        std::vector<IDX> my_pdofs;

        IDX nelem, ndof;
        dof_map<IDX, ndim, conformity> gdofs;

        // communicate the global dofs map
        if(global_dofs.has_value()){
            const auto& global_dofs_val = global_dofs.value();
            nelem = global_dofs_val.nelem();
            ndof = global_dofs_val.ndof;
            gdofs = global_dofs.value();
#ifdef ICEICLE_USE_MPI 
            MPI_Bcast(&nelem, 1, mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
            MPI_Bcast(&ndof, 1, mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
            MPI_Bcast(gdofs.dof_connectivity.cols(), nelem + 1,
                    mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
            MPI_Bcast(gdofs.dof_connectivity.data(), gdofs.dof_connectivity.nnz(),
                    mpi_get_type<IDX>(), myrank, MPI_COMM_WORLD);
#endif
        } else {
#ifdef ICEICLE_USE_MPI 
            MPI_Bcast(&nelem, 1, mpi_get_type<IDX>(),
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            MPI_Bcast(&ndof, 1, mpi_get_type<IDX>(),
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            std::vector<IDX>cols(nelem + 1);
            MPI_Bcast(cols.data(), nelem + 1, mpi_get_type<IDX>(),
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            util::crs<IDX, IDX> gdofs_crs{cols};
            MPI_Bcast(gdofs_crs.data(), gdofs_crs.nnz(), mpi_get_type<IDX>(), 
                    global_dofs.valid_rank(), MPI_COMM_WORLD);
            gdofs = dof_map<IDX, ndim, conformity>{ndof, gdofs_crs};
#endif
        }

        // array of which rank owns each degree of freedom
        std::vector<int> owning_rank(gdofs.size(), nrank);

        // build the dof list
        for(IDX iel = 0; iel < nelem; ++iel){
            int el_rank = el_part.index_map[iel].rank;
            if(el_rank == myrank) for(IDX pdof : gdofs.rowspan(iel)){
                my_pdofs.push_back(pdof);
                owning_rank[pdof] = myrank; // temporarily claim ownership
            }
        }
#ifdef ICEICLE_USE_MPI
        // ownership is determined by lowest MPI rank
        MPI_Allreduce(MPI_IN_PLACE, owning_rank.data(), owning_rank.size(), MPI_INT, MPI_MIN, MPI_COMM_WORLD);
#endif

        // sort and remove duplicates
        std::ranges::sort(my_pdofs);
        auto unique_subrange = std::ranges::unique(my_pdofs);
        my_pdofs.erase(unique_subrange.begin(), unique_subrange.end());


        // create a renumbering that makes contiguous dof ranges 
        // for each process based on ownership
        std::vector<std::vector<IDX>> owned_pdofs(nrank, std::vector<IDX>{});
        std::vector<IDX> renumbering; // old_pdof = renumbering[new_pdof]
        renumbering.reserve(gdofs.size());
        std::vector<IDX> offsets = {0};
        std::vector< p_index<IDX> > index_map;
        index_map.reserve(gdofs.size());
        for(IDX pdof = 0; pdof < gdofs.size(); ++pdof){
            owned_pdofs[owning_rank[pdof]].push_back(pdof);
        }
        for(int irank = 0; irank < nrank; ++irank){
            if(irank == mpi::mpi_world_rank()){
                fmt::println("pdofs rank {}: {}", irank, my_pdofs);
                fmt::println("owned pdofs rank {}: {}", irank, owned_pdofs[irank] );
            }
            mpi::mpi_sync();
        }
        for(int irank = 0; irank < nrank; ++irank){
            renumbering.insert(std::end(renumbering), 
                    std::begin(owned_pdofs[irank]), std::end(owned_pdofs[irank]));
            // construct the pdof_map index_mapping while we are at it
            for(int ldof = 0; ldof < owned_pdofs[irank].size(); ++ldof){
                index_map.emplace_back(irank, ldof);
            }
            offsets.push_back(offsets.back() + owned_pdofs[irank].size());
        }
        fmt::println("renumbering: {}", renumbering);
        // new_pdof = inverse_renumbering[old_pdof]
        std::vector<IDX> inverse_renumbering(renumbering.size());
        for(IDX i = 0; i < renumbering.size(); ++i){
            inverse_renumbering[renumbering[i]] = i;
        }

        // apply the renumbering to my_pdofs 
        std::for_each(my_pdofs.begin(), my_pdofs.end(), 
                [&inverse_renumbering](IDX &n) { n = inverse_renumbering[n]; });
       
        // create a temporary inverse mapping of my_pdofs to ldofs
        std::vector<IDX> inv_pdofs(gdofs.size(), -1);
        for(int ldof = 0; ldof < my_pdofs.size(); ++ldof){
            inv_pdofs[my_pdofs[ldof]] = ldof;
        }

        // setup the cols array for number of process local elements
        IDX nelem_local = el_part.p_indices.size();
        std::vector<IDX> ldof_cols(nelem_local + 1);
        ldof_cols[0] = 0;
        for(IDX iel = 0; iel < nelem; ++iel){
            int el_rank = el_part.index_map[iel].rank;
            if(el_rank == myrank){
                IDX iel_local = el_part.index_map[iel].index;
                // NOTE: only putting the size in for now 
                // need another pass to accumulate
                ldof_cols[iel_local + 1] = gdofs.ndof_el(iel);
            }
        }
        for(int i = 1; i < ldof_cols.size(); ++i)
            ldof_cols[i] = ldof_cols[i - 1] + ldof_cols[i];
        util::crs<IDX, IDX> ldof_crs{std::span<const IDX>{ldof_cols}};
        fmt::println("ldof_cols rank {}: {}", mpi::mpi_world_rank(), ldof_cols);

        /// fill with inverse mapping
        for(IDX iel_local = 0; iel_local < nelem_local; ++iel_local){
            IDX iel_global = el_part.p_indices[iel_local];
            for(int idof = 0; idof < ldof_crs.rowsize(iel_local); ++idof){
                ldof_crs[iel_local, idof] = inv_pdofs[inverse_renumbering[
                    gdofs[iel_global, idof]]];
            }
        }

        return std::tuple{ 
            dof_map< IDX, ndim, conformity >{my_pdofs.size(), std::move(ldof_crs)},
            pindex_map{ index_map, my_pdofs, offsets },
            renumbering
        };
    }
}
