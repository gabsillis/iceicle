#pragma once

#include "fmt/base.h"
#include "iceicle/crs.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include "iceicle/tmp_utils.hpp"
#include <mpi.h>
#include <numeric>
#include <unordered_map>
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
    /// local indices represent indices on the local process
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

        /// @brief for each local degree of freedom, the parallel index 
        /// This can be split into two contiguous disjoint sets 
        /// the first set is all ldofs that map to pdofs owned by this process 
        /// the second set is the ldofs that map to pdofs owned by other processes
        std::vector< IDX > p_indices;

        /// @brief for each parallel index that has a local degree of freedom, 
        /// inv_p_indices[pidx] = local index
        std::unordered_map< IDX, IDX > inv_p_indices;

        /// @brief the offsets of "owned" pindex ranges
        std::vector< IDX > owned_offsets;

        // create an index map that represents serial indices 
        // in the range [0, nindices)
        // all indices belong to the current mpi rank
        [[nodiscard]] static inline constexpr 
        auto create_serial(IDX nindices)
        -> pindex_map<IDX>
        {
            std::vector<IDX> p_indices(nindices);
            std::iota(p_indices.begin(), p_indices.end(), 0);
            std::unordered_map<IDX, IDX> inv_p_indices{};
            for(IDX pindex = 0; pindex < nindices; ++pindex){
                inv_p_indices[pindex] = pindex;
            }
            std::vector< IDX > owned_offsets{0, nindices};
            return pindex_map<IDX>{
                .p_indices = p_indices,
                .inv_p_indices = inv_p_indices,
                .owned_offsets = owned_offsets
            };
        }

        /// @brief check that the invariants described for this class hold
        [[nodiscard]] inline constexpr 
        auto check_invariants() -> bool 
        {
            bool valid = true;

            // check contiguous disjoint sets of p_indices
            for(IDX lidx = 0; lidx < owned_range_size(mpi::mpi_world_rank()); ++lidx){
                IDX pidx = inv_p_indices[lidx];
                if(
                    pidx < owned_offsets[mpi::mpi_world_rank()] 
                    || pidx >= owned_offsets[mpi::mpi_world_rank() + 1]
                ) { valid = false; }
            }

            bool all_valid = valid;
#ifdef ICEICLE_USE_MPI
            MPI_Allreduce(&valid, &all_valid, 1, MPI_C_BOOL, MPI_LAND, MPI_COMM_WORLD);
#endif
            return all_valid;
        }

        /// @brief get the size of the pindex space
        [[nodiscard]] inline constexpr 
        auto size() const -> size_type 
        { return owned_offsets.back(); }

        /// @brief get the number of local indices 
        auto n_lindex() const -> size_type 
        { return p_indices.size(); }

        /// @brief get the rank that owns the given index
        [[nodiscard]] inline constexpr 
        auto owning_rank(IDX index)
        -> int 
        {
            return std::distance(owned_offsets.begin(), 
                std::lower_bound(owned_offsets.begin(), owned_offsets.end(), index)) - 1;
        }

        /// @brief the size of the range of pindices that are owned by the given rank
        [[nodiscard]] inline constexpr 
        auto owned_range_size(int irank) const -> size_type 
        { return owned_offsets[irank + 1] - owned_offsets[irank]; }

        /// @brief get a range of indices 
        [[nodiscard]] inline constexpr
        auto owned_pindex_range(int irank) const noexcept
        { return std::ranges::iota_view{owned_offsets[irank], owned_offsets[irank + 1]}; }

        friend std::ostream& operator<< (std::ostream&out, const pindex_map<IDX> pidx_map) {

            if(mpi::mpi_world_rank() == 0){
                out << fmt::format("index_map:\n");
                out << fmt::format("pindex | rank | lindex\n");
                out << fmt::format("-------+------+-------\n");

            }
            mpi::mpi_sync();

            for(int irank = 0; irank < mpi::mpi_world_size(); ++irank) {
                if(irank == mpi::mpi_world_rank()) for(IDX pidx : pidx_map.owned_pindex_range(irank)){
                    out << fmt::format(" {:5d} | {:4d} | {:5d}\n",
                            pidx, irank, pidx_map.inv_p_indices.at(pidx));
                }

                mpi::mpi_sync();
            }

            for(int irank = 0; irank < mpi::mpi_world_size(); ++irank) {
                if(irank == mpi::mpi_world_rank()){
                    out << fmt::format(" rank{} p_indices:\n", irank);
                    out << fmt::format(" lindex | pindex \n");
                    out << fmt::format("--------+--------\n");
                    for(IDX lindex = 0; lindex < pidx_map.size(); ++lindex){
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
         * This is not true for any conformity other than L2 because of shared dofs
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

        /// @brief get a span over the degrees of freedom for a given element index
        /// @param iel the index of the element to get the dofs for 
        /// @return a span over the dofs
        [[nodiscard]] inline constexpr 
        auto operator[]( index_type iel ) const noexcept 
        -> std::span<const index_type> 
        { return rowspan(iel); }

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

    /// @brief degree of freedom map for L2 elements 
    /// There are no shared degrees of freedom between elements in L2 space,
    /// allowing a specialized mapping for the disjoint nature of indices
    template< class IDX, int ndim >
    struct dof_map<IDX, ndim, l2_conformity(ndim) > {

        // ============
        // = Typedefs =
        // ============
        using index_type = IDX;
        using size_type = std::make_unsigned_t<index_type>;

        // ==============
        // = Properties =
        // ==============

        std::size_t calculate_max_dof_size(std::vector<index_type> &offsets_arg){
            index_type max_dof_sz = 0;
            for(index_type i = 1; i < offsets_arg.size(); ++i){
                max_dof_sz = std::max(max_dof_sz , offsets_arg[i] - offsets_arg[i - 1]);
            }
            return max_dof_sz;
        }

        /**
         * @brief consecutive local degrees of freedom (ignoring vector components)
         * are contiguous in the layout
         * meaning that the data for a an element can be block copied 
         * to a elspan provided the layout parameters are the same
         */
        inline static constexpr bool local_dof_contiguous() noexcept 
        { return true; }

        /// @brief offsets of the start of each element 
        ///        the dofs for each element are in the range 
        ///        [ offsets[ielem], offsets[ielem + 1] )
        std::vector<index_type> offsets;

        /// @brief the max size in number of degrees of freedom for an element
        std::size_t max_dof_size;

        // ================
        // = Constructors =
        // ================

        /** @brief default constructor */
        constexpr dof_map() noexcept : offsets{0}, max_dof_size{0} {}

        /** @brief construct from a given offsets array that we move from */
        constexpr dof_map(std::vector<index_type>&& offsets_arg)
        : offsets{std::move(offsets_arg)}, max_dof_size{calculate_max_dof_size(offsets)}
        {}

        /** @brief construct from a range of elements that can
         * specify the number of basis functions 
         * through a function nbasis() 
         **/
        constexpr dof_map(std::ranges::range auto&& elements) noexcept 
        : offsets(std::ranges::size(elements) + 1), max_dof_size{0} {
            offsets[0] = 0;
            index_type ielem = 0;
            for(const auto& el : elements){
                int ndof = el.nbasis();
                max_dof_size = std::max(max_dof_size, (std::size_t) ndof);
                offsets[ielem + 1] = offsets[ielem] + ndof;
                ++ielem;
            }
        }

        /// @brief overload of range of elements constructor for argument deduction
        constexpr dof_map(std::ranges::range auto elements, tmp::compile_int<l2_conformity(ndim)> l2_arg) noexcept 
        : dof_map<IDX, ndim, l2_conformity(ndim)>(elements) {}

        // === Default nothrow copy and nothrow move semantics ===
        constexpr dof_map(const dof_map<index_type, ndim, l2_conformity(ndim)>& other) noexcept = default;
        constexpr dof_map(dof_map<index_type, ndim, l2_conformity(ndim)>&& other) noexcept = default;

        constexpr dof_map& operator=(const dof_map<index_type, ndim, l2_conformity(ndim)>& other) noexcept = default;
        constexpr dof_map& operator=(dof_map<index_type, ndim, l2_conformity(ndim)>&& other) noexcept = default;

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

        /// @brief get a span over the degrees of freedom for a given element index
        /// @param iel the index of the element to get the dofs for 
        /// @return a span over the dofs
        [[nodiscard]] inline constexpr 
        auto operator[]( index_type iel ) const noexcept 
        -> std::span<const index_type> 
        { return rowspan(iel); }

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

        /// @brief get a span over the degrees of freedom for a given element index
        /// @param iel the index of the element to get the dofs for 
        /// @return a span over the dofs
        [[nodiscard]] inline constexpr 
        auto rowspan( index_type iel ) const noexcept 
        -> std::span<const index_type> 
        { return std::span{std::ranges::iota_view{offsets[iel], offsets[iel + 1]}}; }

        /**
         * @brief get the number of degrees of freedom at the given element index 
         * @param elidx the index of the element to get the ndofs for 
         * @return the number of degrees of freedom 
         */
        [[nodiscard]] constexpr 
        auto ndof_el( index_type elidx ) const noexcept
        -> size_type 
        { return offsets[elidx + 1] - offsets[elidx]; }

        /** @brief get the number of elements represented in the map */
        [[nodiscard]] constexpr size_type nelem() const noexcept { return offsets.size() - 1; }

        /** @brief get the size of the global degree of freedom index space represented by this map */
        [[nodiscard]] constexpr size_type size() const noexcept { return static_cast<size_type>(offsets.back()); }
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
            std::vector<IDX> new_offsets(dofs.nelem() + 1);
            new_offsets[0] = 0;
            for(IDX ielem = 0; ielem < dofs.nelem(); ++ielem){
                new_offsets[ielem + 1] = new_offsets[ielem] + dofs.ndof_el(el_renumbering[ielem]);
            }
            return dof_map< IDX, ndim, conformity >(std::move(new_offsets));
        } else {
            IDX nelem = dofs.nelem();
            std::vector<IDX> newcols(nelem + 1);
            newcols[0] = 0;
            for(IDX ielem = 0; ielem < nelem; ++ielem) {
                newcols[ielem + 1] = newcols[ielem] 
                    + dofs.ndof_el(el_renumbering[ielem]);
            }
            util::crs<IDX, IDX> new_connectivity{std::span<const IDX>{newcols}};
            for(IDX ielem = 0; ielem < new_connectivity.nrow(); ++ielem){
                for(int icol = 0; icol < new_connectivity.rowsize(ielem); ++icol)
                    new_connectivity[ielem, icol] = 
                        dofs[el_renumbering[ielem], icol];
            }
            return dof_map< IDX, ndim, conformity >(dofs.size(), std::move(new_connectivity));
        }
    }

    /// Split a global dof set over a given element partition 
    /// @param el_part the partition of the elements 
    /// @param global_dofs the unpartitioned set of degrees of freedom to partition over the elements 
    ///     These dofs must match the numbering of the parallel indices of el_part
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

        dof_map<IDX, ndim, conformity> gdofs;

        if constexpr( conformity == l2_conformity(ndim) ){
            // === L2 dof conformity ===

            // communicate the global dofs map
            if(global_dofs.has_value()) {
                const auto& global_dofs_val = global_dofs.value();
                IDX nelem = global_dofs_val.nelem();
                gdofs = global_dofs.value();
#ifdef ICEICLE_USE_MPI 
                MPI_Bcast(&nelem, 1, mpi_get_type(nelem), myrank, MPI_COMM_WORLD);
                mpi::mpi_bcast_range(gdofs.offsets, myrank);
#endif
            } else {
#ifdef ICEICLE_USE_MPI
                IDX nelem;
                MPI_Bcast(&nelem, 1, mpi_get_type(nelem), global_dofs.valid_rank(), MPI_COMM_WORLD);
                std::vector<IDX> offsets(nelem + 1);
                mpi::mpi_bcast_range(offsets, global_dofs.valid_rank());
#endif
            }

            std::vector< std::vector<IDX> > rank_offsets(nrank, std::vector<IDX>{0});
            for(IDX iel_global = 0; iel_global < el_part.size(); ++iel_global){
                int irank = el_part.index_map[iel_global].rank;
                rank_offsets[iel_global].push_back(
                        rank_offsets[iel_global].back() + gdofs.ndof_el(iel_global));
            }
            
            std::vector< IDX > pindices(rank_offsets[myrank].back());
            std::iota(pindices.begin(), pindices.end(), 0);
            std::vector< IDX > owned_offsets{0};
            std::vector< IDX > renumbering;

            IDX pidx = 0;
            for(int irank = 0; irank < nrank; ++irank){
                IDX ndof_rank = rank_offsets[irank].back();
                for(IDX ielem = el_part.owned_offsets[irank]; ielem < el_part.owned_offsets[irank + 1]; ++ielem){
                    for(IDX i = 0; i < gdofs.ndof_el(ielem); ++i){
                        renumbering[pidx] = gdofs[ielem, i];
                        ++pidx;
                    }
                }
                owned_offsets.push_back(owned_offsets.back() + ndof_rank);
            }

            std::unordered_map<IDX, IDX> inv_pdofs{};
            for(int ldof = 0; ldof < pindices.size(); ++ldof){
                inv_pdofs[pindices[ldof]] = ldof;
            }

            return std::tuple{
                dof_map< IDX, ndim, conformity >{std::move(rank_offsets[myrank])},
                pindex_map{ pindices, inv_pdofs, owned_offsets },
                renumbering
            };
        } else {
            // === All other dof conformities ===
            std::vector< std::vector < IDX > > all_pdofs(nrank,
                    std::vector<IDX>{});
            std::vector<IDX> my_pdofs;

            IDX nelem, ndof;
            dof_map<IDX, ndim, conformity> gdofs;

            // communicate the global dofs map
            if(global_dofs.has_value()){
                const auto& global_dofs_val = global_dofs.value();
                nelem = global_dofs_val.nelem();
                ndof = global_dofs_val.size();
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
            for(IDX iel : el_part.owned_offsets){
                for(IDX pdof : gdofs.rowspan(iel)){
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

            // reorder the ldofs to satisfy the requirements for p_indices of pindex_map
            std::ranges::sort(my_pdofs);
            IDX start_owned;
            for(IDX ldof = 0; ldof < my_pdofs.size(); ++ldof){
                if(my_pdofs[ldof] >= offsets[mpi::mpi_world_rank()]){
                    start_owned = ldof;
                    break;
                }
            }
            IDX end_owned = start_owned +
                (offsets[mpi::mpi_world_rank() + 1] - offsets[mpi::mpi_world_rank()]);
            {
                std::vector<IDX> pdofs_rearrange{};
                pdofs_rearrange.insert(pdofs_rearrange.end(), 
                        my_pdofs.begin() + start_owned, my_pdofs.begin() + end_owned);
                pdofs_rearrange.insert(pdofs_rearrange.end(), 
                        my_pdofs.begin(), my_pdofs.begin() + start_owned);
                pdofs_rearrange.insert(pdofs_rearrange.end(), 
                        my_pdofs.begin() + end_owned, my_pdofs.end());
                my_pdofs = std::move(pdofs_rearrange);
            }
           
            // create a inverse mapping of my_pdofs to ldofs
            std::unordered_map<IDX, IDX> inv_pdofs{};
            for(int ldof = 0; ldof < my_pdofs.size(); ++ldof){
                inv_pdofs[my_pdofs[ldof]] = ldof;
            }

            // setup the cols array for number of process local elements
            IDX nelem_local = el_part.p_indices.size();
            std::vector<IDX> ldof_cols(nelem_local + 1);
            // first count up the number of dofs for each element
            ldof_cols[0] = 0;
            for(IDX ielem_p : el_part.owned_pindex_range(myrank)){
                IDX ielem_local = el_part.inv_p_indices.at(ielem_p);
                ldof_cols[ielem_local + 1] = gdofs.ndof_el(ielem_p);
            }
            // then accumulate
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
                pindex_map{ my_pdofs, inv_pdofs, offsets },
                renumbering
            };
        }
    }
}
