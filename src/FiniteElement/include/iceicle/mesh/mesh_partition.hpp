#include "iceicle/anomaly_log.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include <execution>
#include <iceicle/mesh/mesh.hpp>
#ifdef ICEICLE_USE_METIS
#include <metis.h>
#ifdef ICEICLE_USE_MPI 
#include "iceicle/mpi_type.hpp"
#include <iceicle/iceicle_mpi_utils.hpp>

#ifdef ICEICLE_USE_PETSC
#include <petsclog.h>
#endif

namespace iceicle {

    /// @brief partition a set of element indices using METIS 
    /// @param elsuel the ELements SUrrounding ELements connectivity 
    /// the indices of all the elements that share a face with a given element
    ///
    /// Indices are renumbered so pindices are contiguous for each process rank
    ///
    /// @return the parallel index map and a std::vector of indices that cointains
    /// old global index at each new global index (the renumbering vector)
    template< class IDX, class IDX_CRS >
    [[nodiscard]]
    auto partition_elements( util::crs<IDX, IDX_CRS>& elsuel )
    -> std::pair<pindex_map<IDX>, std::vector<IDX>> {
        util::crs elsuel_metis_int{util::convert_crs<idx_t, idx_t>(elsuel)};
        idx_t nelem = elsuel.nrow();

        // get mpi information
        int nrank, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nrank);

        // metis will floating point exception when partitioning into 1 partition
        // ...
        // ...
        //
        // :3
        if(nrank == 1) {
            std::vector< p_index<IDX> > ret{};
            for(IDX ielem = 0; ielem < nelem; ++ielem){
                ret.emplace_back(0, ielem);
            }
            std::vector< IDX > el_p_idxs(nelem);
            std::iota(el_p_idxs.begin(), el_p_idxs.end(), 0);
            std::vector< IDX > offsets = {0, nelem + 1};
            std::vector<IDX> renumbering(nelem);
            std::iota(renumbering.begin(), renumbering.end(), 0);
            return std::pair{pindex_map{std::move(ret), std::move(el_p_idxs), std::move(offsets)},
                std::move(renumbering)};
        } 

        std::vector<idx_t> el_partition(nelem);
        // only the first processor will perform the partitioning
        if(myrank == 0) {
            // number of balancing constraints (not specifying so leave at 1)
            IDX ncon = 1;

            // set the options
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_NUMBERING] = 0;

            idx_t obj_val; // the objective value (edge-cut) of the partitioning scheme

            METIS_PartGraphKway(&nelem, &ncon, elsuel_metis_int.cols(), elsuel_metis_int.data(),
                    NULL, NULL, NULL, &nrank, NULL, NULL,
                    options, &obj_val, el_partition.data());
        } 
        // broadcast the completed partitioning
        mpi::mpi_bcast_range(el_partition, 0);

        // form the renumbering
        std::vector<std::vector<IDX>> rank_pindices(nrank);
        for(IDX ielem = 0; ielem < nelem; ++ielem){
            rank_pindices[el_partition[ielem]].push_back(ielem);
        }
        std::vector<IDX> renumbering{};
        std::vector< p_index<IDX> > index_map(nelem);
        std::vector< IDX > p_indices(rank_pindices[myrank].size());
        std::vector< IDX > owned_offsets(nrank + 1);
        owned_offsets[0] = 0;
        renumbering.reserve(nelem);

        IDX pindex_start = 0;
        for(int irank = 0; irank < nrank; ++irank){
            IDX nelem_rank = rank_pindices[irank].size();
            renumbering.insert(renumbering.end(),
                    rank_pindices[irank].begin(),rank_pindices[irank].end());
            for(IDX lindex = 0; lindex < nelem_rank; ++lindex) {
                IDX pindex = lindex + pindex_start;
                index_map[pindex] = p_index<IDX>{.rank = irank, .index = lindex};
                p_indices[lindex] = pindex;
            }
            pindex_start += nelem_rank;
            owned_offsets[irank + 1] = pindex_start;
        }

        return std::pair{
            pindex_map<IDX>{std::move(index_map), 
                std::move(p_indices), std::move(owned_offsets) },
            renumbering
        };
    }

    /// @brief partition the mesh using METIS 
    /// @param mesh the single processor mesh to partition
    /// NOTE: assumes valid mesh is on rank 0, other ranks can be empty or incomplete
    ///
    /// @return the partitioned mesh on each processor 
    template<class T, class IDX, int ndim>
    auto partition_mesh(AbstractMesh<T, IDX, ndim>& mesh) 
    -> AbstractMesh<T, IDX, ndim>
    {
        // get mpi information
        int nrank = mpi::mpi_world_size(), myrank = mpi::mpi_world_rank();

        // metis will floating point exception when partitioning into 1 partition
        // ...
        // ...
        //
        // :3
        if(nrank == 1) {
            return mesh;
        }

        /// @brief create an empty mesh -- this will be the partitioned mesh
        AbstractMesh<T, IDX, ndim> pmesh{};

        // Broadcast all the node data 
        NodeArray<T, ndim> gcoord{};
        unsigned long n_nodes_total;
        if(myrank == 0)
        { 
            gcoord = mesh.coord;
            n_nodes_total = mesh.coord.size();
        }
        MPI_Bcast(&n_nodes_total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if(myrank != 0)
        { gcoord.resize(n_nodes_total); }

        // WARNING: assumption about size of Point type
        MPI_Bcast(gcoord[0].data(), gcoord.size() * ndim, mpi_get_type<T>(), 0, MPI_COMM_WORLD);

        // global node index for each local node
        std::vector<long unsigned int> gnode_idxs;

        // local node index for each global node index;
        std::vector<IDX> inv_gnode_idxs(n_nodes_total, -1);

        // for each global element index, store the local index, or -1
        std::vector<IDX> inv_gel_idxs(mesh.nelem(), -1);

        IDX nelem = mesh.nelem();
        util::crs elsuel{util::convert_crs<idx_t, idx_t>(to_elsuel<T, IDX, ndim>(nelem, mesh.faces))};

        // ====================================
        // = Generate the Partitioned Meshes  =
        // ====================================
        MPI_Status status;

        auto [el_partition, el_renumbering] = partition_elements(elsuel);
        dof_map el_conn_global{apply_el_renumbering(mesh.conn_el, el_renumbering)};
        auto [p_el_conn, p_el_conn_map, nodes_renumbering] =
            partition_dofs<IDX, ndim, h1_conformity(ndim)>(el_partition,
                    mpi::on_rank{el_conn_global, 0});

        // construct the nodes
        NodeArray<T, ndim> p_coord(p_el_conn.size());
        for(IDX inode = 0; inode < p_coord.size(); ++inode) {
            p_coord[inode] = gcoord[nodes_renumbering[inode]];
        }

        std::vector< ElementTransformation<T, IDX, ndim>* > el_transforms{};
        { // element transformations 
            // take care of rank 0 first 
            if(myrank == 0){
                el_transforms.reserve(el_partition.index_map.size());
                for(IDX iel : el_partition.p_indices) {
                    el_transforms.push_back(mesh.el_transformations[el_renumbering[iel]]);
                }
            }

            for(int irank = 1; irank < nrank; ++irank){
                if(myrank == 0){
                    IDX nelem_rank = el_partition.owned_range_size(irank);
                    // send transformation descriptions
                    std::vector<int> el_domains(nelem_rank);
                    std::vector<int> el_orders(nelem_rank);
                    for(IDX iel_local = 0; iel_local < nelem_rank; ++iel_local){
                        IDX iel_global = el_renumbering[el_partition.p_indices[iel_local]];
                        el_domains[iel_local] = static_cast<int>(mesh.el_transformations[iel_global]->domain_type);
                        el_orders[iel_local] = mesh.el_transformations[iel_global]->order;
                    }
                    MPI_Send( el_domains.data(), nelem_rank, MPI_INT, irank, 0, MPI_COMM_WORLD );
                    MPI_Send( el_orders.data(), nelem_rank, MPI_INT, irank, 0, MPI_COMM_WORLD );
                    
                } else if(myrank == irank) {
                    IDX nelem_rank = el_partition.owned_range_size(irank);
                    // recieve tansformation descriptions
                    std::vector<int> el_domains(nelem_rank);
                    std::vector<int> el_orders(nelem_rank);
                    MPI_Recv( el_domains.data(), nelem_rank, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
                    MPI_Recv( el_orders.data(), nelem_rank, MPI_INT, 0, 0, MPI_COMM_WORLD, &status );
                    for(IDX iel_local = 0; iel_local < nelem_rank; ++iel_local){
                        el_transforms.push_back(transformation_table<T, IDX, ndim>.get_transform(
                                (DOMAIN_TYPE) el_domains[iel_local], el_orders[iel_local]));
                    }
                }
            }
        }

        using boundary_face_desc = AbstractMesh<T, IDX, ndim>::boundary_face_desc;
        pmesh = std::move(AbstractMesh{
                    p_coord, p_el_conn, el_transforms, 
                    std::vector<boundary_face_desc>{} });


#ifndef NDEBUG 
        for(int i = 0; i < nrank; ++i){
            MPI_Barrier(MPI_COMM_WORLD);
            if(i == myrank){
                std::cout << "====== Mesh " << myrank << " ======" << std::endl;
                pmesh.printNodes(std::cout);
                pmesh.printElements(std::cout);
                pmesh.printFaces(std::cout);
                std::cout << std::endl;
            }
        }
#endif

        for(int i = 0; i < nrank; ++i){
            MPI_Barrier(MPI_COMM_WORLD);
            if(i == myrank){
                if(util::AnomalyLog::size() > 0) {
                    std::cout << "====== Errors on rank: " << myrank << " ======" << std::endl;
                    util::AnomalyLog::handle_anomalies();
                    // clear out the erronous mesh
                    pmesh = AbstractMesh<T, IDX, ndim>{};
                }
            }
        }
        return pmesh;

    }

}


#endif // ICEICLE_USE_MPI

#else // No metis 
    namespace iceicle {
        template<class T, class IDX, int ndim>
        auto partition_mesh(const AbstractMesh<T, IDX, ndim>& mesh) 
        -> const AbstractMesh<T, IDX, ndim>&
        {
            return mesh;
        }
    } 
#endif
