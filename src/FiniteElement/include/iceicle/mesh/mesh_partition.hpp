#include "iceicle/anomaly_log.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/geometry/transformations_table.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <mpi.h>
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
            std::unordered_map<IDX, IDX> inv_p_indices;
            for(IDX lindex = 0; lindex < el_p_idxs.size(); ++lindex)
                inv_p_indices[el_p_idxs[lindex]] = lindex;
            std::vector< IDX > offsets = {0, nelem + 1};
            std::vector<IDX> renumbering(nelem);
            std::iota(renumbering.begin(), renumbering.end(), 0);
            return std::pair{pindex_map{std::move(ret), std::move(el_p_idxs),
                    std::move(inv_p_indices), std::move(offsets)},
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
            if(irank == myrank){
                for(IDX lindex = 0; lindex < nelem_rank; ++lindex) {
                    IDX pindex = lindex + pindex_start;
                    p_indices[lindex] = pindex;
                }
            }
            for(IDX lindex = 0; lindex < nelem_rank; ++lindex) {
                IDX pindex = lindex + pindex_start;
                index_map[pindex] = p_index<IDX>{.rank = irank, .index = lindex};
            }
            pindex_start += nelem_rank;
            owned_offsets[irank + 1] = pindex_start;
        }

        std::unordered_map<IDX, IDX> inv_p_indices;
        for(IDX lindex = 0; lindex < p_indices.size(); ++lindex)
            inv_p_indices[p_indices[lindex]] = lindex;

        return std::pair{
            pindex_map<IDX>{std::move(index_map), std::move(p_indices),
                std::move(inv_p_indices), std::move(owned_offsets) },
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
        std::cout << el_partition;
        fmt::println("{}", el_renumbering);
        dof_map el_conn_global{apply_el_renumbering(mesh.conn_el, el_renumbering)};
        auto [p_el_conn, p_el_conn_map, nodes_renumbering] =
            partition_dofs<IDX, ndim, h1_conformity(ndim)>(el_partition,
                    mpi::on_rank{el_conn_global, 0});
        std::vector<IDX> inv_nodes_renumbering(nodes_renumbering.size());
        for(IDX inode = 0; inode < nodes_renumbering.size(); ++inode){
            inv_nodes_renumbering[nodes_renumbering[inode]] = inode;
        }

        std::cout << p_el_conn_map;
        // construct the nodes
        NodeArray<T, ndim> p_coord{};
        p_coord.reserve(p_el_conn_map.n_lindex());
        for(IDX inode : p_el_conn_map.p_indices) {
            p_coord.push_back(gcoord[nodes_renumbering[inode]]);
        }
        for(int irank = 0; irank < nrank; ++irank){
            if(irank == myrank){
                fmt::println("p_coord rank {}: {}", irank, p_coord);
            }
            mpi::mpi_sync();
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
                        IDX iel_global = el_renumbering[
                            el_partition.owned_offsets[irank] + iel_local];
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

        for(int irank = 0; irank < nrank; ++irank){
            if(irank == myrank){
                std::cout << "p_el_conn " << irank << ": " << std::endl;
                std::cout << p_el_conn.dof_connectivity;
            }
            mpi::mpi_sync();
        }

        using boundary_face_desc = AbstractMesh<T, IDX, ndim>::boundary_face_desc;
        std::vector<boundary_face_desc> boundary_descs;
        // Create boundary face descriptions of boundary faces
        if(myrank == 0){
            for(IDX ifac = mesh.bdyFaceStart; ifac < mesh.bdyFaceEnd; ++ifac){
                const Face<T, IDX, ndim>& face = *(mesh.faces[ifac]);
                IDX attached_element = el_renumbering[face.elemL];
                int mpi_rank = el_partition.index_map[attached_element].rank;
                
                BOUNDARY_CONDITIONS bctype = face.bctype;
                int bcflag = face.bcflag;
                std::vector<IDX> pnodes{};
                pnodes.reserve(face.n_nodes());
                for(IDX node : face.nodes_span()){
                    pnodes.push_back(inv_nodes_renumbering[node]);
                }

                if(mpi_rank == myrank) {
                    std::vector<IDX> nodes{};
                    nodes.reserve(face.n_nodes());
                    for(IDX node : pnodes)
                        nodes.push_back(p_el_conn_map.inv_p_indices[node]);
                    boundary_descs.push_back(boundary_face_desc{bctype, bcflag, nodes});
                } else {
                    int bctype_int = (int) bctype;
                    std::size_t nnode = pnodes.size();
                    // send bctype, bcflag, number of nodes, then nodes array
                    MPI_Send(&bctype_int, 1, MPI_INT, mpi_rank, 0, MPI_COMM_WORLD);
                    MPI_Send(&bcflag, 1, MPI_INT, mpi_rank, 1, MPI_COMM_WORLD);
                    MPI_Send(&nnode, 1, mpi_get_type<std::size_t>(), mpi_rank, 2, MPI_COMM_WORLD);
                    MPI_Send(pnodes.data(), pnodes.size(), mpi_get_type(pnodes.data()), mpi_rank, 3, MPI_COMM_WORLD);
                }
            }

            for(int irank = 1; irank < nrank; ++irank){
                int stop_code = -1;
                MPI_Send(&stop_code, 1, MPI_INT, irank, 0, MPI_COMM_WORLD);
            }
        } else {
            int bctype_int;
            MPI_Recv(&bctype_int, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            while(bctype_int != -1){
                BOUNDARY_CONDITIONS bctype = static_cast<BOUNDARY_CONDITIONS>(bctype_int);
                int bcflag, nnode;
                MPI_Recv(&bcflag, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&nnode, 1, mpi_get_type<std::size_t>(), 0, 2, MPI_COMM_WORLD, &status);
                std::vector<IDX> pnodes(nnode);
                MPI_Recv(pnodes.data(), nnode, mpi_get_type(pnodes.data()), 0, 3, MPI_COMM_WORLD, &status);
                // translate to local node indices
                std::vector<IDX> nodes{};
                nodes.reserve(pnodes.size());
                for(IDX node : pnodes)
                    nodes.push_back(p_el_conn_map.inv_p_indices[node]);

                boundary_descs.push_back(boundary_face_desc{bctype, bcflag, nodes});

                // get the next bc type
                MPI_Recv(&bctype_int, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }
        }

        std::vector< std::unique_ptr< Face<T, IDX, ndim> > > interprocess_faces{};
        // Create boundary face descriptions for inter-process faces
        if(myrank == 0){

            auto send_face = [myrank, &el_partition, &p_el_conn_map, &interprocess_faces](
                DOMAIN_TYPE face_domain_type,
                DOMAIN_TYPE left_domain_type,
                DOMAIN_TYPE right_domain_type,
                int face_order,
                IDX iel_p,
                IDX ier_p,
                int face_nr_l,
                int face_nr_r,
                int orientation,
                std::vector<IDX> pnodes,
                int face_rank,
                int neighbor_rank,
                bool left
            ) -> void{
                if(face_rank == myrank){
                    // The face we want to make is on rank 0, just make it 

                    // translate the element that we own
                    IDX iel = (left) ? el_partition.inv_p_indices[iel_p] : iel_p;
                    IDX ier = (left) ? ier_p : el_partition.inv_p_indices[ier_p];

                    // translate to local node indices
                    std::vector<IDX> nodes{};
                    nodes.reserve(pnodes.size());
                    for(IDX node : pnodes)
                        nodes.push_back(p_el_conn_map.inv_p_indices[node]);
                    auto fac_opt = make_face<T, IDX, ndim>(face_domain_type, left_domain_type, right_domain_type,
                            face_order, iel, ier, nodes, face_nr_l, face_nr_r, orientation,
                            BOUNDARY_CONDITIONS::PARALLEL_COM, encode_mpi_bcflag(neighbor_rank, left));
                        if(fac_opt)
                            interprocess_faces.push_back(std::move(fac_opt.value()));
                        else 
                            util::AnomalyLog::log_anomaly("Cannot form boundary face");
                    
                } else {
                    // send face geometry specifications
                    int face_domn_int = (int) face_domain_type, 
                        left_domain_int = (int) left_domain_type,
                        right_domain_int = (int) right_domain_type;
                    MPI_Send(&face_domn_int, 1, MPI_INT, face_rank, 1, MPI_COMM_WORLD);
                    MPI_Send(&left_domain_int, 1, MPI_INT, face_rank, 2, MPI_COMM_WORLD);
                    MPI_Send(&right_domain_int, 1, MPI_INT, face_rank, 3, MPI_COMM_WORLD);
                    MPI_Send(&face_order, 1, MPI_INT, face_rank, 4, MPI_COMM_WORLD);

                    // send the parallel element indices
                    MPI_Send(&iel_p, 1, mpi_get_type(iel_p), face_rank, 5, MPI_COMM_WORLD);
                    MPI_Send(&ier_p, 1, mpi_get_type(ier_p), face_rank, 6, MPI_COMM_WORLD);

                    // send the face nodes
                    std::size_t nnode = pnodes.size();
                    MPI_Send(&nnode, 1, mpi_get_type<std::size_t>(), face_rank, 7, MPI_COMM_WORLD);
                    MPI_Send(pnodes.data(), pnodes.size(), mpi_get_type(pnodes.data()), face_rank, 8, MPI_COMM_WORLD);

                    // send face numbers and orientation 
                    MPI_Send(&face_nr_l, 1, MPI_INT, face_rank, 9, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, face_rank, 10, MPI_COMM_WORLD);
                    MPI_Send(&orientation, 1, MPI_INT, face_rank, 11, MPI_COMM_WORLD);

                    // send the bcflag
                    int bcflag = encode_mpi_bcflag(neighbor_rank, left);
                    MPI_Send(&bcflag, 1, MPI_INT, face_rank, 12, MPI_COMM_WORLD);
                }
            };

            for(IDX iface = mesh.interiorFaceStart; iface < mesh.interiorFaceEnd; ++iface){
                Face<T, IDX, ndim>& face = *(mesh.faces[iface]);
                IDX iel = el_renumbering[face.elemL];
                IDX ier = el_renumbering[face.elemR];
                int rank_l = el_partition.index_map[iel].rank;
                int rank_r = el_partition.index_map[ier].rank;
                if(rank_l != rank_r){
                    // create the nodes list
                    std::vector<IDX> pnodes;
                    pnodes.reserve(face.n_nodes());
                    for(IDX node : face.nodes_span()){
                        pnodes.push_back(inv_nodes_renumbering[node]);
                    }
                    // send left
                    send_face(face.domain_type(), 
                            mesh.el_transformations[face.elemL]->domain_type,
                            mesh.el_transformations[face.elemR]->domain_type,
                            face.geometry_order(),
                            el_renumbering[face.elemL],
                            el_renumbering[face.elemR],
                            face.face_nr_l(),
                            face.face_nr_r(),
                            face.orientation_r(),
                            pnodes,
                            rank_l,
                            rank_r,
                            true
                    );
                    // send right
                    send_face(face.domain_type(), 
                            mesh.el_transformations[face.elemL]->domain_type,
                            mesh.el_transformations[face.elemR]->domain_type,
                            face.geometry_order(),
                            el_renumbering[face.elemL],
                            el_renumbering[face.elemR],
                            face.face_nr_l(),
                            face.face_nr_r(),
                            face.orientation_r(),
                            pnodes,
                            rank_r,
                            rank_l,
                            false
                    );

                }
            }

             // send stop codes to all processes
            for(int irank = 1; irank < nrank; ++irank){
                int stop_code = -1;
                MPI_Send(&stop_code, 1, MPI_INT, irank, 1, MPI_COMM_WORLD);
            }
        } else {
            // not rank 0: recieve face information

            auto recieve_face = [myrank, &el_partition, &p_el_conn_map, &interprocess_faces]()
            -> bool {
                constexpr int root_rank = 0;
                MPI_Status status;
                // Get the face domain type or a stop code
                int face_domn_int, left_domain_int, right_domain_int, face_order;
                MPI_Recv(&face_domn_int, 1, MPI_INT, root_rank, 1, MPI_COMM_WORLD, &status);
                if(face_domn_int == -1) {
                    return false; // stop code
                } 

                MPI_Recv(&left_domain_int, 1, MPI_INT, root_rank, 2, MPI_COMM_WORLD, &status);
                MPI_Recv(&right_domain_int, 1, MPI_INT, root_rank, 3, MPI_COMM_WORLD, &status);
                MPI_Recv(&face_order, 1, MPI_INT, root_rank, 4, MPI_COMM_WORLD, &status);

                // parallel element indices
                IDX iel_p, ier_p;
                MPI_Recv(&iel_p, 1, mpi_get_type(iel_p), root_rank, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&ier_p, 1, mpi_get_type(ier_p), root_rank, 6, MPI_COMM_WORLD, &status);

                // face nodes
                std::size_t nnode;
                MPI_Recv(&nnode, 1, mpi_get_type(nnode), root_rank, 7, MPI_COMM_WORLD, &status);
                std::vector<IDX> pnodes(nnode);
                MPI_Recv(pnodes.data(), nnode, mpi_get_type(pnodes.data()), 
                        root_rank, 8, MPI_COMM_WORLD, &status);

                // face numbers and orientation
                int face_nr_l, face_nr_r, orientation;
                MPI_Recv(&face_nr_l, 1, MPI_INT, root_rank, 9, MPI_COMM_WORLD, &status);
                MPI_Recv(&face_nr_r, 1, MPI_INT, root_rank, 10, MPI_COMM_WORLD, &status);
                MPI_Recv(&orientation, 1, MPI_INT, root_rank, 11, MPI_COMM_WORLD, &status);

                // bcflag
                int bcflag;
                MPI_Recv(&bcflag, 1, MPI_INT, root_rank, 12, MPI_COMM_WORLD, &status);

                auto [neighbor_rank, left] = decode_mpi_bcflag(bcflag);

                // translate the element that we own
                IDX iel = (left) ? el_partition.inv_p_indices[iel_p] : iel_p;
                IDX ier = (left) ? ier_p : el_partition.inv_p_indices[ier_p];

                // translate to local node indices
                std::vector<IDX> nodes{};
                nodes.reserve(pnodes.size());
                for(IDX node : pnodes)
                    nodes.push_back(p_el_conn_map.inv_p_indices[node]);
                auto fac_opt = make_face<T, IDX, ndim>(
                        static_cast<DOMAIN_TYPE>(face_domn_int), 
                        static_cast<DOMAIN_TYPE>(left_domain_int), 
                        static_cast<DOMAIN_TYPE>(right_domain_int), 
                        face_order, iel, ier, nodes, face_nr_l, face_nr_r, orientation,
                        BOUNDARY_CONDITIONS::PARALLEL_COM, bcflag);
                    if(fac_opt)
                        interprocess_faces.push_back(std::move(fac_opt.value()));
                    else 
                        util::AnomalyLog::log_anomaly("Cannot form boundary face");

                // construct the face
                return true;
            };

            while( recieve_face() ) {}
        }

        // create the mesh
        pmesh = std::move(AbstractMesh{
                    p_coord, p_el_conn, el_transforms, 
                    boundary_descs, interprocess_faces});


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
