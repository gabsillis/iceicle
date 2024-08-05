#include "iceicle/anomaly_log.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/face_utils.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/mpi_type.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/mpi_type.hpp>
#include <mpi_proto.h>
#include <petsclog.h>
#ifdef ICEICLE_USE_METIS
#include <metis.h>
#ifdef ICEICLE_USE_MPI 
#include <mpi.h>

namespace iceicle {


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
        int nproc, iproc;
        MPI_Comm_rank(MPI_COMM_WORLD, &iproc);
        MPI_Comm_size(MPI_COMM_WORLD, &nproc);

        // metis will floating point exception when partitioning into 1 partition
        // ...
        // ...
        //
        // :3
        if(nproc == 1) return mesh;

        /// @brief create an empty mesh -- this will be the partitioned mesh
        AbstractMesh<T, IDX, ndim> pmesh{};

        // Broadcast all the node data 
        NodeArray<T, ndim> gcoord{};
        unsigned long n_nodes_total;
        if(iproc == 0)
        { 
            gcoord = mesh.nodes;
            n_nodes_total = mesh.nodes.size();
        }
        MPI_Bcast(&n_nodes_total, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if(iproc != 0)
        { gcoord.resize(n_nodes_total); }

        // WARNING: assumption about size of Point type
        MPI_Bcast(gcoord[0].data(), gcoord.size() * ndim, mpi_get_type<T>(), 0, MPI_COMM_WORLD);

        // only the first processor will perform the partitioning
        if(iproc == 0){
            IDX nelem = mesh.elements.size();
            util::crs elsuel{util::convert_crs<idx_t, idx_t>(to_elsuel<T, IDX, ndim>(nelem, mesh.faces))};

            // number of balancing constraints (not specifying so leave at 1)
            IDX ncon = 1;

            // set the options
            idx_t options[METIS_NOPTIONS];
            METIS_SetDefaultOptions(options);
            options[METIS_OPTION_NUMBERING] = 0;

            idx_t obj_val; // the objective value (edge-cut) of the partitioning scheme
            std::vector<idx_t> el_partition(nelem);

            METIS_PartGraphKway(&nelem, &ncon, elsuel.cols(), elsuel.data(),
                    NULL, NULL, NULL, &nproc, NULL, NULL,
                    options, &obj_val, el_partition.data());

            // ====================================
            // = Generate the Partitioned Meshes  =
            // ====================================

            // === Step 1: Node List ===

            std::vector<std::vector<unsigned long>> node_idxs(nproc, std::vector<unsigned long>{});

            // build each processor node list
            for(IDX iel = 0; iel < nelem; ++iel){
                IDX element_rank = el_partition[iel];
                for(IDX inode : mesh.elements[iel]->nodes_span()){
                    node_idxs[element_rank].push_back(inode);
                }
            }

            // send the node lists to each processor 
            for(int irank = 1; irank < nproc; ++irank){
                // send the size 
                unsigned long sz = node_idxs[irank].size();
                MPI_Send(&sz, 1, MPI_UNSIGNED_LONG, irank, 0, MPI_COMM_WORLD);
                MPI_Send(node_idxs[irank].data(), sz, MPI_UNSIGNED_LONG, irank, 0, MPI_COMM_WORLD);
            }

            std::vector<unsigned long>& my_node_idxs{node_idxs[0]};
            std::ranges::sort(my_node_idxs);
            auto unique_subrange = std::ranges::unique(my_node_idxs);
            my_node_idxs.erase(unique_subrange.begin(), unique_subrange.end());

            std::vector<IDX> inv_gnode_idxs(n_nodes_total, -1);
            for(IDX ilocal = 0; ilocal < my_node_idxs.size(); ++ilocal)
                { inv_gnode_idxs[my_node_idxs[ilocal]] = ilocal; }


            pmesh.nodes.reserve(my_node_idxs.size());
            for(unsigned long inode : my_node_idxs)
                { pmesh.nodes.push_back(gcoord[inode]); }

            // === Step 2: Elements ===

            // for each global element index, store the local index, or -1
            std::vector<IDX> inv_gel_idxs(mesh.elements.size(), -1);

            for(IDX iel = 0; iel < nelem; ++iel){ 
                IDX element_rank = el_partition[iel];
                if(element_rank == 0){
                    GeometricElement<T, IDX, ndim> &el = *(mesh.elements[iel]);
                    inv_gel_idxs[iel] = pmesh.elements.size();
                    std::vector<IDX> elnodes(el.n_nodes());
                    for(int inode = 0; inode < el.n_nodes(); ++inode){
                        elnodes[inode] = inv_gnode_idxs[el.nodes()[inode]];
                    }
                    pmesh.elements.push_back(create_element<T, IDX, ndim>(el.domain_type(), el.geometry_order(), elnodes));
                } else {
                    GeometricElement<T, IDX, ndim> &el = *(mesh.elements[iel]);

                    int domain_type = static_cast<int>(el.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, element_rank, 1, MPI_COMM_WORLD);

                    MPI_Send(&iel, 1, mpi_get_type<IDX>(), element_rank, 0, MPI_COMM_WORLD);

                    int geo_order = el.geometry_order();
                    MPI_Send(&geo_order, 1, MPI_INT, element_rank, 0, MPI_COMM_WORLD);

                    int n_nodes = el.n_nodes();
                    MPI_Send(&n_nodes, 1, MPI_INT, element_rank, 0, MPI_COMM_WORLD);

                    MPI_Send(el.nodes(), n_nodes, mpi_get_type<IDX>(), element_rank, 0, MPI_COMM_WORLD);
                }
            }

            // send signal to stop to each process (domain type = N_DOMAIN_TYPES)
            for(int iproc = 1; iproc < nproc; ++iproc)
            {
                int stop_signal = (int) DOMAIN_TYPE::N_DOMAIN_TYPES;
                MPI_Send(&stop_signal, 1, MPI_INT, iproc, 1, MPI_COMM_WORLD);
            }

            // === Step 3: Faces ===
            MPI_Status status;
            for(IDX iface = mesh.interiorFaceStart; iface < mesh.interiorFaceEnd; ++iface) {
                Face<T, IDX, ndim>& face = *(mesh.faces[iface]);

                IDX iel = face.elemL;
                IDX ier = face.elemR;
                int rank_l = el_partition[iel];
                int rank_r = el_partition[ier];

                if(rank_l == 0) {
                    if(rank_r == 0){
                        // Easy case: internal face on this process
                        IDX iel_parallel = inv_gel_idxs[iel];
                        IDX ier_parallel = inv_gel_idxs[ier];

                        auto face_parallel = make_face(iel_parallel, ier_parallel, 
                            pmesh.elements[iel_parallel].get(), pmesh.elements[ier_parallel].get());

                        if(face_parallel.has_value()){
                            pmesh.faces.push_back(std::move(face_parallel.value()));
                        } else {
                            util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                            return pmesh;
                        }
                    } else {
                        // boundary with another process
                        int rank_other = rank_r;
                        // send all the information needed for them to make the face 
                        int domain_type = static_cast<int>(face.domain_type());
                        MPI_Send(&domain_type, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                        // send the rank of the left element and reight element 
                        MPI_Send(&rank_l, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                        MPI_Send(&rank_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                        // exchange local element index 
                        IDX iel_local = inv_gnode_idxs[iel];
                        IDX ier_local;
                        MPI_Sendrecv(
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 1,
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 1, 
                            MPI_COMM_WORLD, &status
                        );

                        // send the geometry order 
                        int geo_order = face.geometry_order();
                        MPI_Send(&geo_order, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                        // send the face nodes 
                        int n_face_nodes = face.n_nodes();
                        MPI_Send(&n_face_nodes, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                        MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_other, 0, MPI_COMM_WORLD);

                        // send the face numbers and orientation 
                        int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                        int orient_r = face.orientation_r();
                        MPI_Send(&face_nr_l, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                        MPI_Send(&orient_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                        auto face_opt = make_face<T, IDX, ndim>(
                            static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                            std::span{face.nodes(), (std::size_t) n_face_nodes}, face_nr_l, face_nr_r, orient_r,
                            BOUNDARY_CONDITIONS::PARALLEL_COM, rank_other
                        );

                        if(face_opt) {
                            pmesh.faces.push_back(std::move(face_opt.value()));
                        } else {
                            util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                            return pmesh;
                        }
                    }

                } else if(rank_r == 0){
                    // rank_l == 0 is already covered
                    // boundary with another process
                    int rank_other = rank_l;
                    // send all the information needed for them to make the face 
                    int domain_type = static_cast<int>(face.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                    // send the rank of the left element and reight element 
                    MPI_Send(&rank_l, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                    MPI_Send(&rank_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                    // exchange local element index 
                    IDX iel_local;
                    IDX ier_local = inv_gnode_idxs[iel];
                    MPI_Sendrecv(
                        &ier_local, 1, mpi_get_type<IDX>(), rank_other, 1,
                        &iel_local, 1, mpi_get_type<IDX>(), rank_other, 1, 
                        MPI_COMM_WORLD, &status
                    );

                    // send the geometry order 
                    int geo_order = face.geometry_order();
                    MPI_Send(&geo_order, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                    // send the face nodes 
                    int n_face_nodes = face.n_nodes();
                    MPI_Send(&n_face_nodes, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                    MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_other, 0, MPI_COMM_WORLD);

                    // send the face numbers and orientation 
                    int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                    int orient_r = face.orientation_r();
                    MPI_Send(&face_nr_l, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);
                    MPI_Send(&orient_r, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                    auto face_opt = make_face<T, IDX, ndim>(
                        static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                        std::span{face.nodes(), (std::size_t) n_face_nodes}, face_nr_l, face_nr_r, orient_r,
                        BOUNDARY_CONDITIONS::PARALLEL_COM, -rank_other // NOTE: negative rank because we are right
                    );

                    if(face_opt) {
                        pmesh.faces.push_back(std::move(face_opt.value()));
                    } else {
                        util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                        return pmesh;
                    }
                } else {
                    // both ranks are on another process 

                    // send all the information needed for them to make the face 
                    int domain_type = static_cast<int>(face.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&domain_type, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);

                    // send the rank of the left element and reight element 
                    MPI_Send(&rank_l, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&rank_r, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&rank_l, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);
                    MPI_Send(&rank_r, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);

                    // send the geometry order 
                    int geo_order = face.geometry_order();
                    MPI_Send(&geo_order, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&geo_order, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);

                    // send the face nodes 
                    int n_face_nodes = face.n_nodes();
                    MPI_Send(&n_face_nodes, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&n_face_nodes, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);
                    MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_r, 0, MPI_COMM_WORLD);

                    // send the face numbers and orientation 
                    int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                    int orient_r = face.orientation_r();
                    MPI_Send(&face_nr_l, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&orient_r, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_l, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);
                    MPI_Send(&orient_r, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);
                }
            }

            // send signal to stop to each process (domain type = N_DOMAIN_TYPES)
            for(int iproc = 1; iproc < nproc; ++iproc)
            {
                int stop_signal = (int) DOMAIN_TYPE::N_DOMAIN_TYPES;
                MPI_Send(&stop_signal, 1, MPI_INT, iproc, 1, MPI_COMM_WORLD);
            }

        } else {
            // ====================================
            // = Generate the Partitioned Meshes  =
            // ====================================

            // === Step 1: Node List ===

            MPI_Status status;
            unsigned long n_nodes;
            MPI_Recv(&n_nodes, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
            std::vector<unsigned long> node_idxs(n_nodes);
            MPI_Recv(node_idxs.data(), n_nodes, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);

            std::ranges::sort(node_idxs);
            auto unique_subrange = std::ranges::unique(node_idxs);
            node_idxs.erase(unique_subrange.begin(), unique_subrange.end());


            std::vector<IDX> inv_gnode_idxs(n_nodes_total, -1);
            for(IDX ilocal = 0; ilocal < node_idxs.size(); ++ilocal)
                { inv_gnode_idxs[node_idxs[ilocal]] = ilocal; }

            pmesh.nodes.reserve(n_nodes);
            for(unsigned long inode : node_idxs)
                { pmesh.nodes.push_back(gcoord[inode]); }

            // === Step 2: Elements ===

            // for each global element index, store the local index, or -1
            std::vector<IDX> inv_gel_idxs(mesh.elements.size(), -1);

            int domain_type;
            MPI_Recv(&domain_type, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            while (domain_type != (int) DOMAIN_TYPE::N_DOMAIN_TYPES) {
                IDX iel;
                MPI_Recv(&iel, 1, mpi_get_type<IDX>(), 0, 0, MPI_COMM_WORLD, &status);
                inv_gel_idxs[iel] = pmesh.elements.size();
                int geo_order, n_nodes;
                MPI_Recv(&geo_order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&n_nodes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                std::vector<IDX> elnodes(n_nodes);
                MPI_Recv(elnodes.data(), n_nodes, mpi_get_type<IDX>(), 0, 0, MPI_COMM_WORLD, &status);
                for(int inode = 0; inode < n_nodes; ++inode){
                    elnodes[inode] = inv_gnode_idxs[elnodes[inode]];
                }
                pmesh.elements.push_back(create_element<T, IDX, ndim>(
                    static_cast<DOMAIN_TYPE>(domain_type), geo_order, elnodes));

                // get the next domain type
                MPI_Recv(&domain_type, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            } 

            // === Step 3: Faces ===

            MPI_Recv(&domain_type, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            while (domain_type != (int) DOMAIN_TYPE::N_DOMAIN_TYPES) {

                // get the rank of the left element then right element
                int rank_l, rank_r;
                MPI_Recv(&rank_l, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&rank_r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

                bool im_left = (rank_l == iproc);
                int rank_other = (im_left) ? rank_r : rank_l;

                // get the global element indices for the left and right element 
                IDX iel, ier;
                MPI_Recv(&iel, 1, mpi_get_type<IDX>(), 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&ier, 1, mpi_get_type<IDX>(), 0, 0, MPI_COMM_WORLD, &status);

                // one of these will be -1, we will overwrite with the local from the other rank
                IDX iel_local = inv_gel_idxs[iel];
                IDX ier_local = inv_gel_idxs[ier];

                if(rank_l != rank_r){
                    // exchange local element index with othe rank
                    if(im_left)
                        MPI_Sendrecv(
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 1,
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 1, 
                            MPI_COMM_WORLD, &status
                        );
                    else
                        MPI_Sendrecv(
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 1,
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 1, 
                            MPI_COMM_WORLD, &status
                        );
                }

                // get the geometry_order
                int geo_order;
                MPI_Recv(&geo_order, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

                // get the face_nodes
                int n_face_nodes;
                MPI_Recv(&n_face_nodes, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                std::vector<IDX> face_nodes(n_face_nodes);
                MPI_Recv(face_nodes.data(), n_face_nodes, mpi_get_type<IDX>(), 0, 0, MPI_COMM_WORLD, &status);
                for(int inode = 0; inode < n_face_nodes; ++inode){
                    face_nodes[inode] = inv_gnode_idxs[face_nodes[inode]];
                }

                // get the face numbers and orientation
                int face_nr_l, face_nr_r, orient_r;
                MPI_Recv(&face_nr_l, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&face_nr_r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
                MPI_Recv(&orient_r, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);

                // the boundary flag is the rank of the other process, 
                // negated if we are the right side of the face
                BOUNDARY_CONDITIONS bctype = (rank_l == rank_r) ?
                    BOUNDARY_CONDITIONS::INTERIOR : BOUNDARY_CONDITIONS::PARALLEL_COM;
                int bcflag = (im_left) ? rank_other : -rank_other;

                auto face_opt = make_face<T, IDX, ndim>(
                    static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                    face_nodes, face_nr_l, face_nr_r, orient_r, bctype, bcflag
                );

                if(face_opt) {
                    pmesh.faces.push_back(std::move(face_opt.value()));
                } else {
                    util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                    return pmesh;
                }
            }



        }
        return pmesh;

    }

}


#endif // ICEICLE_USE_MPI

#else // No metis 
    
    template<class T, class IDX, int ndim>
    auto partition_mesh(AbstractMesh<T, IDX, ndim>& mesh) 
    -> AbstractMesh<T, IDX, ndim>
{
    return mesh;
}
#endif
