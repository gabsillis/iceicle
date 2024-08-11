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
        int nrank, myrank;
        MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
        MPI_Comm_size(MPI_COMM_WORLD, &nrank);

        // metis will floating point exception when partitioning into 1 partition
        // ...
        // ...
        //
        // :3
        if(nrank == 1) {
            // set up empty communication arrays that are used in parallel structures
            mesh.el_send_list = std::vector<std::vector<IDX>>(nrank, std::vector<IDX>());
            mesh.el_recv_list = std::vector<std::vector<IDX>>(nrank, std::vector<IDX>());
            mesh.communicated_elements.resize(nrank);
            return mesh;
        }

        /// @brief create an empty mesh -- this will be the partitioned mesh
        AbstractMesh<T, IDX, ndim> pmesh{};

        // Broadcast all the node data 
        NodeArray<T, ndim> gcoord{};
        unsigned long n_nodes_total;
        if(myrank == 0)
        { 
            gcoord = mesh.nodes;
            n_nodes_total = mesh.nodes.size();
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
        std::vector<IDX> inv_gel_idxs(mesh.elements.size(), -1);

        // initialize the communication requirement matrix
        pmesh.el_send_list = std::vector<std::vector<IDX>>(nrank, std::vector<IDX>());
        pmesh.el_recv_list = std::vector<std::vector<IDX>>(nrank, std::vector<IDX>());
        pmesh.communicated_elements.resize(nrank);

        // only the first processor will perform the partitioning
        if(myrank == 0){
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
                    NULL, NULL, NULL, &nrank, NULL, NULL,
                    options, &obj_val, el_partition.data());

            // ====================================
            // = Generate the Partitioned Meshes  =
            // ====================================

            // === Step 1: Node List ===

            std::vector<std::vector<unsigned long>> all_node_idxs(nrank, std::vector<unsigned long>{});

            // build each processor node list
            for(IDX iel = 0; iel < nelem; ++iel){
                IDX element_rank = el_partition[iel];
                for(IDX inode : mesh.elements[iel]->nodes_span()){
                    all_node_idxs[element_rank].push_back(inode);
                }
            }

            // send the node lists to each processor 
            for(int irank = 1; irank < nrank; ++irank){
                // send the size 
                unsigned long sz = all_node_idxs[irank].size();
                MPI_Send(&sz, 1, MPI_UNSIGNED_LONG, irank, 0, MPI_COMM_WORLD);
                MPI_Send(all_node_idxs[irank].data(), sz, MPI_UNSIGNED_LONG, irank, 0, MPI_COMM_WORLD);
            }

            gnode_idxs = all_node_idxs[0];
            std::ranges::sort(gnode_idxs);
            auto unique_subrange = std::ranges::unique(gnode_idxs);
            gnode_idxs.erase(unique_subrange.begin(), unique_subrange.end());

            for(IDX ilocal = 0; ilocal < gnode_idxs.size(); ++ilocal)
                { inv_gnode_idxs[gnode_idxs[ilocal]] = ilocal; }


            pmesh.nodes.reserve(gnode_idxs.size());
            for(unsigned long inode : gnode_idxs)
                { pmesh.nodes.push_back(gcoord[inode]); }

            // === Step 2: Elements ===


            for(IDX iel = 0; iel < nelem; ++iel){ 
                IDX element_rank = el_partition[iel];
                if(element_rank == 0){
                    GeometricElement<T, IDX, ndim> &el = *(mesh.elements[iel]);
                    inv_gel_idxs[iel] = pmesh.elements.size();
                    std::vector<IDX> elnodes(el.n_nodes());
                    for(int inode = 0; inode < el.n_nodes(); ++inode){
                        elnodes[inode] = inv_gnode_idxs[el.nodes()[inode]];
                    }
                    auto el_opt = create_element<T, IDX, ndim>(el.domain_type(), el.geometry_order(), elnodes);
                    if(el_opt)
                        pmesh.elements.push_back(std::move(el_opt.value()));
                    else 
                        util::AnomalyLog::log_anomaly(
                                util::Anomaly{"Could not create valid element.", util::general_anomaly_tag{}});
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
            for(int myrank = 1; myrank < nrank; ++myrank)
            {
                int stop_signal = (int) DOMAIN_TYPE::N_DOMAIN_TYPES;
                MPI_Send(&stop_signal, 1, MPI_INT, myrank, 1, MPI_COMM_WORLD);
            }

            // === Step 3: Interior Faces ===
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
                        }
                    } else {
                        // boundary with another process
                        int rank_other = rank_r;
                        // send all the information needed for them to make the face 
                        int domain_type = static_cast<int>(face.domain_type());
                        MPI_Send(&domain_type, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                        // send the rank of the left element and reight element 
                        MPI_Send(&rank_l, 1, MPI_INT, rank_other, 1, MPI_COMM_WORLD);
                        MPI_Send(&rank_r, 1, MPI_INT, rank_other, 2, MPI_COMM_WORLD);

                        // send the global element indices of the left and right element
                        MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_other, 3, MPI_COMM_WORLD);
                        MPI_Send(&ier, 1, mpi_get_type<IDX>(), rank_other, 4, MPI_COMM_WORLD);

                        // exchange local element index 
                        IDX iel_local = inv_gel_idxs[iel];
                        IDX ier_local;
                        MPI_Sendrecv(
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 5,
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 5, 
                            MPI_COMM_WORLD, &status
                        );
                        pmesh.el_send_list[rank_other].push_back(iel_local);
                        pmesh.el_recv_list[rank_other].push_back(ier_local);

                        // send the geometry order 
                        int geo_order = face.geometry_order();
                        MPI_Send(&geo_order, 1, MPI_INT, rank_other, 6, MPI_COMM_WORLD);

                        // send the face nodes 
                        int n_face_nodes = face.n_nodes();
                        MPI_Send(&n_face_nodes, 1, MPI_INT, rank_other, 7, MPI_COMM_WORLD);
                        MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_other, 8, MPI_COMM_WORLD);

                        // create a list of local nodes
                        std::vector<IDX> local_face_nodes(n_face_nodes);
                        for(int inode = 0; inode < n_face_nodes; ++inode)
                            local_face_nodes[inode] = inv_gnode_idxs[face.nodes()[inode]];

                        // send the face numbers and orientation 
                        int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                        int orient_r = face.orientation_r();
                        MPI_Send(&face_nr_l, 1, MPI_INT, rank_other, 9, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_r, 1, MPI_INT, rank_other, 10, MPI_COMM_WORLD);
                        MPI_Send(&orient_r, 1, MPI_INT, rank_other, 11, MPI_COMM_WORLD);

                        auto face_opt = make_face<T, IDX, ndim>(
                            static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                            local_face_nodes, face_nr_l, face_nr_r, orient_r,
                            BOUNDARY_CONDITIONS::PARALLEL_COM, encode_mpi_bcflag(rank_other, true)
                        );

                        if(face_opt) {
                            pmesh.faces.push_back(std::move(face_opt.value()));
                        } else {
                            util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                        }
                    }

                } else if(rank_r == 0){
                    // rank_l == 0 is already covered
                    // boundary with another process
                    int rank_other = rank_l;
                    // send all the information needed for them to make the face 
                    int domain_type = static_cast<int>(face.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, rank_other, 0, MPI_COMM_WORLD);

                    // send the rank of the left element and right element 
                    MPI_Send(&rank_l, 1, MPI_INT, rank_other, 1, MPI_COMM_WORLD);
                    MPI_Send(&rank_r, 1, MPI_INT, rank_other, 2, MPI_COMM_WORLD);

                    // send the global element indices of the left and right element
                    MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_other, 3, MPI_COMM_WORLD);
                    MPI_Send(&ier, 1, mpi_get_type<IDX>(), rank_other, 4, MPI_COMM_WORLD);

                    // exchange local element index 
                    IDX iel_local;
                    IDX ier_local = inv_gel_idxs[ier];
                    MPI_Sendrecv(
                        &ier_local, 1, mpi_get_type<IDX>(), rank_other, 5,
                        &iel_local, 1, mpi_get_type<IDX>(), rank_other, 5, 
                        MPI_COMM_WORLD, &status
                    );
                    pmesh.el_send_list[rank_other].push_back(ier_local);
                    pmesh.el_recv_list[rank_other].push_back(iel_local);

                    // send the geometry order 
                    int geo_order = face.geometry_order();
                    MPI_Send(&geo_order, 1, MPI_INT, rank_other, 6, MPI_COMM_WORLD);

                    // send the face nodes 
                    int n_face_nodes = face.n_nodes();
                    MPI_Send(&n_face_nodes, 1, MPI_INT, rank_other, 7, MPI_COMM_WORLD);
                    MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_other, 8, MPI_COMM_WORLD);

                    // create a list of local nodes
                    std::vector<IDX> local_face_nodes(n_face_nodes);
                    for(int inode = 0; inode < n_face_nodes; ++inode)
                        local_face_nodes[inode] = inv_gnode_idxs[face.nodes()[inode]];

                    // send the face numbers and orientation 
                    int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                    int orient_r = face.orientation_r();
                    MPI_Send(&face_nr_l, 1, MPI_INT, rank_other, 9, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, rank_other, 10, MPI_COMM_WORLD);
                    MPI_Send(&orient_r, 1, MPI_INT, rank_other, 11, MPI_COMM_WORLD);

                    auto face_opt = make_face<T, IDX, ndim>(
                        static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                        local_face_nodes, face_nr_l, face_nr_r, orient_r,
                        BOUNDARY_CONDITIONS::PARALLEL_COM, encode_mpi_bcflag(rank_other, false)
                    );

                    if(face_opt) {
                        pmesh.faces.push_back(std::move(face_opt.value()));
                    } else {
                        util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                    }
                } else {
                    // both ranks are on another process 

                    if(rank_l != rank_r){
                        // send all the information needed for them to make the face 
                        int domain_type = static_cast<int>(face.domain_type());
                        MPI_Send(&domain_type, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);
                        MPI_Send(&domain_type, 1, MPI_INT, rank_r, 0, MPI_COMM_WORLD);

                        // send the rank of the left element and reight element 
                        MPI_Send(&rank_l, 1, MPI_INT, rank_l, 1, MPI_COMM_WORLD);
                        MPI_Send(&rank_r, 1, MPI_INT, rank_l, 2, MPI_COMM_WORLD);
                        MPI_Send(&rank_l, 1, MPI_INT, rank_r, 1, MPI_COMM_WORLD);
                        MPI_Send(&rank_r, 1, MPI_INT, rank_r, 2, MPI_COMM_WORLD);

                        // send the global element indices of the left and right element
                        MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_l, 3, MPI_COMM_WORLD);
                        MPI_Send(&ier, 1, mpi_get_type<IDX>(), rank_l, 4, MPI_COMM_WORLD);
                        MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_r, 3, MPI_COMM_WORLD);
                        MPI_Send(&ier, 1, mpi_get_type<IDX>(), rank_r, 4, MPI_COMM_WORLD);

                        // send the geometry order 
                        int geo_order = face.geometry_order();
                        MPI_Send(&geo_order, 1, MPI_INT, rank_l, 6, MPI_COMM_WORLD);
                        MPI_Send(&geo_order, 1, MPI_INT, rank_r, 6, MPI_COMM_WORLD);

                        // send the face nodes 
                        int n_face_nodes = face.n_nodes();
                        MPI_Send(&n_face_nodes, 1, MPI_INT, rank_l, 7, MPI_COMM_WORLD);
                        MPI_Send(&n_face_nodes, 1, MPI_INT, rank_r, 7, MPI_COMM_WORLD);
                        MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_l, 8, MPI_COMM_WORLD);
                        MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_r, 8, MPI_COMM_WORLD);

                        // send the face numbers and orientation 
                        int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                        int orient_r = face.orientation_r();
                        MPI_Send(&face_nr_l, 1, MPI_INT, rank_l, 9, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_r, 1, MPI_INT, rank_l, 10, MPI_COMM_WORLD);
                        MPI_Send(&orient_r, 1, MPI_INT, rank_l, 11, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_l, 1, MPI_INT, rank_r, 9, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_r, 1, MPI_INT, rank_r, 10, MPI_COMM_WORLD);
                        MPI_Send(&orient_r, 1, MPI_INT, rank_r, 11, MPI_COMM_WORLD);
                    } else {
                        // send all the information needed for them to make the face 
                        int domain_type = static_cast<int>(face.domain_type());
                        MPI_Send(&domain_type, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);

                        // send the rank of the left element and reight element 
                        MPI_Send(&rank_l, 1, MPI_INT, rank_l, 1, MPI_COMM_WORLD);
                        MPI_Send(&rank_r, 1, MPI_INT, rank_l, 2, MPI_COMM_WORLD);

                        // send the global element indices of the left and right element
                        MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_l, 3, MPI_COMM_WORLD);
                        MPI_Send(&ier, 1, mpi_get_type<IDX>(), rank_l, 4, MPI_COMM_WORLD);

                        // send the geometry order 
                        int geo_order = face.geometry_order();
                        MPI_Send(&geo_order, 1, MPI_INT, rank_l, 6, MPI_COMM_WORLD);

                        // send the face nodes 
                        int n_face_nodes = face.n_nodes();
                        MPI_Send(&n_face_nodes, 1, MPI_INT, rank_l, 7, MPI_COMM_WORLD);
                        MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_l, 8, MPI_COMM_WORLD);

                        // send the face numbers and orientation 
                        int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                        int orient_r = face.orientation_r();
                        MPI_Send(&face_nr_l, 1, MPI_INT, rank_l, 9, MPI_COMM_WORLD);
                        MPI_Send(&face_nr_r, 1, MPI_INT, rank_l, 10, MPI_COMM_WORLD);
                        MPI_Send(&orient_r, 1, MPI_INT, rank_l, 11, MPI_COMM_WORLD);
                    }
                }
            }

            // send signal to stop to each process (domain type = N_DOMAIN_TYPES)
            for(int myrank = 1; myrank < nrank; ++myrank) {
                int stop_signal = (int) DOMAIN_TYPE::N_DOMAIN_TYPES;
                MPI_Send(&stop_signal, 1, MPI_INT, myrank, 0, MPI_COMM_WORLD);
            }

            // reorganize the faces so that the now boundary faces (PARALLEL_COM) are at the end
            IDX boundary_faces_begin = pmesh.faces.size();
            for(IDX iface = 0; iface < boundary_faces_begin; ++iface) {
                if(pmesh.faces[iface]->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM)
                {
                    --boundary_faces_begin;
                    std::swap(pmesh.faces[iface], pmesh.faces[boundary_faces_begin]);
                    --iface;
                }
            }
            pmesh.interiorFaceStart = 0;
            pmesh.interiorFaceEnd = boundary_faces_begin;
            pmesh.bdyFaceStart = boundary_faces_begin;

            // === Step 4: Boundary Faces ===

            for(IDX iface = mesh.bdyFaceStart; iface < mesh.bdyFaceEnd; ++iface) {

                Face<T, IDX, ndim>& face = *(mesh.faces[iface]);

                IDX iel = face.elemL;
                int rank_l = el_partition[iel];

                if(rank_l == 0) {

                    // create a list of local nodes
                    std::vector<IDX> local_face_nodes(face.n_nodes());
                    for(int inode = 0; inode < face.n_nodes(); ++inode)
                        local_face_nodes[inode] = inv_gnode_idxs[face.nodes()[inode]];
                    // face is local
                    auto face_opt = make_face<T, IDX, ndim>(
                        face.domain_type(), face.geometry_order(), inv_gel_idxs[iel], 0,
                        local_face_nodes, 
                        face.face_nr_l(), face.face_nr_r(), face.orientation_r(),
                        face.bctype, face.bcflag
                    );

                    if(face_opt) {
                        pmesh.faces.push_back(std::move(face_opt.value()));
                    } else {
                        util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                    }
                } else {
                    // send all the information needed for them to make the face 
                    int domain_type = static_cast<int>(face.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, rank_l, 0, MPI_COMM_WORLD);

                    // send the global element indices of the left element
                    MPI_Send(&iel, 1, mpi_get_type<IDX>(), rank_l, 1, MPI_COMM_WORLD);

                    // send the geometry order 
                    int geo_order = face.geometry_order();
                    MPI_Send(&geo_order, 1, MPI_INT, rank_l, 2, MPI_COMM_WORLD);

                    // send the face nodes 
                    int n_face_nodes = face.n_nodes();
                    MPI_Send(&n_face_nodes, 1, MPI_INT, rank_l, 3, MPI_COMM_WORLD);
                    MPI_Send(face.nodes(), n_face_nodes, mpi_get_type<IDX>(), rank_l, 4, MPI_COMM_WORLD);

                    // send the face numbers and orientation 
                    int face_nr_l = face.face_nr_l(), face_nr_r = face.face_nr_r();
                    int orient_r = face.orientation_r();
                    MPI_Send(&face_nr_l, 1, MPI_INT, rank_l, 5, MPI_COMM_WORLD);
                    MPI_Send(&face_nr_r, 1, MPI_INT, rank_l, 6, MPI_COMM_WORLD);
                    MPI_Send(&orient_r, 1, MPI_INT, rank_l, 7, MPI_COMM_WORLD);

                    // send the boundary condition information 
                    int bctype = static_cast<int>(face.bctype);
                    int bcflag = face.bcflag;
                    MPI_Send(&bctype, 1, MPI_INT, rank_l, 8, MPI_COMM_WORLD);
                    MPI_Send(&bcflag, 1, MPI_INT, rank_l, 9, MPI_COMM_WORLD);
                }
            }

            // send signal to stop to each process (domain type = N_DOMAIN_TYPES)
            for(int myrank = 1; myrank < nrank; ++myrank) {
                int stop_signal = (int) DOMAIN_TYPE::N_DOMAIN_TYPES;
                MPI_Send(&stop_signal, 1, MPI_INT, myrank, 0, MPI_COMM_WORLD);
            }

            pmesh.bdyFaceEnd = pmesh.faces.size();
        } else {
            // ====================================
            // = Generate the Partitioned Meshes  =
            // ====================================

            // === Step 1: Node List ===

            MPI_Status status;
            unsigned long n_nodes;
            MPI_Recv(&n_nodes, 1, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);
            gnode_idxs.resize(n_nodes);
            MPI_Recv(gnode_idxs.data(), n_nodes, MPI_UNSIGNED_LONG, 0, 0, MPI_COMM_WORLD, &status);

            std::ranges::sort(gnode_idxs);
            auto unique_subrange = std::ranges::unique(gnode_idxs);
            gnode_idxs.erase(unique_subrange.begin(), unique_subrange.end());


            for(IDX ilocal = 0; ilocal < gnode_idxs.size(); ++ilocal)
                { inv_gnode_idxs[gnode_idxs[ilocal]] = ilocal; }

            pmesh.nodes.reserve(n_nodes);
            for(unsigned long inode : gnode_idxs)
                { pmesh.nodes.push_back(gcoord[inode]); }

            // === Step 2: Elements ===

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
                auto el_opt = create_element<T, IDX, ndim>(
                    static_cast<DOMAIN_TYPE>(domain_type), geo_order, elnodes);
                if(el_opt)
                    pmesh.elements.push_back(std::move(el_opt.value()));
                else 
                    util::AnomalyLog::log_anomaly(
                            util::Anomaly{"Could not create valid element.", util::general_anomaly_tag{}});

                // get the next domain type
                MPI_Recv(&domain_type, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            } 

            // === Step 3: Interior Faces ===

            MPI_Recv(&domain_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            while (domain_type != (int) DOMAIN_TYPE::N_DOMAIN_TYPES) {

                // get the rank of the left element then right element
                int rank_l, rank_r;
                MPI_Recv(&rank_l, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
                MPI_Recv(&rank_r, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

                bool im_left = (rank_l == myrank);
                int rank_other = (im_left) ? rank_r : rank_l;

                // get the global element indices for the left and right element 
                IDX iel, ier;
                MPI_Recv(&iel, 1, mpi_get_type<IDX>(), 0, 3, MPI_COMM_WORLD, &status);
                MPI_Recv(&ier, 1, mpi_get_type<IDX>(), 0, 4, MPI_COMM_WORLD, &status);

                // one of these will be -1, we will overwrite with the local from the other rank
                IDX iel_local = inv_gel_idxs[iel];
                IDX ier_local = inv_gel_idxs[ier];

                if(rank_l != rank_r){
                    // exchange local element index with other rank
                    if(im_left){
                        MPI_Sendrecv(
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 5,
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 5, 
                            MPI_COMM_WORLD, &status
                        );
                        pmesh.el_send_list[rank_other].push_back(iel_local);
                        pmesh.el_recv_list[rank_other].push_back(ier_local);
                    } else {
                        MPI_Sendrecv(
                            &ier_local, 1, mpi_get_type<IDX>(), rank_other, 5,
                            &iel_local, 1, mpi_get_type<IDX>(), rank_other, 5, 
                            MPI_COMM_WORLD, &status
                        );
                        pmesh.el_send_list[rank_other].push_back(ier_local);
                        pmesh.el_recv_list[rank_other].push_back(iel_local);
                    }
                }

                // get the geometry_order
                int geo_order;
                MPI_Recv(&geo_order, 1, MPI_INT, 0, 6, MPI_COMM_WORLD, &status);

                // get the face_nodes
                int n_face_nodes;
                MPI_Recv(&n_face_nodes, 1, MPI_INT, 0, 7, MPI_COMM_WORLD, &status);
                std::vector<IDX> face_nodes(n_face_nodes);
                MPI_Recv(face_nodes.data(), n_face_nodes, mpi_get_type<IDX>(), 0, 8, MPI_COMM_WORLD, &status);
                for(int inode = 0; inode < n_face_nodes; ++inode)
                    face_nodes[inode] = inv_gnode_idxs[face_nodes[inode]];

                // get the face numbers and orientation
                int face_nr_l, face_nr_r, orient_r;
                MPI_Recv(&face_nr_l, 1, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);
                MPI_Recv(&face_nr_r, 1, MPI_INT, 0, 10, MPI_COMM_WORLD, &status);
                MPI_Recv(&orient_r, 1, MPI_INT, 0, 11, MPI_COMM_WORLD, &status);

                // the boundary flag is the rank of the other process, 
                // negated if we are the right side of the face
                BOUNDARY_CONDITIONS bctype = (rank_l == rank_r) ?
                    BOUNDARY_CONDITIONS::INTERIOR : BOUNDARY_CONDITIONS::PARALLEL_COM;
                int bcflag = encode_mpi_bcflag(rank_other, im_left);

                auto face_opt = make_face<T, IDX, ndim>(
                    static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, ier_local, 
                    face_nodes, face_nr_l, face_nr_r, orient_r, bctype, bcflag
                );

                if(face_opt) {
                    pmesh.faces.push_back(std::move(face_opt.value()));
                } else {
                    util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                }

                // get the next domain type
                MPI_Recv(&domain_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }

            // reorganize the faces so that the now boundary faces (PARALLEL_COM) are at the end
            IDX boundary_faces_begin = pmesh.faces.size();
            for(IDX iface = 0; iface < boundary_faces_begin; ++iface)
            {
                if(pmesh.faces[iface]->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM)
                {
                    --boundary_faces_begin;
                    std::swap(pmesh.faces[iface], pmesh.faces[boundary_faces_begin]);
                    --iface;
                }
            }
            pmesh.interiorFaceStart = 0;
            pmesh.interiorFaceEnd = boundary_faces_begin;
            pmesh.bdyFaceStart = boundary_faces_begin;

            // === Step 4: Boundary Faces ===
            MPI_Recv(&domain_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            while (domain_type != (int) DOMAIN_TYPE::N_DOMAIN_TYPES) {
                IDX iel;

                // get the global element index
                MPI_Recv(&iel, 1, mpi_get_type<IDX>(), 0, 1, MPI_COMM_WORLD, &status);
                IDX iel_local = inv_gel_idxs[iel];

                // get the geometry_order
                int geo_order;
                MPI_Recv(&geo_order, 1, MPI_INT, 0, 2, MPI_COMM_WORLD, &status);

                // get the face_nodes
                int n_face_nodes;
                MPI_Recv(&n_face_nodes, 1, MPI_INT, 0, 3, MPI_COMM_WORLD, &status);
                std::vector<IDX> face_nodes(n_face_nodes);
                MPI_Recv(face_nodes.data(), n_face_nodes, mpi_get_type<IDX>(), 0, 4, MPI_COMM_WORLD, &status);
                for(int inode = 0; inode < n_face_nodes; ++inode){
                    face_nodes[inode] = inv_gnode_idxs[face_nodes[inode]];
                }

                // get the face numbers and orientation
                int face_nr_l, face_nr_r, orient_r;
                MPI_Recv(&face_nr_l, 1, MPI_INT, 0, 5, MPI_COMM_WORLD, &status);
                MPI_Recv(&face_nr_r, 1, MPI_INT, 0, 6, MPI_COMM_WORLD, &status);
                MPI_Recv(&orient_r, 1, MPI_INT, 0, 7, MPI_COMM_WORLD, &status);

                // get the boundary condition information
                int bctype, bcflag;
                MPI_Recv(&bctype, 1, MPI_INT, 0, 8, MPI_COMM_WORLD, &status);
                MPI_Recv(&bcflag, 1, MPI_INT, 0, 9, MPI_COMM_WORLD, &status);

                auto face_opt = make_face<T, IDX, ndim>(
                    static_cast<DOMAIN_TYPE>(domain_type), geo_order, iel_local, 0,
                    face_nodes, face_nr_l, face_nr_r, orient_r, 
                    static_cast<BOUNDARY_CONDITIONS>(bctype), bcflag
                );

                if(face_opt) {
                    pmesh.faces.push_back(std::move(face_opt.value()));
                } else {
                    util::AnomalyLog::log_anomaly(util::Anomaly{"could not find face", util::general_anomaly_tag{}});
                }

                // get the next domain type
                MPI_Recv(&domain_type, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            }
            pmesh.bdyFaceEnd = pmesh.faces.size();

        }

        // sort the element communication lists and remove duplicates
        for(int i = 0; i < nrank; ++i){
            std::ranges::sort(pmesh.el_recv_list[i]);
            auto unique_subrange = std::ranges::unique(pmesh.el_recv_list[i]);
            pmesh.el_recv_list[i].erase(unique_subrange.begin(), unique_subrange.end());
        }


        MPI_Status status;
        // communicate boundary geo elements
        for(int irank = 0; irank < nrank; ++irank){
            if(irank == myrank){
                // send all the information to create elements
                for(int jrank = 0; jrank < nrank; ++jrank){
                    // basis order, quadrature_type, and basis_type we know
                    int geometry_order, domain_type;
                    for(IDX ielem : pmesh.el_send_list[jrank]){
                        geometry_order = (int) pmesh.elements[ielem]->geometry_order();
                        MPI_Send(&geometry_order, 1, MPI_INT, jrank, ielem, MPI_COMM_WORLD);
                        domain_type = (int) pmesh.elements[ielem]->domain_type();
                        MPI_Send(&domain_type, 1, MPI_INT, jrank, ielem, MPI_COMM_WORLD);
                        int n_nodes = pmesh.elements[ielem]->n_nodes();
                        MPI_Send(&n_nodes, 1, MPI_INT, jrank, ielem, MPI_COMM_WORLD);

                        // global indices of element nodes
                        // NOTE: for some reason storing this in a vector and sending the array fails 
                        // ...
                        // ...
                        for(IDX inode : pmesh.elements[ielem]->nodes_span()){
                            MPI_Send(&gnode_idxs[inode], 1, mpi_get_type<IDX>(), jrank, 10, MPI_COMM_WORLD);
                        }
                    }
                }
            } else {
                // recieve all the information to create elements from irank
                for(IDX ielem : pmesh.el_recv_list[irank]){
                    int geometry_order, domain_type, n_nodes;
                    MPI_Recv(&geometry_order, 1, MPI_INT, irank, ielem, MPI_COMM_WORLD, &status);
                    MPI_Recv(&domain_type, 1, MPI_INT, irank, ielem, MPI_COMM_WORLD, &status);
                    MPI_Recv(&n_nodes, 1, MPI_INT, irank, ielem, MPI_COMM_WORLD, &status);

                    std::vector<IDX> el_gnodes(n_nodes);
                    for(int i = 0; i < n_nodes; ++i)
                        MPI_Recv(&el_gnodes[i], 1, mpi_get_type<IDX>(), irank, 10, MPI_COMM_WORLD, &status);

                    std::vector<IDX> el_nodes;
                    for(IDX ignode : el_gnodes){
                        IDX inode = inv_gnode_idxs[ignode];

                        // check if node is found or add it
                        if(inode == -1){
                            // add a new node
                            inode = gnode_idxs.size();
                            inv_gnode_idxs[gnode_idxs.size()] = ignode;
                            gnode_idxs.push_back(ignode);
                            pmesh.nodes.push_back(gcoord[ignode]);
                        }

                        el_nodes.push_back(inode);
                    }

                    auto el_opt = create_element<T, IDX, ndim>((DOMAIN_TYPE) domain_type, geometry_order, el_nodes);

                    if(el_opt)
                        pmesh.communicated_elements[irank].push_back(std::move(el_opt.value()));
                    else 
                        util::AnomalyLog::log_anomaly(
                                util::Anomaly{"Could not create valid element.", util::general_anomaly_tag{}});
                }
            }
            MPI_Barrier(MPI_COMM_WORLD);
        }

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
    
    template<class T, class IDX, int ndim>
    auto partition_mesh(AbstractMesh<T, IDX, ndim>& mesh) 
    -> AbstractMesh<T, IDX, ndim>
{
    return mesh;
}
#endif
