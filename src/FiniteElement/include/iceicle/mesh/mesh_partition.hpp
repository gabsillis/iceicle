#include "iceicle/fe_definitions.hpp"
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

            for(IDX iel = 0; iel < nelem; ++iel){ 
                IDX element_rank = el_partition[iel];
                if(element_rank == 0){
                    GeometricElement<T, IDX, ndim> &el = *(mesh.elements[iel]);
                    std::vector<IDX> elnodes(el.n_nodes());
                    for(int inode = 0; inode < el.n_nodes(); ++inode){
                        elnodes[inode] = inv_gnode_idxs[el.nodes()[inode]];
                    }
                    pmesh.elements.push_back(create_element<T, IDX, ndim>(el.domain_type(), el.geometry_order(), elnodes));
                } else {
                    GeometricElement<T, IDX, ndim> &el = *(mesh.elements[iel]);

                    int domain_type = static_cast<int>(el.domain_type());
                    MPI_Send(&domain_type, 1, MPI_INT, element_rank, 1, MPI_COMM_WORLD);

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

            int domain_type;
            MPI_Recv(&domain_type, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, &status);
            while (domain_type != (int) DOMAIN_TYPE::N_DOMAIN_TYPES) {
                MPI_Status status;
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
