
#pragma once

#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/algo.hpp"
#include <map>
#include <list>

namespace iceicle {

    template<class riemann_t>
    concept calculates_wavespeed = 
    requires(
        const riemann_t& riemann_solver,
        std::array<typename riemann_t::value_type, riemann_t::nv_comp> uL,
        std::array<typename riemann_t::value_type, riemann_t::nv_comp> uR
    ){
        {riemann_solver.wavespeeds(uL, uR)} 
            -> std::same_as< std::vector< typename riemann_t::value_type > >;
    };

    template<class T, class IDX, int ndim, 
        class LayoutPolicy, class AccessorPolicy, class riemann_t>
    auto extrude_mesh( 
        FESpace<T, IDX, ndim>& fespace,
        fespan<T, LayoutPolicy, AccessorPolicy> u,
        riemann_t riemann_solver,
        std::function<void(const T*, T*)> ic,
        T tfinal
    ) -> std::optional< AbstractMesh<T, IDX, ndim + 1> > 
    {
        AbstractMesh<T, IDX, ndim>& mesh_old = *(fespace.meshptr);
        if constexpr(calculates_wavespeed<riemann_t>){

        }
    }

    /**
     *
     *  NOTE: we assume the time dimension is always the last
     */
    template<class T, class IDX, int ndim>
    auto compute_st_node_connectivity(
        AbstractMesh<T, IDX, ndim> &mesh_past,
        AbstractMesh<T, IDX, ndim> &mesh_current
    ) -> std::map<IDX, IDX> {
        static_assert(ndim > 1, "Assumes at least 2D spacetime mesh");
        using Face = Face<T, IDX, ndim>;
        std::vector<bool> past_nodes_connected(mesh_past.n_nodes(), false);
        std::vector<bool> current_nodes_connected(mesh_current.n_nodes(), false);

        // look for nodes in SPACETIME_FUTURE boundary in the past mesh
        for(IDX ibface = mesh_past.bdyFaceStart; ibface < mesh_past.bdyFaceEnd; ++ibface){
            const Face& face = *(mesh_past.faces[ibface]);
            if(face.bctype == BOUNDARY_CONDITIONS::SPACETIME_FUTURE){
                for(IDX inode : face.nodes_span()){
                    past_nodes_connected[inode] = true;
                }
            }
        }

        // look for nodes in SPACETIME_PAST boundary in the current mesh
        for(IDX ibface = mesh_current.bdyFaceStart; ibface < mesh_current.bdyFaceEnd; ++ibface){
            const Face& face = *(mesh_current.faces[ibface]);
            if(face.bctype == BOUNDARY_CONDITIONS::SPACETIME_PAST){
                for(IDX inode : face.nodes_span()){
                    current_nodes_connected[inode] = true;
                }
            }
        }

        // create array of the node indices on the past mesh that are on the ST boundary
        std::vector<IDX> past_nodes{};
        for(IDX inode = 0; inode < mesh_past.n_nodes(); ++inode){
            if(past_nodes_connected[inode]) past_nodes.push_back(inode);
        }

        // find connected nodes 
        std::map<IDX, IDX> curr_to_past_nodes;
        for(IDX inode_curr = 0; inode_curr < mesh_current.n_nodes(); ++inode_curr){
            if(current_nodes_connected[inode_curr]){
                // search for a node in the past_nodes list that is close enough in all coordinates except the last 
                // NOTE: we assume the time dimension is always the last
                for(IDX inode_past : past_nodes){
                    std::span<T> space_coords_current = std::span{mesh_current.coord[inode_curr].begin(), mesh_current.coord[inode_curr].end() - 1};
                    std::span<T> space_coords_past = std::span{mesh_past.coord[inode_past].begin(), mesh_past.coord[inode_past].end() - 1};
                    if(util::eqset(space_coords_current, space_coords_past)){
                        curr_to_past_nodes[inode_curr] = inode_past;
                        break;
                    }
                }
            }
        }

        return curr_to_past_nodes;
    }

    template<
        class T,
        class IDX,
        int ndim,
        class upastLayout
    >
    class SpacetimeConnection {

        FESpace<T, IDX, ndim> &fespace_past;
        FESpace<T, IDX, ndim> &fespace_current;
        fespan<T, upastLayout> u_past;

        /// map the index of nodes in the current fespace 
        /// to nodes in the past fespace
        std::map<IDX, IDX> &curr_to_past_nodes;

        /// @brief Geometric faces that represent the connection between currrent and past elements
        std::vector< std::unique_ptr< Face<T, IDX, ndim> > > connection_faces;

        /// @brief trace spaces that represent the connection between current and past elements
        /// keyed by the current trace index
        std::unordered_map<IDX, TraceSpace<T, IDX, ndim> > connection_traces;

        public:

        SpacetimeConnection(
            FESpace<T, IDX, ndim> &fespace_past,
            FESpace<T, IDX, ndim> &fespace_current,
            fespan<T, upastLayout> &u_past,
            std::map<IDX, IDX> &curr_to_past_nodes
        ) noexcept : fespace_past{fespace_past}, fespace_current{fespace_current}, 
                     u_past{u_past}, curr_to_past_nodes{curr_to_past_nodes}
        {
            using TraceSpace = TraceSpace<T, IDX, ndim>;

            // === find attached traces ===
            
            // boundary face indexes that still need to be connected
            std::list<IDX> past_bface_idxs{};
            for(int ibface = fespace_past.bdy_trace_start; ibface < fespace_past.bdy_trace_end; ++ibface) {
                TraceSpace past_trace = fespace_past.traces[ibface];

                // from the perspective of past fespace, the current fespace faces are connected at SPACETIME_FUTURE
                if(past_trace.face.bctype == BOUNDARY_CONDITIONS::SPACETIME_FUTURE) {
                    past_bface_idxs.push_back(ibface);
                }
            }

            IDX itrace = 0;
            for(TraceSpace &current_trace : fespace_current.get_boundary_traces()) {
                std::vector<IDX> curr_nodes_connected_idxs{current_trace.face->nodes_span()};
                for(IDX& inode : curr_nodes_connected_idxs) inode = curr_to_past_nodes[inode];

                for(IDX ibface_past : past_bface_idxs){
                    TraceSpace past_trace = fespace_past.traces[ibface_past];

                    if(util::eqset(std::span{curr_nodes_connected_idxs}, past_trace.face->nodes_span())){
                        // make a new face
                        Face<T, IDX, ndim>* faceptr = make_face(
                                current_trace.elL.elidx,
                                past_trace.elL.elidx,
                                current_trace.elL.geo_el,
                                past_trace.elL.geo_el,
                                current_trace.face.face_nr_l(),
                                past_trace.face.face_nr_l(),
                                BOUNDARY_CONDITIONS::INTERIOR, // boundary condition doesn't have a lot of meaning here
                                0
                        );
                        connection_faces.emplace_back(faceptr);

                        // make a new trace space
                        TraceSpace connecting_trace{
                            faceptr,
                            current_trace.elL,
                            past_trace.elL,
                            current_trace.trace_basis,
                            current_trace.quadrule,
                            current_trace.qp_evals,
                            itrace++
                        };
                        connection_traces[current_trace.facidx] = connecting_trace;
                    }
                }
            }
        }
    };

}
