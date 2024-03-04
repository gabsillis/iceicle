
#pragma once

#include "iceicle/element/TraceSpace.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/fespace/fespace.hpp"
#include "iceicle/geometry/face.hpp"
#include <map>
#include <list>
namespace DISC {
    template<
        class T,
        class IDX,
        int ndim,
        class upastLayout
    >
    class SpacetimeConnection {

        FE::FESpace<T, IDX, ndim> &fespace_past;
        FE::FESpace<T, IDX, ndim> &fespace_current;
        FE::fespan<T, upastLayout> u_past;

        /// map the index of nodes in the current fespace 
        /// to nodes in the past fespace
        std::map<IDX, IDX> &curr_to_past_nodes;

        public:

        SpacetimeConnection(
            FE::FESpace<T, IDX, ndim> &fespace_past,
            FE::FESpace<T, IDX, ndim> &fespace_current,
            FE::fespan<T, upastLayout> &u_past,
            std::map<IDX, IDX> &curr_to_past_nodes
        ) noexcept : fespace_past{fespace_past}, fespace_current{fespace_current}, 
                     u_past{u_past}, curr_to_past_nodes{curr_to_past_nodes}
        {
            using TraceSpace = ELEMENT::TraceSpace<T, IDX, ndim>;

            // === find attached traces ===
            
            // boundary face indexes that still need to be connected
            std::list<IDX> past_bface_idxs{};
            for(int ibface = fespace_past.bdy_trace_start; ibface < fespace_past.bdy_trace_end; ++ibface) {
                TraceSpace past_trace = fespace_past.traces[ibface];

                // from the perspective of past fespace, the current fespace faces are connected at SPACETIME_FUTURE
                if(past_trace.face.bctype == ELEMENT::BOUNDARY_CONDITIONS::SPACETIME_FUTURE) {
                    past_bface_idxs.push_back(ibface);
                }
            }

            for(TraceSpace &current_trace : fespace_current) {
                
            }

        }
    };

}
