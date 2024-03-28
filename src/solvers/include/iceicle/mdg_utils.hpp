/**
 * @brief utilities for solving mdg problems
 * 
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#include "Numtool/point.hpp"
#include "iceicle/fe_function/nodal_fe_function.hpp"
#include "iceicle/fespace/fespace.hpp"
#include <vector>
namespace FE {

    /**
     * @brief given the current locations of surface nodes 
     * recalculate the locations of interior nodes for elements 
     * according to their barycentric weights 
     * @param fespace the finite element space
     */
    template<class T, class IDX, int ndim>
    auto regularize_interior_nodes(FESpace<T, IDX, ndim> &fespace) -> void {
        for(auto& element : fespace.elements){
            element.geo_el->regularize_interior_nodes(fespace.meshptr->nodes);
        }
    }


    /**
     * @brief get a vector of doubles 
     * representing the distance to the nearest node for every node 
     * NOTE: in terms of node movement: 0.5 * radius should be enough to prevent overlaps
     */
    template<class T, class IDX, int ndim>
    auto node_freedom_radii(
        FESpace<T, IDX, ndim> &fespace
    ) -> std::vector<T> {
        using element_t = FESpace<T, IDX, ndim>::ElementType;
        static constexpr T big_number = 1e100;
        NodalFEFunction<T, ndim>& coord = fespace.meshptr->nodes;
        
        std::vector<T> radii(coord.n_nodes(), big_number);

        // NOTE: loop over elements because interior nodes exist
        for(element_t& elem: fespace.elements){
            // loop over all nodes and get min distance to surrounding nodes
            for(IDX inode : elem.geo_el->nodes_span()){
                for(IDX jnode : elem.geo_el->nodes_span()) if (jnode != inode){
                    T distance = MATH::GEOMETRY::distance(
                        coord[inode].clone_pt(),
                        coord[jnode].clone_pt()
                    );

                    radii[inode] = std::min(radii[inode], distance);
                    radii[jnode] = std::min(radii[jnode], distance);
                }
            }
        }

        return radii;
    }
}
