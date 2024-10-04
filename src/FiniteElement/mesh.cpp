/**
 * @file mesh.cpp
 * @brief mesh implementation
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include <iceicle/mesh/mesh.hpp> 
#include <iceicle/build_config.hpp>
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/simplex_element.hpp>
#include <iceicle/geometry/hypercube_element.hpp>
#include <string>
#include <map>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <utility>

namespace iceicle {
    
    using T = build_config::T;
    using IDX = build_config::IDX;

    static constexpr int PN_GEOM_MAX = 10;

//    template<>
//    AbstractMesh<T, IDX, 2>::AbstractMesh(mfem::FiniteElementSpace &mfem_fespace)
//    : nodes{} {
//        static constexpr int ndim = 2;
//        int myid;
//        MPI_Comm_rank(MPI_COMM_WORLD, &myid);
//        // only read on the host cpu
//        if(myid == 1){
//            
//            const mfem::Mesh *mfem_mesh = mfem_fespace.GetMesh();
//            mfem::Vector mfem_nodes;
//            mfem_mesh->GetNodes(mfem_nodes);
//            int nnode = mfem_nodes.Size() / ndim;
//            nodes.resize(nnode);
//
//            // copy the nodes in
//            for(int inode = 0; inode < nnode; ++inode){
//                double node[ndim];
//                mfem_mesh->GetNode(inode, node); // TODO: use the Vector we just pulled
//                for(int idim = 0; idim < ndim; ++idim){
//                    nodes[inode][idim] = static_cast<T>(node[idim]);
//                }
//            }
//
//            // Get the element connectivity
//            for(int iel = 0; iel < mfem_fespace.GetNE(); ++iel){
//                
//                const mfem::FiniteElement *el = mfem_fespace.GetFE(iel);
//                mfem::Geometry::Type eltype  = el->GetGeomType();
//                switch(eltype){
//                    case mfem::Geometry::Type::TRIANGLE: {
//                        auto createTriangle = [el]<int Pn>(int Pn_arg){
//                            if constexpr (Pn > PN_GEOM_MAX){
//                                throw std::logic_error("Order is limited to " + std::to_string(PN_GEOM_MAX));
//                            } else {
//                                ELEMENT::SimplexGeoElement<T, IDX, ndim, Pn> tri{};
//                                for(int inode = 0; inode < tri.n_nodes(); ++inode){
//                                    tri.setNode(inode, el->GetNodes().IntPoint(inode).index);
//                                }
//                            }
//                        };
//                        break;
//                    }
//
//                    default:{
//                        throw std::logic_error("Unsupported Element Type");
//                        break;
//                    }
//                }
//            }
//
//
//            
//        }
//   }

}
