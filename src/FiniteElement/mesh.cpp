/**
 * @file mesh.cpp
 * @brief mesh implementation
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include <iceicle/mesh/mesh.hpp> 
#include <iceicle/build_config.hpp>
#include <iceicle/geometry/face.hpp>
#include <iceicle/geometry/simplex_element.hpp>
#include <string>
#include <map>
#include <sstream>
#include <fstream>
#include <iostream>
#include <exception>
#include <stdexcept>
#include "mpi.h"

namespace MESH {
    
    using T = BUILD_CONFIG::T;
    using IDX = BUILD_CONFIG::IDX;

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

    template<>
    AbstractMesh<T, IDX, 2>::AbstractMesh(std::string_view filepath)
    : nodes{}, elements{}, faces{} {
        
        static constexpr int ndim = 2;
        std::ifstream infile{std::string(filepath)};
        if(!infile){
            throw std::logic_error("Could not read file: " + std::string(filepath) + "\n");
        }
        std::cout << "Reading Gmsh File: " << filepath << "\n";
        // toss all lines until $PhysicalNames
        for(std::string line; std::getline(infile, line);){
            if(line.find("$PhysicalNames") != std::string::npos) break;
        }

        // catalogue the tags to boundary conditions
        IDX ignore;
        int ntags;
        std::map<int, int> gmshTagMap;
        infile >> ntags;
        for(int itag = 0; itag < ntags; ++itag){
            int gmshTag;
            std::string tagname;
            infile >> ignore >> gmshTag >> tagname;
            tagname.erase(remove(tagname.begin(), tagname.end(), '\"'), tagname.end());
            gmshTagMap[gmshTag] = itag;

            // get BC from tagnaame;
            if(tagname == "wall"){
                gmshTagMap[gmshTag] = ELEMENT::BOUNDARY_CONDITIONS::WALL_GENERAL;
            } else if (tagname == "inlet"){
                gmshTagMap[gmshTag] = ELEMENT::BOUNDARY_CONDITIONS::INLET;
            } else if (tagname == "outlet"){
                gmshTagMap[gmshTag] = ELEMENT::BOUNDARY_CONDITIONS::OUTLET;
            } else if (tagname == "Dirichlet"){
                gmshTagMap[gmshTag] = ELEMENT::BOUNDARY_CONDITIONS::DIRICHLET;
            } else {
                throw std::logic_error("unrecognized bc type: " + tagname + "\n");
            }
        }

        // toss all lines until $Entities
        for(std::string line; std::getline(infile, line);){
            if(line.find("$Entities") != std::string::npos) break;
        }        
        IDX npt, ncurv, nsurf, nvol;
        IDX ignoreidx;
        T ignoreflt;
        infile >> npt >> ncurv >> nsurf >> nvol;

        // skip points
        for(int ipt = 0; ipt < npt; ++ipt){
            int ntags;
            infile >> ignoreidx >> ignoreflt >> ignoreflt >> ignoreflt >> ntags;
            for(int itag = 0; itag < ntags; ++itag) infile >> ignoreidx;
        }


        std::map<int, std::vector<int>> curv_tagmap;
        for(int icurv = 0; icurv < ncurv; ++icurv){
            int ntags, nbpoin;
            IDX curvtag;
            infile >> curvtag 
                   >> ignoreflt >> ignoreflt >> ignoreflt // min pt
                   >> ignoreflt >> ignoreflt >> ignoreflt  // max pt
                   >> ntags;
            curv_tagmap[curvtag] = std::vector<int>();
            int tagidx;
            for(int itag = 0; itag < ntags; ++itag) {
                infile >> tagidx;
                curv_tagmap[curvtag].push_back(tagidx);
            }
            // ignore bounding points
            infile >> nbpoin;
            for(int ibpoin = 0; ibpoin < nbpoin; ++ibpoin) infile >> ignoreidx;
        }

        // toss all lines until $Nodes
        for(std::string line; std::getline(infile, line);){
            if(line.find("$Nodes") != std::string::npos) break;
        }        

        IDX neblocks, nnodes;
        infile >> neblocks >> nnodes >> ignore >> ignore;
        std::cout << "nnodes: " << nnodes << "\n";
        nodes.resize(nnodes);
        for(IDX ientity = 0; ientity < neblocks; ++ientity){
            IDX nblocknodes;
            infile >> ignore >> ignore >> ignore >> nblocknodes;
            std::vector<IDX> nodeTags;
            for(IDX inodetag = 0; inodetag < nblocknodes; ++inodetag){
                int nodetag;
                infile >> nodetag;
                nodeTags.push_back(nodetag);
            }

            for(IDX inode = 0; inode < nblocknodes; ++inode){
                T x, y, z;
                infile >> x >> y >> z;
                IDX nodetag_shifted = nodeTags[inode] - 1; // shift for 1 indexing
                nodes[nodetag_shifted][0] = x;
                nodes[nodetag_shifted][1] = y;
            }
        }

        // toss all lines until $Elements
        for(std::string line; std::getline(infile, line);){
            if(line.find("$Elements") != std::string::npos) break;
        }        

        IDX nelem, ifacglobal = 0;
        infile >> neblocks >> nelem >> ignore >> ignore;

        for(IDX iblock = 0; iblock < neblocks; ++iblock){
            IDX entitydim, entityTag, elementType, nelemBlock;
            infile >> entitydim >> entityTag >> elementType >> nelemBlock;

            if(entitydim == 0){
                for(int ipoin = 0; ipoin < nelemBlock; ++ipoin){
                    infile >> ignore >> ignore; // ignore both tag and node number
                }
            } else if (entitydim == 1){
                for(int ifac = 0; ifac < nelemBlock; ++ifac){
                    if(elementType == 1){
                        // 2-point line
                        IDX node1, node2;
                        infile >> ignore >> node1 >> node2;

                        // shift nodes for 0 indexing
                        node1--; node2--;

                        // add the face to the list of faces
                        // haven't found elemL yet or orientations
                        // this will be done after all element processing
//                        auto facptr = std::make_unique<ELEMENT::SegmentFace<T, IDX>>(
//                                node1, node2, -1, -1, -1, -1, -1, -1
//                        );
                        // add pointer to the boundary first
                        IDX phys_name_tag = curv_tagmap[entityTag][0]; // assume first physical tag is the BC 
//                        auto &bdy = boundaries[gmshTagMap[phys_name_tag]];
//                        bdy.faces.push_back(facptr.get());
//                        bdy.bcflags.push_back(-1);
//                        bdy.bdyFacIdx.push_back(ifacglobal);
//                        faces.push_back(std::move(facptr));
                        ifacglobal++;
                    } else if (elementType == 8){
                        // 3-point line
//                        ERR::throwWarning("2nd order faces not yet supported");
                    }
                }
            } else if (entitydim == 2){
                for(IDX ielem = 0; ielem < nelemBlock; ++ielem){
                    if(elementType == 2){
                        // linear triangle
                        IDX eltag, node1, node2, node3;
                        infile >> eltag >> node1 >> node2 >> node3;

                        // shift nodes for 0 indexing
                        node1--; node2--; node3--;

                        // make element
//                        elements.push_back(
//                            std::move(std::make_unique<ELEMENT::Triangle<T, IDX, 1>>(
//                                std::initializer_list<IDX>{node1, node2, node3}
//                            ))
//                        );
                    } else {
//                        ERR::throwWarning("Unsupported Element Type " + std::to_string(elementType));
                    }
                }

            } else {
//                ERR::throwError("Cannot process 3D mesh file.");
            }
        }
    }
    
}
