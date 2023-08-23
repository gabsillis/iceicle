#pragma once

#include <Numtool/integer_utils.hpp>
#include <Numtool/point.hpp>
#include <complex>
#include <vector>

namespace ELEMENT::TRANSFORMATIONS {
    
    /**
     * @brief Transformation for an arbitrary simplex to the reference simplex
     *
     * @tparam T the floating point type
     * @tparam IDX the index type for large integers
     * @tparam ndim the number of dimensions
     * @tparam Pn the polynomial order of the simplex
     */
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexElementTransformation {
       private:
        // === Aliases ===
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        // =================
        // = Helper funcs  =
        // = for Silvester =
        // =================

        template<int narg>
        int ijk_sum(int ijk[narg]){
            int sum = 0;
            for(int i = 0; i < narg; ++i) sum += ijk[i];
            return sum;
        }

        template<int narg>
        void ijk_increment(
            int ijk_old[narg], int ijk_new[narg], 
            int endpoint_nodes[ndim + 1], int inode, int N
        ){
            // fill with zeros
            for(int i = 0; i < narg; ++i) ijk_new[i] = 0;
            int end = narg - 1;
            if(ijk_old[end - 1] > 0){
                for(int i = 0; i < end - 1; i++) ijk_new[i] = ijk_old[i];
                ijk_new[end - 1] = ijk_old[end - 1] - 1;
                if constexpr(narg - 1 > 0){
                    if(ijk_sum<narg - 1>(ijk_new) == 0) endpoint_nodes[end] = inode;
                    ijk_new[end] = N - ijk_sum<narg - 1>(ijk_new);
                }
            } else {
                if constexpr(narg - 1 > 0)
                    ijk_increment<narg - 1>(ijk_old, ijk_new, endpoint_nodes, inode, N);
                ijk_new[end] = 0;
            }
        }

        // ===================
        // = Shape Functions =
        // ===================
        /**
         * @brief Calculate the shape function product for the argument (xi)
         * 
         * @param xi the n-dimensional argument (gets converted to area cordinates implicitly)
         * @param inode the node to calculate the shape function for
         * @return T the shape function N for the given node (see iceicle reference for more details)
         */
        T shp(const T *xi, int inode) const {
            T lambda_n = 1;
            int m = ijk_poin[inode][0];
            T prod = (m == 0) ? 1.0 : shapefcn_1d(m, xi[0]);
            lambda_n -= xi[0];
            for(int idim = 1; idim < ndim; ++idim){
                m = ijk_poin[inode][idim];
                prod *= (m == 0) ? 1.0 : shapefcn_1d(m, xi[idim]);
                lambda_n -= xi[idim];
            }
            m = ijk_poin[inode][ndim];
            prod *= (m == 0) ? 1.0 : shapefcn_1d(m, lambda_n);
            return prod;
        }

        /**
         * @brief Calculate the first derivative of the shape function at the point xi in the reference domain
         *
         * @param xi the n-dimensional point in the reference domain
         * @param inode the node number (basis number) to calculate the shape functio for
         * @param ideriv the direction index of the derivative
         * @return the derivative for the given node in the given direction at xi
         */
        T dshp(const T *xi, int inode, int ideriv) const {
            if constexpr(ndim == 2){
                int i = ijk_poin[inode][0];
                int j = ijk_poin[inode][1];
                int k = ijk_poin[inode][2];
                T lambda_3 = 1 - xi[0] - xi[1];
                if(ideriv == 0){ // xi derivative
                    return ( 
                        dshapefcn_1d(i, xi[0]) * shapefcn_1d(j, xi[1]) * shapefcn_1d(k, lambda_3)
                        - shapefcn_1d(i, xi[0]) * shapefcn_1d(j, xi[1]) * dshapefcn_1d(k, lambda_3)
                    );
                } else { // eta derivative
                    return ( 
                        shapefcn_1d(i, xi[0]) * dshapefcn_1d(j, xi[1]) * shapefcn_1d(k, lambda_3)
                        - shapefcn_1d(i, xi[0]) * shapefcn_1d(j, xi[1]) * dshapefcn_1d(k, lambda_3)
                    );
                }
            } else {
                T lambda_n = 1;
                for(int i = 0; i < ndim; ++i) lambda_n -= xi[i];
                T term1 = dshapefcn_1d(ijk_poin[inode][ideriv], xi[ideriv]);
                for(int idim = 0; idim < ideriv; ++idim)
                    term1 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                for(int idim = ideriv + 1; idim < ndim; ++idim)
                    term1 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                term1 *= shapefcn_1d(ijk_poin[inode][ndim], lambda_n);

                T term2 = dshapefcn_1d(ijk_poin[inode][ndim], lambda_n);
                for(int idim = 0; idim < ndim; ++idim)
                    term2 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
            
                return term1 - term2;
            }
        }

        T dshp2(const T *xi, int inode, int ideriv, int jderiv) const {
            T term1, term2;
            T lambda_n = 1;
            for(int i = 0; i < ndim; ++i) lambda_n -= xi[i];
            if(ideriv == jderiv){ // use second derivative
                term1 = shapefcn_1d(ijk_poin[inode][ndim], lambda_n);
                for(int idim = 0; idim < ndim; ++idim){
                    if(idim == ideriv){
                        term1 *= d2shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                    } else {
                        term1 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                    }
                }
            } else { // 2 1st derivatives in different indices
                term1 = shapefcn_1d(ijk_poin[inode][ndim], lambda_n);
                for(int idim = 0; idim < ndim; ++idim){
                    if(idim == ideriv || idim == jderiv){
                        term1 *= dshapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                    } else {
                        term1 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);
                    }
                }
            }

            term2 = d2shapefcn_1d(ijk_poin[inode][ndim], lambda_n);
            for(int idim = 0; idim < ndim; ++idim)
                term2 *= shapefcn_1d(ijk_poin[inode][idim], xi[idim]);

            return term1 - term2;
        }

        /**
         * @brief Get the third area coordinate for a reference Triangle
         * given the location in the reference domain
         * 
         * @param xi the xi reference domain coordinate (1st area coordinate)
         * @param eta the eta reference domain coordinate (2nd area coordinate)
         * @return T The third area coordinate
         */
        inline T area_coord_3(T xi, T eta) const { return 1.0 - xi - eta; }


        static constexpr int nbary = ndim + 1; // the number of barycentric coordinates needed to describe a point
        static constexpr int nnode = MATH::binomial<Pn + ndim, ndim>(); // the number of nodes
        /// The number of faces
        static constexpr int nfac = ndim + 1;
        /// The array of the Silvester numbering for each reference domain node
        int ijk_poin[nnode][ndim + 1];
        /// The reference domain coordinates of each reference node
        Point xi_poin[nnode];

        std::vector<std::vector<int>> face_inodes; /// the local node indices of the points of each face in order

        /// provides connectivity to local inode for face endpoints
        int face_endpoints[nfac][ndim];

        /// @brief the node indices of the endpoints
        /// NOTE: Convention Note
        /// An endpoint is where the ijk_poin array has an entry whose integer value == Pn
        /// Let the index of the ijk_poin[inode] in which this occurs be the endpoint index
        int endpoint_nodes[ndim + 1]; 


        public:
        SimplexElementTransformation() : face_inodes(nfac) {
            // Generate the Silvester numbering for each reference domain node
            ijk_poin[0][0] = Pn;
            endpoint_nodes[0] = 0;
            for(int i = 1; i < ndim + 1; ++i){
                ijk_poin[0][i] = 0;
                face_inodes[i].push_back(0); // when shp is zero that is on the corresponding face number
            }
            for(int inode = 1; inode < nnode; ++inode){
                ijk_increment<ndim + 1>(
                    ijk_poin[inode - 1], ijk_poin[inode], 
                    endpoint_nodes, inode, Pn
                );

                // add the face_inode if applicable
                for(int ishp = 0; ishp < nbary; ++ishp){
                    if(ijk_poin[inode][ishp] == 0){ 
                        // when a shape function is zero, 
                        // then that point is on the corresponding face number
                        // of the shape number index
                        face_inodes[ishp].push_back(inode);
                    }
                }
            }

            // Generate the reference domain coordinates for each reference domain node
            for(int inode = 0; inode < nnode; ++inode){
                for(int idim = 0; idim < ndim; ++idim){
                    T xi_idim = ((T) ijk_poin[inode][idim]) / Pn;
                    xi_poin[inode][idim] = xi_idim;
                }
            }

            for(int facenr = 0; facenr < nfac; ++facenr){
                int facevert = 0;
                for(int ivert = 0; ivert < nfac; ++ivert){
                    if(ivert != facenr) {
                        face_endpoints[facenr][facevert] = endpoint_nodes[ivert];
                        ++facevert;
                    }
                }
            }
        }

        /** @brief get the number of nodes that define the element */
        constexpr int nnodes(){
            return nnode;
        }

        /**
         * @brief transform from the reference domain to the physcial domain
         * T(s): s -> x
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to the element
         * @param [in] xi the position in the refernce domain
         * @param [out] x the position in the physical domain
         */
        void Transform(
                std::vector<Point> &node_coords,
                std::vector<IDX> &node_indices,
                T *xi, T *x
        ){
            x[0] = 0;
            x[1] = 0;
            for(int inode = 0; inode < nnode; ++inode){
                T shpval = shp(xi, inode);
                IDX node_id = node_indices[inode];
                x[0] += node_coords[node_id][0] * shpval;
                x[1] += node_coords[node_id][1] * shpval;
            }
        }
    };
}
