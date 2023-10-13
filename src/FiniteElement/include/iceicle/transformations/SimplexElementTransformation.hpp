#pragma once

#include <Numtool/integer_utils.hpp>
#include <Numtool/point.hpp>
#include <vector>
#include <span>
#include <string>
#include <sstream>
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
        friend void _test_simplex_transform(); // give testing class access

        // === Aliases ===
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        // === Constants ===
        static constexpr int nbary = ndim + 1; // the number of barycentric coordinates needed to describe a point
        static constexpr int nnode = MATH::binomial<Pn + ndim, ndim>(); // the number of nodes
        /// The number of faces
        static constexpr int nfac = ndim + 1;

        // =================
        // = Helper funcs  =
        // = for Silvester =
        // =================

        void gen_free_index_set(
            int nfree, int maxdim, bool cgns_flip,
            std::vector<std::vector<int>> &free_index_set
        ){
            using init_list = std::initializer_list<int>;
            // base case: 2 free indices
            if(nfree == 2){
                for(int idim = nfree-1; idim <= maxdim; ++idim){
                    // generate all the free indices added in the idim-th dimension
                    for(int jdim = 0; jdim < idim; ++jdim){
                        free_index_set.emplace_back((init_list){jdim, idim});
                    }
                }

                // Manual fix for CGNS shenanigans
                // (for the 2D triangle a ccw convention is adopted and then promptly abandoned for higher dim)
                if(cgns_flip && free_index_set.size() > 2){
                    free_index_set[1] = {1, 2};
                    free_index_set[2] = {0, 2};
                }
            } else {
                for(int idim = nfree-1; idim <= maxdim; ++idim){
                    // generate all the free indices recursively
                    // [[nfree = ndim-1, maxdim = idim-1], idim]
                    std::vector<std::vector<int>> sublist{};
                    gen_free_index_set(nfree-1, idim-1, cgns_flip, sublist);
                    for(std::vector<int> &free_indices : sublist){
                        std::vector<int> &new_free_indices = free_indices;
                        new_free_indices.push_back(idim);
                        free_index_set.push_back(std::move(new_free_indices));
                    }
                }
            }
        }

        /**
         * @brief generate the Silvester numbers for nodes
         * starting with the vertices, then edge nodes, then face nodes, ...
         * @param nfree the number of "free indices" these are the index locations
         *  in the Silvester numbers that are nonzero. ndim free indices makes a vertex, ndim-1 
         *  makes an edge, and so on
         * @param free_indices the free indices to generate all the combinations for
         *  note: the full array will be passed recursively but smaller spans will be used each
         *  recursion
         * @param order_sum the maximum order of the free indices (note this is not Pn because
         *  this is a recursive algorithm
         * @param cgns_flip use the reverse order for the 3-1 edge, 
         *  this happens in the cgns standard
         * @param [out] ijk_list the pointer of where to start adding nodes
         * @return the number of nodes filled
         */
        int gen_silvester_numbers(
            int nfree,
            std::vector<int> &free_indices,
            int order_sum,
            bool cgns_flip,
            std::array<int, nbary> *ijk_list
        ){
            if(nfree == 2){
                int inode = 0;
                // base case:
                // The two Silvester numbers sum up to order_sum
                // Silvester number != 0
                // Generate all the combinations
                if(!cgns_flip)[[likely]] { // Generally we don't need the flip
                    for(int ibary = order_sum - 1; ibary >= 1; --ibary, ++inode){
                        std::array<int, nbary> &point = ijk_list[inode];
                        std::fill_n(point.begin(), nbary, 0);
                        point[free_indices[0]] = ibary;
                        point[free_indices[1]] = order_sum - ibary;
                    }
                } else {
                    // reverse the order for the 3-1 edge in cgns
                    for(int ibary = 1; ibary < order_sum; ++ibary, ++inode){
                        std::array<int, nbary> &point = ijk_list[inode];
                        std::fill_n(point.begin(), nbary, 0);
                        point[free_indices[0]] = ibary;
                        point[free_indices[1]] = order_sum - ibary;
                    }
                }
                return inode;
            } else {
                // the highest you can put in this index
                // while filling the rest with ones
                int max_bary = order_sum - nfree + 1;

                int inode = 0;
                std::array<int, nbary> *list_unfilled = ijk_list;
                for(int ibary = 1; ibary <= max_bary; ++ibary){
                    // recursive fill
                    int npoins = gen_silvester_numbers(
                            nfree-1, free_indices,
                            order_sum-ibary, cgns_flip,
                            list_unfilled
                    );
                    
                    // fill the rest with ibary and increment list_unfilled to next section
                    std::array<int, nbary> *start = list_unfilled;
                    for(;list_unfilled != start + npoins; ++list_unfilled){
                        (*list_unfilled)[free_indices[nfree-1]] = ibary;
                    }
                    inode += npoins;
                }
                return inode;
            }
        }

        void fill_nodes(
            bool cgns_flip,
            std::array<std::array<int, nbary>, nnode> &ijk_list
        ){
            // vertices
            for(int inode = 0; inode < nbary; ++inode){
                std::array<int, nbary> &point = ijk_list[inode];
                std::fill_n(point.begin(), nbary, 0);
                point[inode] = Pn;
            }

            // all interior nodes (hierarchically)
            int fill_start = nbary; // already filled vertices
            for(int nfree = 2; nfree <= nbary; ++nfree){
                std::vector<std::vector<int>> free_index_set{};
                gen_free_index_set(nfree, ndim, cgns_flip, free_index_set);

                for(auto &free_indices : free_index_set){
                    // loop over all sets of free indices and generate the silvester numbers
                    if(free_indices[0] == 0 && free_indices[1] == 2){
                        // for cgns flip the 3-1 edge
                        // note if cgns_flip = false this changes nothing
                        int fill_inc = gen_silvester_numbers(
                            nfree, free_indices, Pn, cgns_flip,
                            ijk_list.data() + fill_start);
                        fill_start += fill_inc;
                    } else {
                        int fill_inc = gen_silvester_numbers(
                            nfree, free_indices, Pn, false,
                            ijk_list.data() + fill_start);
                        fill_start += fill_inc;
                    }
                }
            }
        }

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
         * @brief Calculate the 1d shape function value at xi
         * P_m(xi) as defined by Silvester
         * @param m the shape function index
         * @param xi the function argument
         * @return T the shape function P_m(xi)
         */
        inline T shapefcn_1d(int m, T xi) const {
            if(m == 0) return (T) 1.0; // TODO: check if branchless cmov is used
            T prod = (Pn * xi); // case i = 1
            for(int i = 2; i <= m; ++i){
                prod *= (Pn * xi - i + 1) / i;
            }
            return prod;
        }

        /** @brief calculate the 1d shape function derivative
         *         using the logarithmic derivative
         *  WARNING: has singularity
         *  @param m the shape function index
         *  @param xi the function argument
         *  @return T the shape function derivative
         */
        inline T dshapefcn_1d_logder(int m, T xi) const {
            // if m is zero then derivative is zero
            if(m == 0) return (T) 0;

            // more expensive version when close logarithmic derivative singularity (doesn't catch all cases)
            if(std::abs(xi) <= 1e-8) return dshapefcn_1d(m, xi);

            // first term of logarithmic derivative version
            // product of shape functions
            T prodTerm = shapefcn_1d(m, xi);
            // sum term of the logarithmic derivatives
            T sumTerm = 0;
            for(int i = 1; i <= m; ++i){
                sumTerm += ((T) Pn) / (Pn * xi - i + 1);
            }
            return prodTerm * sumTerm;
        }

        /** @brief calculate the 1d shape function derivative
         *         using the full product rule O(Pn^2)
         *
         *  @param m the shape function index
         *  @param xi the function argument
         *  @return T the shape function derivative
         */
        inline T dshapefcn_1d(int m, T xi) const {
            
            // if m is zero then derivative is zero
            if(m == 0) return 0;

            T sum = 0;
            for(int i = 1; i <= m; ++i){
                T prod = (i == 1) ? 1 : Pn * xi; // base case j = 1
                for(int j = 2; j < i; ++j) prod *= (Pn * xi - j + 1) / j;
                for(int j = i + 1; j <= m; ++j) prod *= (Pn * xi - j + 1) / j;

                sum += (Pn * prod) / i; // prod makes sure floating point division happens
            }
            return sum;
        }


        /** @brief calculate the 1d shape function second derivative
         *         using the logarithmic derivative
         *  WARNING: has singularity
         *  @param m the shape function index
         *  @param xi the function argument
         *  @return T the shape function second derivative
         */
        inline T d2shapefcn_1d_logder(int m, T xi) const {
            // second derivative zero conditions
            if(m == 0 || m == 1) return 0;

            // more expensive version when close logarithmic derivative singularity
            if(std::abs(xi) <= 1e-8) return dshapefcn_1d(m, xi);

            // first term of logarithmic derivative version
            // product of shape functions
            T prodTerm = shapefcn_1d(m, xi);
            // sum term of the logarithmic derivatives
            T sumTerm = 0;
            for(int i = 1; i <= m; ++i){
                sumTerm += Pn / (Pn * xi - i + 1);
            }

            // sum term for logarithmic 2nd deriv
            T sumTerm2 = 0;
            for(int i = 1; i <= m; ++i){
                sumTerm2 += SQUARED(Pn) / SQUARED(Pn * xi - i + 1);
            }
            return prodTerm * sumTerm * (sumTerm - sumTerm2);
        }

        
        /** @brief calculate the 1d shape function second derivative
         *         using the product rule
         *  WARNING: does not handle the case where m is 0 or 1
         *  @param m the shape function index
         *  @param xi the function argument
         *  @return T the shape function second derivative
         */
        inline T d2shapefcn_1d(int m, T xi) const {
            // second derivative zero conditions
            if(m == 0 || m == 1) return 0;

            T sum = 0;
            for(int i = 1; i <= m; ++i){
                T inner_sum = 0;
                for(int j = 1; j <= m; ++j) if(j != i) {
                    T prod = 1;
                    for(int k = 1; k <= m; ++k) if(k != i && k !=j) {
                        prod *= (Pn * xi - k + 1) / k;
                    }
                    inner_sum += (Pn * prod) / j;
                }
                sum += (Pn * inner_sum) / i;
            }
            return sum;
        }

        public:

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
    
        private:
        /**
         * @brief Get the third area coordinate for a reference Triangle
         * given the location in the reference domain
         * 
         * @param xi the xi reference domain coordinate (1st area coordinate)
         * @param eta the eta reference domain coordinate (2nd area coordinate)
         * @return T The third area coordinate
         */
        inline T area_coord_3(T xi, T eta) const { return 1.0 - xi - eta; }


        /// The array of the Silvester numbering for each reference domain node
        std::array<std::array<int, nbary>, nnode> ijk_poin;
        /// The reference domain coordinates of each reference node
        Point xi_poin[nnode];

 //       std::vector<std::vector<int>> face_inodes; /// the local node indices of the points of each face in order

        /// provides connectivity to local inode for face endpoints
        int face_endpoints[nfac][ndim];

        /// @brief the node indices of the endpoints
        /// NOTE: Convention Note
        /// An endpoint is where the ijk_poin array has an entry whose integer value == Pn
        /// Let the index of the ijk_poin[inode] in which this occurs be the endpoint index
        int endpoint_nodes[nbary]; 


        public:
        SimplexElementTransformation() {
            
            /**
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
            */
            fill_nodes(true, ijk_poin);

            // Generate the reference domain coordinates for each reference domain node
            for(int inode = 0; inode < nnode; ++inode){
                
                for(int ivert = 0; ivert < nbary; ++ivert) {
                    if(ijk_poin[inode][ivert] == Pn) endpoint_nodes[ivert] = inode;
                }

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
        constexpr int nnodes() const {
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
        void transform(
                std::vector<Point> &node_coords,
                IDX *node_indices,
                const Point &xi, Point &x
        ) const {
            std::fill_n(&(x[0]), ndim, 0.0);
            for(int inode = 0; inode < nnode; ++inode){
                T shpval = shp(xi, inode);
                IDX node_id = node_indices[inode];
                for(int idim = 0; idim < ndim; ++idim){
                    x[idim] += node_coords[node_id][idim] * shpval;
                }
            }
        }

        /**
         * @brief get the Jacobian matrix of the transformation
         * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to this element in order
         * @param [in] xi the position in the reference domain at which to calculate the Jacobian
         * @param [out] the jacobian matrix
         */
        void Jacobian(
            std::vector<Point> &node_coords,
            const IDX *node_indices,
            const Point &xi,
            T J[ndim][ndim]
        ) const {
            // Get a 1D pointer representation of the matrix head
            T *Jptr = J[0];

            // fill with zeros
            std::fill_n(Jptr, ndim * ndim, 0.0);

            for(int inode = 0; inode < nnode; ++inode){
                IDX global_inode = node_indices[inode];
                const Point &node = node_coords[global_inode];
                for(int idim = 0; idim < ndim; ++idim) { // idim corresponds to x
                    for(int jdim = 0; jdim < ndim; ++jdim) { // jdim corresponds to \xi
                        J[idim][jdim] += dshp(xi, inode, jdim) * node[idim];
                    }
                }
            }
        }

        /**
         * @brief get the Hessian of the transformation
         * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
         *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] node_indices the indices in node_coords that pretain to this element in order
         * @param [in] xi the position in the reference domain at which to calculate the hessian
         * @param [out] the Hessian in tensor form indexed [k][i][j] as described above
         */
        void Hessian(
            std::vector<Point> &node_coords,
            const IDX *node_indices,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const {
            // Get a 1D pointer representation
            T *Hptr = hess[0][0];

            // fill with zeros
            std::fill_n(Hptr, ndim * ndim * ndim, 0.0);

            for(int inode = 0; inode < nnode; ++inode){
                IDX global_inode = node_indices[inode];
                const Point &node = node_coords[global_inode];
                for(int kdim = 0; kdim < ndim; ++kdim) { // k corresponds to xi
                    for(int idim = 0; idim < ndim; ++idim) {
                        for(int jdim = idim; jdim < ndim; ++jdim) { // fill symmetric part later
                            // H_{kij} = \sum\limits_p \frac{\partial N_p(\xi)}{\partial \xi_i \partial \xi_j} x_{p, k}
                            hess[kdim][idim][jdim] += dshp2(xi, inode, idim, jdim) * node[kdim];
                        }
                    }
                }
            }

            // finish filling symmetric part
            for(int kdim = 0; kdim < ndim; ++kdim){
                for(int idim = 0; idim < ndim; ++idim){
                    for(int jdim = 0; jdim < idim; ++jdim){
                        hess[kdim][idim][jdim] = hess[kdim][jdim][idim];
                    }
                }
            }
        }

        /**
         * @brief get a pointer to the array of Lagrange points in the reference domain
         * @return the Lagrange points in the reference domain
         */
        const Point *reference_nodes() const { return xi_poin; }

        // ===============
        // = Diagnostics =
        // ===============

        std::string print_ijk_poin(){
            using namespace std;
            ostringstream pt_ss{};
            for(int inode = 0; inode < nnodes(); ++inode){
                pt_ss << inode << ". [ ";
                for(int ibary = 0; ibary < nbary; ++ibary){
                    pt_ss << ijk_poin[inode][ibary] << " ";
                }
                pt_ss << "]" << endl;
            }
            return pt_ss.str();
        }
    };


    /**
     * @brief transformation from the reference trace space
     * to the left and right reference element space
     *
     * note: this is a reference space to reference space transformation
     * @tparam T the floating point type
     * @tparam IDX the indexing type for large lists
     * @tparam ndim the number of dimensions (for the element space)
     * @tparam Pn the polynomial order of the elements
     */
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexTraceTransformation {
        private:
        static constexpr int trace_ndim = ndim - 1;
        using ElPoint = MATH::GEOMETRY::Point<T, ndim>;
        using TracePoint = MATH::GEOMETRY::Point<T, trace_ndim>;
        
        public:

        /**
         * @brief transform from the trace space reference domain 
         *        to the left and right element reference domain
         *
         * WARNING: This assumes the vertices are the first ndim+1 nodes
         * This is the current ordering used in SimplexElementTransformation
         * @param [in] node_indicesL the global node indices for the left element
         * @param [in] node_indicesR the global node indices for the right element
         * @param [in] faceNrL the left trace number 
         * (the position of the barycentric coordinate that is 0 for all points on the face)
         * @param [in] faceNrR the right trace number (see faceNrL)
         * @param [in] s the location in the trace reference domain
         * @param [out] xiL the position in the left reference domain
         * @param [out] xiR the position in the right reference domain
         */
        void transform(
                IDX *node_indicesL,
                IDX *node_indicesR,
                int traceNrL, int traceNrR,
                const TracePoint &s,
                ElPoint &xiL, ElPoint &xiR
        ) const {
            // Generate the barycentric coordinates for the TracePoint
            T sbary[trace_ndim + 1];
            sbary[trace_ndim] = 1.0;
            for(int idim = 0; idim < trace_ndim; ++idim){
                sbary[idim] = s[idim];
                sbary[trace_ndim] -= s[idim];
            }

            // Left element
            // set the barycentric coordinate to 0
            // unless this is out of bounds (the modular index will just get overwritten)
            xiL[traceNrL % ndim] = 0;
            // copy over the barycentric coordinates, skipping faceNr
            for(int idim = 0; idim < traceNrL; ++idim)
                { xiL[idim] = sbary[idim]; }
            for(int idim = traceNrL + 1; idim < ndim; ++idim)
                { xiL[idim] = sbary[idim - 1]; }

            // Right element
            // set the barycentric coordinate to 0
            // unless this is out of bounds (the modular index will just get overwritten)
            xiR[traceNrR % ndim] = 0;
            static constexpr int nvert = ndim + 1;
            // find the vertex indices where the global node index is the same
            // O(n^2) but probably better than nlogn algorithms because of small size
            for(int ivert = 0; ivert < nvert - 1; ++ivert){ // don't need last bary coord
                for(int jvert = 0; jvert < nvert; ++jvert){
                    if(node_indicesR[ivert] == node_indicesL[jvert]){
                        xiR[ivert] = xiL[jvert]; // copy the matching barycentric coordinate
                        continue; // move on to the next right element vertex
                    }
                }
            }
        }
        
    };
}
