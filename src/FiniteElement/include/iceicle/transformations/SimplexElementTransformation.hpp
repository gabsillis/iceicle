#pragma once

#include <Numtool/integer_utils.hpp>
#include <Numtool/fixed_size_tensor.hpp>
#include <iceicle/fe_definitions.hpp>
#include <algorithm>
#include <vector>
#include <string>
#include <sstream>
namespace iceicle::transformations {

    template<class T, class IDX>
    struct triangle { 
        
        static constexpr int ndim = 2;
        static constexpr int nshp = ndim + 1;
        static constexpr int nfac = 3;
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;


        /// @brief extract the element coordinates from the global coordinate array and node indices
        static constexpr
        auto get_el_coord(const NodeArray<T, ndim>& coord, const IDX* nodes) noexcept
        -> std::vector<Point> {
            return std::vector<Point>{coord[nodes[0]], coord[nodes[1]], coord[nodes[2]]};
        }

        /**
        * @brief transform from the reference domain to the physcial domain
        * T(s): s -> x
        * @param [in] el_coord the coordinates of each node for the element
        * element
        * @param [in] xi the position in the refernce domain
        * @param [out] x the position in the physical domain
        */
        static constexpr
        auto transform(std::span<Point> el_coord, const Point &pt_ref) noexcept -> Point {

            Point pt_phys{};

            std::array<T, nshp> lambdas = {
                pt_ref[0], 
                pt_ref[1],
                1.0 - pt_ref[0] - pt_ref[1]
            };

            std::ranges::fill(pt_phys, 0.0);
            for(int ishp = 0; ishp < nshp; ++ishp) {
                pt_phys[0] += lambdas[ishp] * el_coord[ishp][0];
                pt_phys[1] += lambdas[ishp] * el_coord[ishp][1];
            }

            return pt_phys;
        }


        /**
        * @brief get the Jacobian matrix of the transformation
        * J = \frac{\partial T(s)}{\partial s} = \frac{\partial x}[\partial \xi}
        *
        * @param [in] el_coord the coordinates of each node for the element
        * @param [in] xi the position in the reference domain at which to calculate the Jacobian
        * @return the Jacobian matrix
        */
        static constexpr
        auto jacobian(
            std::span<Point> el_coord,
            const Point &xi
        ) noexcept -> JacobianType {

            T x0 = el_coord[0][0], x1 = el_coord[1][0], 
              x2 = el_coord[2][0];
            T y0 = el_coord[0][1], y1 = el_coord[1][1], 
              y2 = el_coord[2][1];

            JacobianType J {{
                { x0 - x2, x1 - x2 },
                { y0 - y2, y1 - y2 }
            }};
            return J;
        }

        /**
        * @brief get the Hessian of the transformation
        * H_{kij} = \frac{\partial T(s)_k}{\partial s_i \partial s_j} 
        *         = \frac{\partial x_k}{\partial \xi_i \partial \xi_j}
        * @param [in] node_coords the coordinates of all the nodes
        * @param [in] node_indices the indices in node_coords that pretain to this element in order
        * @param [in] xi the position in the reference domain at which to calculate the hessian
        * @return the Hessian in tensor form indexed [k][i][j] as described above
        */
        static constexpr
        auto hessian(
            std::span<Point> el_coord,
            const Point &xi
        ) noexcept -> HessianType {
            HessianType H;
            H = 0;
            return H;
        }


        // =====================
        // = Face Connectivity =
        // =====================

        static constexpr
        auto face_domain_type(int face_number) noexcept 
        -> DOMAIN_TYPE 
        {
            // we use the hypercube domain type for the line segment face 
            // x \in [-1, 1]
            return DOMAIN_TYPE::HYPERCUBE;
        };

        static constexpr
        auto n_face_vert( int face_number ) noexcept 
        -> int 
        { return 2; }

        static constexpr 
        auto get_face_vert( int face_number, std::span<IDX> el_nodes ) noexcept
        -> std::vector<IDX> {
            std::vector<IDX> vert_fac(2);
            // face nodes are the indices that are not the face number 
            // i.e face 0 has nodes 1 and 2
            switch(face_number){
                case 0:
                    vert_fac[0] = el_nodes[1];
                    vert_fac[1] = el_nodes[2];
                    break;
                case 1:
                    vert_fac[0] = el_nodes[2];
                    vert_fac[1] = el_nodes[0];
                    break;
                case 2:
                    vert_fac[0] = el_nodes[0];
                    vert_fac[1] = el_nodes[1];
            }
            return vert_fac;
        }

        static constexpr
        auto n_face_nodes(int face_number) noexcept 
        -> int 
        { return 2; }

        static constexpr
        auto get_face_nodes(
            int face_number,
            std::span<IDX> el_nodes
        ) noexcept -> std::vector<IDX> 
        {
            std::vector<IDX> nodes_fac(2);
            switch(face_number){
                case 0:
                    nodes_fac[0] = el_nodes[1];
                    nodes_fac[1] = el_nodes[2];
                    break;
                case 1:
                    nodes_fac[0] = el_nodes[2];
                    nodes_fac[1] = el_nodes[0];
                    break;
                case 2:
                    nodes_fac[0] = el_nodes[0];
                    nodes_fac[1] = el_nodes[1];
            }
            return nodes_fac;
        }

        static constexpr
        auto get_face_nr(
            std::span<IDX> vert_fac,
            std::span<IDX> el_nodes
        ) noexcept -> int 
        {
            if(vert_fac[0] == el_nodes[0]){
                if(vert_fac[1] == el_nodes[1])
                    return 2;
                else 
                    return 1;
            } else if (vert_fac[0] == el_nodes[1]) {
                if(vert_fac[1] == el_nodes[0])
                    return 2;
                else 
                    return 0;
            } else {
                if(vert_fac[1] == el_nodes[0])
                    return 1;
                else 
                    return 0;
            }
            return -1;
        }
    };

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
        using PointView = MATH::GEOMETRY::PointView<T, ndim>;

        // === Constants ===
        static constexpr int nbary = ndim + 1; // the number of barycentric coordinates needed to describe a point
        static constexpr int nnode = MATH::binomial<Pn + ndim, ndim>(); // the number of nodes
        /// The number of faces
        static constexpr int nfac = ndim + 1;
        /// the number of vertices (endpoints)
        static constexpr int nvert_el = ndim + 1;
        /// the number of vertices on a face (endpoints)
        static constexpr int nvert_tr = ndim;

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
            if(m == 0) {
                T ret;
                ret = 1.0; // friendlier to AD 
                return ret; // TODO: check if branchless cmov is used
            }
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

        /**
         * @brief Calculate the second derivative of the shape function at the point xi in the reference domain
         *
         * @param xi the n-dimensional point in the reference domain
         * @param inode the node number (basis number) to calculate the shape functio for
         * @param ideriv the direction index of the derivative
         * @param jderiv the direction index for the second derivative
         * @return the second derivative for the given node in the given directions at xi
         */
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
                    // Make conversion friendlier for AD 
                    T poin_real;
                    poin_real = ijk_poin[inode][idim];
                    T xi_idim = (poin_real) / Pn;
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
                NodeArray<T, ndim> &node_coords,
                const IDX *node_indices,
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
         * @return the Jacobian matrix
         */
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
            NodeArray<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi
        ) const {
            using namespace NUMTOOL::TENSOR::FIXED_SIZE;
            // Get a 1D pointer representation of the matrix head
            Tensor<T, ndim, ndim> J = {0};

            for(int inode = 0; inode < nnode; ++inode){
                IDX global_inode = node_indices[inode];
                const Point& node{node_coords[global_inode]};
                for(int idim = 0; idim < ndim; ++idim) { // idim corresponds to x
                    for(int jdim = 0; jdim < ndim; ++jdim) { // jdim corresponds to \xi
                        J[idim][jdim] += dshp(xi, inode, jdim) * node[idim];
                    }
                }
            }
            return J;
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
            NodeArray<T, ndim> &node_coords,
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
                const Point& node{node_coords[global_inode]};
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
         * @brief get global node indices for a trace in order 
         * given the element global node indices 
         * @param [in] trace number the number that identifies the face 
         *        (barycentric coordinate that = 0)
         * @param [in] vertices_el the element global node indices of the 
         *             vertices in order 
         *             NOTE: this will be the first nvert_el indices 
         *             in the global node array for simplices 
         *
         * @param [out] vertices_tr the global node indices of the vertices on the face 
         */
        void getTraceVertices(int traceNr, const IDX vertices_el[nvert_el], IDX vertices_fac[nvert_tr]) const {
            for(int ivert = 0; ivert < traceNr; ++ivert){
                vertices_fac[ivert] = vertices_el[ivert];
            }
            for(int ivert = traceNr + 1; ivert < nvert_el; ++ivert){
                vertices_fac[ivert - 1] = vertices_el[ivert];
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
     * @brief transformation from the global reference trace space to the 
     *        reference trace space of the right element
     *        (the left element trace is defined to be equivalent to the global orientation)
     * @tparam T the floating point type
     * @tparam IDX the indexing type for large lists
     * @tparam ndim the number of dimensions (for the element)
     */
    template<typename T, typename IDX, int ndim>
    class SimplexTraceOrientTransformation {
        static constexpr int trace_ndim = ndim - 1;
        static constexpr int nvert_tr = ndim;

        std::vector<std::vector<IDX>> vertex_permutations;

        using TracePoint = MATH::GEOMETRY::Point<T, trace_ndim>;
        public:

        static constexpr int norient = MATH::factorial<nvert_tr>(); // number of permutations
        
        SimplexTraceOrientTransformation(){
            std::vector<IDX> first_perm{};
            for(int i = 0; i < nvert_tr; ++i) first_perm.push_back(i);

            vertex_permutations.push_back(first_perm);
            vertex_permutations.push_back(first_perm);
            int iperm = 1;
            while(std::next_permutation(vertex_permutations[iperm].begin(), vertex_permutations[iperm].end())){
                // make a copy of this permutation to get the next one
                vertex_permutations.push_back(vertex_permutations[iperm]);
                ++iperm;
            }
        }

        /** 
         * @brief get the orientation of the right element given 
         * the vertices for the left and right element 
         * NOTE: the orientation for the left element is always 0 
         *
         * @param verticesL the node indices for the left element 
         * @param verticesR the node indices for the right element 
         * @return the orientation for the right element or -1 if the nodes don't match
         */
        int getOrientation(
            IDX verticesL[nvert_tr],
            IDX verticesR[nvert_tr]
        ) const {
            // TODO: exploit lexographic structure of permutations for efficiency

            // loop through all the orientation permutations
            for(int i = 0; i < norient; ++i){

                bool match = true;
                // loop over all the nodes (to check for issues)
                for(int j = 0; j < nvert_tr; ++j){
                    if(verticesR[vertex_permutations[i][j]] != verticesL[j]){
                        match = false;
                        continue;
                    }
                }

                // if all the nodes match this permutation is the right one 
                // and defines the orientation
                if(match) return i;
            }

            return -1;
        }

        /**
         * @brief transform from the reference trace space
         *  to the right reference trace space
         *
         *  The reference trace space oriented with the left element space is 
         *  defined to be the same orientation as the general reference trace space
         *  so only the right element orientation needs to be considered
         *
         * @param [in] orientationR the orientation of the right element
         * @param [in] s the position in the reference trace space
         * @param [out] the posotion in the local reference trace space for the right element
         */
        void transform(
            int orientationR,
            const TracePoint& s,
            TracePoint& sR
        ) const {
            // use the property of the barycentric coordinates of triangles
            // that we can copy the barycentric coordinates for matching nodes to get the oriented barycentric coords
            T baryL[nvert_tr];
            baryL[nvert_tr - 1] = 1.0;
            for(int idim = 0; idim < trace_ndim; ++idim){
                baryL[idim] = s[idim];
                baryL[nvert_tr - 1] -= s[idim];
            }

            for(int idim = 0; idim < trace_ndim; ++idim){
                sR[idim] = baryL[vertex_permutations[orientationR][idim]];
            }
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
     */
    template<typename T, typename IDX, int ndim>
    class SimplexTraceTransformation {
        
        public:
        static constexpr int trace_ndim = ndim - 1;
        static constexpr int nvert_el = ndim + 1;
        static constexpr int nvert_tr = ndim;
        static constexpr int ntrace = nvert_el;

        private:
        using ElPoint = MATH::GEOMETRY::Point<T, ndim>;
        using TracePoint = MATH::GEOMETRY::Point<T, trace_ndim>;
        using FaceNodeTensorType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<long, ntrace, nvert_tr + 1>;

        // the face node incides that preserve ccw orientation
        // WARNING: not correct
        inline static FaceNodeTensorType face_node_orders = 
            [](){
                FaceNodeTensorType ret{};
                for(int iface = 0; iface < ntrace; ++iface){
                    ret[iface][nvert_tr] = iface; // last index is the face number 
                   
                    // fill with nodes of that face in 
                    // lexographic order 
                    for(int j = 0; j < iface; ++j)
                        { ret[iface][j] = j; }
                    for(int j = iface; j < nvert_tr; ++j)
                        { ret[iface][j] = j + 1; }

                    // check parity of the entire list
                    int n_negative = 0;
                    for(int i = 0; i < nvert_tr + 1; ++i) {
                        for(int j = 0; j < i; ++j){
                            // multiply signs of difference (in effect)
                            // convert to long for negative
                            if(( (long) ret[iface][i] - (long) ret[iface][j]) < 0) {
                                n_negative++;
                            }
                        }
                    }

                    // if negative parity, swap the last two nodes 
                    // (last in this case doesn't include the face number)
                    if(n_negative % 2 != 0){
                        std::swap(ret[iface][nvert_tr - 1], ret[iface][nvert_tr - 2]);
                        // also keep track of this by negating face number sign 
                        ret[iface][nvert_tr] = -ret[iface][nvert_tr];
                    }
                }

                return ret;
            }();

        public:

        /**
         * @brief transform from the trace space reference domain 
         *        to the left and right element reference domain
         *
         * WARNING: This assumes the vertices are the first ndim+1 nodes
         * This is the current ordering used in SimplexElementTransformation
         *
         * @param [in] node_indices the global node indices for the element
         * @param [in] faceNr the trace number
         * (for simplices: the position of the barycentric coordinate that is 0 for all points on the face)
         * @param [in] s the location in the trace reference domain
         * @param [out] xi the position in the reference element domain
         */
        void transform(
                IDX *node_indices,
                int traceNr,
                const TracePoint &s,
                ElPoint& xi
        ) const {
            // Generate the barycentric coordinates for the TracePoint
            T sbary[trace_ndim + 1];
            sbary[trace_ndim] = 1.0;
            for(int idim = 0; idim < trace_ndim; ++idim){
                sbary[idim] = s[idim];
                sbary[trace_ndim] -= s[idim];
            }

            // set the barycentric coordinate to 0
            // unless this is out of bounds (the modular index will just get overwritten)
            xi[traceNr % ndim] = 0;
            for(int idim = 0; idim < traceNr; ++idim)
                { xi[idim] = sbary[idim]; }
            for(int idim = traceNr + 1; idim < ndim; ++idim)
                { xi[idim] = sbary[idim - 1]; }

            if constexpr(ndim > 1) {
                std::size_t lc_indices[ndim];
                lc_indices[0] = traceNr;
                for(int idim = 0; idim < trace_ndim; ++idim){
                    if(idim < traceNr){
                    lc_indices[idim + 1] = idim;
                    } else {
                    lc_indices[idim + 1] = idim + 1;
                    }
                }

                T lc = NUMTOOL::TENSOR::FIXED_SIZE::levi_civita<T, ndim>.list_index(lc_indices);
                // for every face number except the last 
                // the normal vector is in the negative coordinate direction 
                // of that face number 
                if(traceNr < ndim){
                    // correct the sign with negative levi_civita 
                    if(traceNr == 0) xi[1] *= -lc;
                    else xi[0] *= -lc;
                } else  {
                    // correct to positive with levi_civita
                    // correct the sign with negative levi_civita 
                    if(traceNr == 0) xi[1] *= lc;
                    else xi[0] *= lc;
                }
            }
        }
        
    };
}
