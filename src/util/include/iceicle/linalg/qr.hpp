
#include "linalg_utils.hpp"
#include "mdspan/mdspan.hpp"
#include <cmath>
namespace iceicle::linalg {
    
    /// @brief compute the Lp norm of the given vector 
    /// @param x the vector to compute the norm of
    template<int p = 2>
    inline constexpr 
    auto norm(in_vector auto x) -> decltype(x)::element_type 
    {
        using index_type = decltype(x)::index_type;
        using value_type = decltype(x)::value_type;
        value_type sum = 0;
        for(index_type i = 0; i < x.extent(0); ++i)
            sum += x[i] * x[i];
        if constexpr (p == 2) {
            return std::sqrt(sum);
        } else {
            return std::pow(sum, 1.0 / p);
        }
    }

    /// @brief compute the householder reflection of x
    /// https://www.cs.utexas.edu/~flame/laff/alaff/chapter03-real-HQR-factorization.html
    ///
    /// u = | 1   |
    ///     | u_2 |
    ///
    /// such that 
    ///
    ///  (I - 1/tau * u * u^T) * x = | +/- ||x|| |
    ///                              |    0      |
    ///
    /// @param [in] x the vector to get the householder reflection of 
    /// @param [out] u the householder reflection of x
    /// in place of u[0] we will store rho, which is the signed norm of x
    /// @return tau
    ///
    inline constexpr 
    auto householder_reflect(in_vector auto x, out_vector auto u) -> decltype(x)::value_type 
    {
        using index_type = decltype(x)::index_type;
        using value_type = decltype(x)::value_type;

        value_type X1 = x[0];

        // get the norm of everything but the first element
        value_type X2 = 0;
        for(index_type i = 1; i < x.extent(0); ++i) 
            X2 += x[i] * x[i];
        X2 = std::sqrt(X2);

        // get the norm of x as a whole 
        value_type alpha = std::sqrt(X1 * X1 + X2 * X2);

        // calculate the reflector
        value_type rho = -copysign(alpha, X1);
        u[0] = rho;
        value_type nu = X1 - rho;
        for(index_type i = 1; i < u.extent(0); ++i)
            u[i] = x[i] / nu;
        X2 /= std::abs(nu); // this is || u_2 ||
        return (1 + X2 * X2) / 2.0;
    }

    /// @brief Compute an in place householder QR factorization 
    /// @param [in/out] A is an m x n matrix which after decomposition
    /// has the R matrix in the upper triangle 
    /// and the householder reflectors (omitting the first element) in the lower triangle 
    /// The householder reflectors are in the form 
    /// u = | 1  |
    ///     | u2 |
    ///
    /// so that H_k = I - 1/tau_k * (u * u')
    /// The full size H_k is 
    /// H_k,full = I | 0 
    ///            --+----
    ///            0 | H_k
    /// @param [out] tau the multipliers for the householder reflectors as described above
    ///
    inline constexpr 
    auto householder_qr(inout_matrix auto A, out_vector auto tau) -> void
    {
        using index_type = decltype(A)::index_type;
        using value_type = decltype(A)::value_type;
        const auto m = A.extent(0);
        const auto n = A.extent(1);

        for(index_type k = 0; k < n; ++k) {

            //            A00  |  a01   A02
            //          -------+-----------
            // k --->    a10T  |  a11  a12T
            //           A20   |  a21  A22

            auto x = std::submdspan(A, std::pair{k, m}, k);
            std::vector<value_type> vdata(x.extent(0));
            std::mdspan v{vdata.data(), std::extents{x.extent(0)}};
            tau[k] = householder_reflect(x, v);


            // fill in the householder vector in the below diagonal
            // and fill diagonal entry
            A[k, k] = v[0];
            for(index_type i = k + 1; i < m; ++i){
                A[i, k] = v[i - k];
            }

            std::vector<value_type> wdata(n - k - 1);
            std::mdspan w{wdata.data(), std::extents{n - k - 1}};
            auto a12 = std::submdspan(A, k, std::pair{k+1,n});

            // compute w = (a12' + u' * A22)' / tau
            copy(a12, w);
            for(index_type j = k + 1; j < n; ++j){
                auto u2 = std::submdspan(v, std::pair{1, v.extent(0)});
                auto A22j = std::submdspan(A, std::pair{k+1, m}, j);
                wdata[j - (k+1)] += dot(u2, A22j);
            }
            for(index_type i = 0; i < w.extent(0); ++i) 
                w[i] /= tau[k];

            // a12 := a12 - w
            axpy(-1, w, a12);

            // A22 := A22 - u * w'
            auto A22 = std::submdspan(A, std::pair{k+1, m}, std::pair{k+1, n});
            for(index_type i = 0; i < A22.extent(0); ++i){
                for(index_type j = 0; j < A22.extent(1); ++j){
                    A22[i, j] -= v[i + 1] * w[j];
                }
            }
        }
    }

    /// @brief form the reduced Q matrix (size m x n) in place 
    /// @param [in/out] Q Given a matrix decomposed by householder_qr(A, tau)
    ///                 Transforms this matrix in place to the Q matrix of the 
    ///                 QR factorization 
    /// @param [in] tau the multiplers for the householder reflectors from householder_qr
    inline constexpr
    auto form_Q(inout_matrix auto Q, in_vector auto tau) -> void 
    {
        using index_type = decltype(Q)::index_type;
        using value_type = decltype(Q)::value_type;
        const auto m = Q.extent(0);
        const auto n = Q.extent(1);
        for(index_type k = n-1; k != (index_type) -1; --k) {
            auto a21 = std::submdspan(Q, std::pair{k+1, m}, k);
            auto a12 = std::submdspan(Q, k, std::pair{k+1, n});
            auto Q22 = std::submdspan(Q, std::pair{k+1, m}, std::pair{k+1, n});

            Q[k, k] = 1 - 1 / tau[k];
            for(index_type j = 0; j < Q22.extent(1); ++j){
                auto Q22col = std::submdspan(Q22, std::full_extent, j);
                a12[j] = -dot(a21, Q22col) / tau[k];
            }
            for(index_type i = 0; i < Q22.extent(0); ++i){
                for(index_type j = 0; j < Q22.extent(1); ++j){
                    Q22[i, j] += a12[j] * a21[i];
                }
            }
            for(index_type i = 0; i < a21.extent(0); ++i)
                a21[i] /= -tau[k];
        }
    }

    // @brief form the reduced Q matrix (size m x n)
    // @param [in] A the matrix decomposed by householder_qr 
    // @param [in] tau the multiplers for the householder reflectors from householder_qr 
    // @param [out] Q (size m x n) the orthogonal matrix Q of the QR decomposition
    inline constexpr
    auto form_Q(in_matrix auto A, in_vector auto tau, out_matrix auto Q) -> void 
    {
        using index_type = decltype(A)::index_type;
        using value_type = decltype(A)::value_type;
        const auto m = A.extent(0);
        const auto n = A.extent(1);

        for(index_type i = 0; i < m; ++i)
            for(index_type j = 0; j < n; ++j)
                Q[i, j] = A[i, j];
        form_Q(Q, tau);
    }

    // @brief apply the transpose (and also inverse) of the Q matrix to a vector 
    // @param [in] A the matrix decomposed by householder_qr 
    // @param [in] tau the multiplers for the householder reflectors from householder_qr 
    // @param [in] x the vector (size m) the multiply 
    // @param [out] y the vector (size m) that is the result of Q^T * x
    inline constexpr 
    auto apply_QT(in_matrix auto A, in_vector auto tau, in_vector auto x, out_vector auto y)
    {
        using index_type = decltype(A)::index_type;
        using value_type = decltype(A)::value_type;
        const auto m = A.extent(0);
        const auto n = A.extent(1);

        copy(x, y);
        for(index_type k = 0; k < n; ++k){
            auto a21 = std::submdspan(A, std::pair{k+1, m}, k);
            value_type phi = y[k];
            auto y2 = std::submdspan(y, std::pair{k+1, m});
            value_type w = (phi + dot(a21, y2)) / tau[k];
            y[k] -= w;
            axpy(-w, a21, y2);
        }
    }
    

}
