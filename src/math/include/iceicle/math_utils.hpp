#pragma once
#define SQUARED(X) ((X) * (X))

namespace MATH {

    /**
     * @brief Constant expression ceiling function
     * 
     * @param arg the floating point number
     * @return constexpr int the ceiling of that floating point number
     */
    constexpr int ceil(double arg) {
        return (static_cast<float>(static_cast<int>(arg)) == arg || arg < 0)
            ? static_cast<int>(arg)
            : static_cast<int>(arg) + 1;
    }

    /**
     * @brief TMP version of the kroneker delta
     * 
     * @tparam i value of the first index
     * @tparam j value of the second index
     * @return constexpr int 1 if i == j, 0 otherwise
     */
    template<int i, int j>
    constexpr int kronecker(){
        if constexpr(i == j){
            return 1;
        } else {
            return 0;
        }
    }

    /**
     * @brief Take an integer to the power of another iteger
     * ::value stores the result a^b
     * @tparam a the base
     * @tparam b the exponent
     */
    template<int a, int b>
    struct power_T{
        static_assert(b < 100); // protect recursion depth
        static_assert(b > -1);
        static const int value = a * power_T<a, b-1>::value;
    };

    template<int a>
    struct power_T<a, 0>{
        static const int value = 1;
    };

    /**
     * @brief Get the max of two integers
     * 
     * @tparam a the first integer
     * @tparam b the second integer
     * @return constexpr int the max of a and b
     */
    template<int a, int b>
    constexpr int max_i_T(){
        if constexpr ( a > b ) return a;
        else return b;
    }

    /**
     * @brief Get the min of two integers
     * 
     * @tparam a the first integer
     * @tparam b the second integer
     * @return constexpr int the min of a and b
     */
    template<int a, int b>
    constexpr int min_i_T(){
        if constexpr ( a > b ) return b;
        else return a;
    }

    /**
     * @brief Divide two integers but round up
     * 
     * @tparam dividend the dividend
     * @tparam divisor the divisor
     * @return constexpr int ceil(dividend / divisor)f
     */
    template<int dividend, int divisor>
    constexpr int ceildivide(){
        if constexpr (dividend % divisor > 0) return dividend / divisor + 1;
        else return dividend / divisor;
    }


    /**
     * @brief Get a constexpr factorial of an integer
     * 
     * @tparam n the integer to get the factorial of
     * @return constexpr int the factorial
     */
    template<int n>
    constexpr int factorial(){
        if constexpr (n > 0){
            return n * factorial<n-1>();
        } else {
            return 1;
        }
    }

    /**
     * @brief Get the consecutive sum of integers between the first and last integer specified
     * if both are the same return just the first
     * @tparam first the first integer
     * @tparam last the last integer
     * @return constexpr int the consecutive sum of all integers in [first, last] (inclusive)
     */
    template<int first, int last>
    constexpr int consecutiveSum(){
        if constexpr(first == last){ return first; }
        constexpr int n = last - first + 1;
        return (n * (first + last)) / 2;
    }

    /**
     * @brief Get the binomial coefficient
     * 
     * | n |
     * | k |
     * 
     * @tparam n the size of the selection pool
     * @tparam k the number to chose
     * @return constexpr int n choose k
     */
    template<int n, int k>
    constexpr int binomial(){
        if constexpr(k < n && k > 0)
            return binomial<n-1, k-1>() + binomial<n - 1, k>();
        else if constexpr(k == n)
            return 1;
        else if constexpr(k == 0)
            return 1;
        else
            return 0;
    }

    /**
     * @brief Templated power method
     * 
     * @tparam T The type of the base
     * @tparam n the power exponent
     * @param x the base
     * @return T x ^ n
     */
    template<typename T, int n>
    T pow_T(T x){
        T prod = 1;
        for(int i = 0; i < n; i++){
            prod *= x;
        }
        return prod;
    }
    namespace MATRIX_T {
        /**
        * @brief Gets the adjugate of A for 1x1, 2x2, or 3x3 matrix
        * 
        * @tparam matsize the matrix size
        * @tparam T the floating point type
        * @param A the matrix
        * @param adj [out] the adjugate (must be presized)
        */
        template<std::size_t matsize, typename T>
        void adjugate(const T *Ain, T *adjin){
            T (* A)[matsize] = (T (*)[matsize])Ain;
            T (* adj)[matsize] = (T (*)[matsize])adjin;
            if constexpr (matsize == 1){
                adj[0][0] = 1;
            } else if constexpr(matsize == 2){
                adj[0][0] = A[1][1];
                adj[0][1] = -A[0][1];

                adj[1][0] = -A[1][0];
                adj[1][1] = A[0][0];
            } else {
                adj[0][0] = A[1][1] * A[2][2] - A[1][2] * A[2][1];
                adj[0][1] = A[0][2] * A[2][1] - A[0][1] * A[2][2];
                adj[0][2] = A[0][1] * A[1][2] - A[1][1] * A[0][2];

                adj[1][0] = A[1][2] * A[2][0] - A[1][0] * A[2][2];
                adj[1][1] = A[0][0] * A[2][2] - A[0][2] * A[2][0];
                adj[1][0] = A[0][1] * A[2][0] - A[0][0] * A[2][1];

                adj[2][0] = A[0][1] * A[1][2] - A[0][2] * A[1][1];
                adj[2][1] = A[0][1] * A[2][0] - A[0][0] * A[2][1];
                adj[2][2] = A[0][0] * A[1][1] - A[0][1] * A[1][0];
            }
        }

        template<std::size_t matsize>
        double determinant(double *Ain){
            double (* A)[matsize] = (double (*)[matsize])Ain;
            if constexpr (matsize == 1){
                return A[0][0];
            } else if constexpr(matsize == 2){
                return A[0][0] * A[1][1] - A[1][0] * A[0][1];
            } else if constexpr(matsize == 3){
                return A[0][0] * A[1][1] * A[2][2] +//aei
                        A[0][1] * A[1][2] * A[2][0] +//bfg
                        A[0][2] * A[1][0] * A[2][1] -//cdh
                        A[0][2] * A[1][1] * A[2][0] -//ceg
                        A[0][1] * A[1][0] * A[2][2] -//bdi
                        A[0][0] * A[1][2] * A[2][1]; //afh
            } else {
                return std::nan("0"); //
            }
        }

        template<std::size_t matsize, typename T>
        T determinant(T *Ain){
        T (* A)[matsize] = (T (*)[matsize])Ain;
            if constexpr (matsize == 1){
                return A[0][0];
            } else if constexpr(matsize == 2){
                return A[0][0] * A[1][1] - A[1][0] * A[0][1];
            } else if constexpr(matsize == 3){
                return A[0][0] * A[1][1] * A[2][2] +//aei
                        A[0][1] * A[1][2] * A[2][0] +//bfg
                        A[0][2] * A[1][0] * A[2][1] -//cdh
                        A[0][2] * A[1][1] * A[2][0] -//ceg
                        A[0][1] * A[1][0] * A[2][2] -//bdi
                        A[0][0] * A[1][2] * A[2][1]; //afh
            } else {
                return std::nan("0"); //
            }
        }
    }
}