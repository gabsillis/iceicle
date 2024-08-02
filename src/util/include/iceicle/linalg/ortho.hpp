/// @brief utilities related to orthogonality
/// i.e Givens rotations
///
/// @author Gianni Absillis (gabsill@ncsu.edu)

#include <cmath>

namespace iceicle::linalg {

    /**
     * @brief Compute a Givens rotation for a vector [ f ; g ]
     * 
     * A givens rotation is the matrix:
     * G = 
     * |  c  s |
     * | -s  c |
     * 
     * That rotates the vector s.t G * [f ; g] = [r; 0]
     * 
     * This is an orthogonal rotation
     * @tparam T the floating point type
     * @param f the first component of the vector
     * @param g the second component of the vector
     * @param c [out] the cosine of the rotation (c in the matrix)
     * @param s [out] the sine of the rotation (s in the matrix)
     * @param r [out] the r which is the top componenet of the resulting vector
     * 
     * Continuous real plane rotation based on Edward Anderson
     * http://www.netlib.org/lapack/lawnspdf/lawn150.pdf
     */
    template<typename T>
    void givens(T f, T g, T *c, T *s, T *r) {
        if(g == 0) {
            *c = copysign(1, f);
            *s = 0;
            *r = abs(f);
        } else if (f == 0){
            *c = 0;
            *s = copysign(1, g);
            *r = abs(g);
        } else if (abs(f) > abs(g)) {
            T t = g / f;
            T u = copysign(sqrt(1 + t * t), f);
            *c = 1 / u;
            *s = t * *c;
            *r = f * u;
        } else {
            T t = f / g;
            T u = copysign(sqrt(1 + t * t), g);
            *s = 1/ u;
            *c = t * *s;
            *r = g * u;
        }
    }

}
