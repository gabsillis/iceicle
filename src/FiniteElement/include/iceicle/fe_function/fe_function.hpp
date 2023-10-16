/**
 * @file fe_function.hpp
 * @brief contains classes that represent a function inside a finite element space
 *        these are containers for coefficients of the finite element basis functions
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once
#include <iceicle/element/finite_element.hpp>

namespace FE {
    

    /**
     * @brief a wrapper around a raw pointer of the floating point type
     *        allows the range represented by the pointer to be interpreted as a 
     *        set of coefficients for the basis functions and provides intuitive array access operations
     *
     *        This does not own the data pointer
     */
    template<typename T, int neq>
    class ElementData {
        
        private:

        T *data; // not owned
        int nbasis_;

        public:

        int nbasis() const {return nbasis_;}

        /**
         * @brief Construct a new Element Data object
         * 
         * @param nbasis the number of basis functions
         * @param data the pointer to the start of the data for this element
         */
        ElementData(int nbasis, T *data) : data(data), nbasis_(nbasis){}

        // delete the copy constructor
        ElementData(const ElementData<T, neq> &other) = delete;

        /**
         * @brief move constructor
         *
         * @param other the data to move
         */
        ElementData(ElementData<T, neq> &&other) : data(other.data), nbasis_(other.nbasis_)
        { other.data = nullptr; }

        /**
         * @brief Index into the element data
         * ibasis fastest
         * 
         * @param ieq the index of the equation
         * @param ibasis the index of the basis
         * @return T& the data at the given indices
         */
        inline T &getValue(int ieq, int ibasis) const {
            return data[ieq * nbasis_ + ibasis];
        }

        /**
         * @brief Index into the element data
         * 
         * @param ieq the index of the equation
         * @param ibasis the index of the basis
         * @return T& the data at the given indices
         */
        inline T &operator()(int ieq, int ibasis){
            return getValue(ieq, ibasis);
        }

        /** @brief set every coefficient in the range of this element to 0 */
        void zeroFill(){
            for(int i = 0; i < neq * nbasis_; i++) data[i] = 0;
        }

        /**
         * @brief add a multiple of another ElementData to this
         * this = this + alpha * other
         * @param alpha the multiplier
         * @param other the other coefficients
         */
        inline void addScaled(T alpha, ElementData<T, neq> &other){
            for(int eq = 0; eq < neq; eq++){
                for(int b = 0; b < nbasis_; b++){
                    getValue(eq, b) += alpha * other.getValue(eq, b);
                }
            }
        }

        /** @brief get the data pointer */
        inline T *getData() const{return data;}

        /** @brief get a pointer to the coefficients for each basis for the given equation index */
        T *getDataPtrForEq(int ieq) const{ return &data[ieq * nbasis_]; }

        /** @brief multiply each coefficient in the range of this element by alpha */
        void multiplyScalar(T alpha) {
            for(int i = 0; i < neq * nbasis_; i++) data[i] *= alpha;
        }

        /** @brief get the sum of squares of all the coefficients */
        inline T sumOfSquares() const {
            T ret = 0;
            for(int i = 0 ; i < neq * nbasis_; i++) ret += SQUARED(data[i]);
            return ret;
        }
    };
    
}
