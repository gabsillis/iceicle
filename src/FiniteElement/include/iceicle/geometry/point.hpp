/**
 * @file Point.hpp
 * @author Gianni Absillis (gabsill@ncsu.edu)
 * @brief Definition of a Point
 * @version 0.1
 * @date 2022-01-11
 * 
 * @copyright Copyright (c) 2022
 * 
 */
#pragma once
#include <array>
#include <cstdarg>
#include <cmath>
#include <iceicle/math_utils.hpp

namespace GEOMETRY {
    template<typename T, int ndim>
    class Point{
        public:
        std::array<T, ndim> data;

        Point(){}

        Point(const T *data_args) : data{} {
            for(int idim = 0; idim < ndim; ++idim) data[idim] = data_args[idim];
        }

        template<typename ...ArgT>
        Point(ArgT... ts): data{static_cast<T>(ts)...} {}



        inline T &operator[] (int j){ return data[j]; }
        inline const T & operator[](int j) const { return data[j]; }

        inline operator const T *() const { return data.data(); }


    };

    /**
     * @brief 0d point for template continuity
     * 
     * @tparam T the floating point type
     */
    template<typename T>
    class Point<T, 0> {
        public:
        std::array<T, 1> data;

        Point() {data[0] = 0;}
        inline T &operator[](int j){return data[j];}
        inline const T & operator[](int j) const {return data[j];}
        inline operator const T *() const { return &data[0]; }
    };

    /**
     * @brief Get the distance between two points
     * 
     * @tparam T the floating point type
     * @tparam ndim the number of dimensions
     * @param a the first point
     * @param b the second point
     * @return T the euclidean distance between the points
     */
    template<typename T, int ndim>
    T distance(const Point<T, ndim> &a, const Point<T, ndim> &b){
        T dist = 0;
        for(int idim = 0; idim < ndim; idim++) dist += SQUARED(b[idim] - a[idim]);
        return std::sqrt(dist);
    }

    template<typename T, int ndim>
    T distancef(const T* a, const T*b){
        T dist = 0;
        for(int idim = 0; idim < ndim; idim++) dist += SQUARED(b[idim] - a[idim]);
        return std::sqrt(dist);
    }
}
