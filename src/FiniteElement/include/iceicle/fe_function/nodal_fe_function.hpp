
#pragma once
#include <initializer_list>
#include <vector>
#include <random>
#include <stdexcept>
#include <Numtool/point.hpp>
#include <algorithm>

namespace FE {
    /** @brief view into a NodalFEFunction at a given node index */
    template<typename T, int ndim>
    struct NodeData{
        T *_ptr;

        T &operator[](int j){return _ptr[j];}

        const T &operator[](int j) const {return _ptr[j];}

        /** @brief set the data to the values in a point */
        NodeData<T, ndim> &operator=(const MATH::GEOMETRY::Point<T, ndim> &pt){
            std::copy_n((const T *) pt, ndim, _ptr);
            return *this;
        }

        operator T *() { return _ptr; }
    };
   
    /**
     * @brief function on the finite element space using nodal basis functions
     * essentially a container for nodal coefficients
     * @tparam T the floating point type
     * @tparam ndim the number of dimensions
     */
    template<typename T, int ndim>
    class NodalFEFunction{
        private:
            std::size_t _ndof;
        T *_data;
        public:

        /** @brief construct an empty NodalFEFunction */
        NodalFEFunction() : _ndof(0), _data(nullptr) {}

        /** @brief construct uninitialized nodal fe function with nnodes*/
        NodalFEFunction(std::size_t nnode)
        : _ndof(nnode * ndim), _data(new T[_ndof]) {}

        /** @brief construct from a 2d initializer list */
        NodalFEFunction(std::initializer_list<std::initializer_list<T>> values)
        : _ndof(values.size() * ndim), _data(new T[_ndof]) {
           
            // fill with the values of the initializer list
            T *follower = _data;
            for(auto it1 = values.begin(); it1 != values.end(); ++it1){
                
                if(it1->size() != ndim) throw std::out_of_range("Initializer list not correct shape");
                for(T val : *it1){
                    *follower = val;
                    follower++;
                }
            }
        }
   
        /** @brief copy constructor */
        NodalFEFunction(const NodalFEFunction<T, ndim> &other)
        : _ndof(other._ndof), _data(new T[_ndof]) {
            std::copy_n(other._data, _ndof, _data);
        }

        /** @brief move constructor */
        NodalFEFunction(NodalFEFunction<T, ndim> &&other)
        : _ndof(other.ndof), _data(other._data) {
            other._data = nullptr;
        }

        /** @brief copy assignment operator */
        NodalFEFunction<T, ndim>& operator=(const NodalFEFunction<T, ndim> &other){
            if(_data != nullptr) delete[] _data;
            _ndof = other._ndof;
            _data = new T[_ndof];
            std::copy_n(other._data, _ndof, _data);
            return *this;
        }

        /** @brief move assignment operator */
        NodalFEFunction<T, ndim>& operator=(NodalFEFunction<T, ndim> &&other){
            if(_data != nullptr) delete[] _data;
            _ndof = other._ndof;
            _data = other._data;
            other._data = nullptr;
            return *this;
        }

        /** @brief Destructor */
        ~NodalFEFunction(){ if(_data != nullptr) delete[] _data; }

        /** @brief fill with val */
        NodalFEFunction<T, ndim>& operator=(T val){
            std::fill_n(_data, _ndof, val);
        }

        /** @brief get the data for the ith node */
        NodeData<T, ndim> operator[](std::size_t inode){ return NodeData<T, ndim>{_data + inode * ndim}; }

        /** @brief get the data for the ith node */
        const NodeData<T, ndim> operator[](std::size_t inode) const { return NodeData<T, ndim>{_data + inode * ndim}; }

        /** @brief get the number of nodes */
        std::size_t n_nodes() const { return _ndof / ndim; }

        /** @brief resize and copy over the data up to the min of new and old capacity
         * @param new_nnodes the new number of nodes */
        void resize(std::size_t new_nnodes){
            std::size_t new_ndof = ndim * new_nnodes;
            T *newdata = new T[new_ndof];
            std::copy_n(_data, std::min(_ndof, new_ndof), newdata);
            if(_data != nullptr) delete[] _data;
            _ndof = new_ndof;
            _data = newdata;
        }

        /**
         * @brief randomly peturb all the nodes 
         * @param min_peturb the minimum peturbation
         * @param max_peturb the maximum peturbation 
         */
        void random_perturb(T min_peturb, T max_peturb){
            using namespace std;

            random_device rdev{};
            default_random_engine engine{rdev()};
            uniform_real_distribution<T> dist{min_peturb, max_peturb};

            for(int idof = 0; idof < _ndof; ++idof){
                _data[idof] += dist(engine);
            }
        }
    };
}
