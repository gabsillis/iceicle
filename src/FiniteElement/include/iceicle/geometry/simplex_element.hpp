/**
 * @file simplex_element.hpp
 * @brief GeometricElement implementation for simplices
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/transformations/SimplexElementTransformation.hpp>

namespace ELEMENT {
    
    template<typename T, typename IDX, int ndim, int Pn>
    class SimplexGeoElement final : public GeometricElement<T, IDX, ndim> {
        private:
        // namespace aliases
        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:
        /// @brief the transformation that properly converts to the reference domain for this element (must be inline to init)
        static inline TRANSFORMATIONS::SimplexElementTransformation<T, IDX, ndim, Pn> transform{};

        private:
        // ================
        // = Private Data =
        // =   Members    =
        // ================
        IDX _nodes[transform.n_nodes()];

        public:
        // ====================
        // = GeometricElement =
        // =  Implementation  =
        // ====================
        constexpr int n_nodes() override { return transform.n_nodes(); }

        IDX *nodes() override { return _nodes; }

        void Jacobian(
            std::vector< Point > &node_coords,
            const Point &xi,
            T J[ndim][ndim]
        ) const override {
            return transform.Jacobian(node_coords, _nodes, xi, J);
        }

        void Hessian(
            std::vector<Point> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const override {
            return transform.Hessian(node_coords, _nodes, xi, hess);
        }
    };
}
