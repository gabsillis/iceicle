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
        static inline TRANSFORMATIONS::SimplexElementTransformation<T, IDX, ndim, Pn> transformation{};

        private:
        // ================
        // = Private Data =
        // =   Members    =
        // ================
        IDX _nodes[transformation.nnodes()];

        public:
        // ====================
        // = GeometricElement =
        // =  Implementation  =
        // ====================
        constexpr int n_nodes() const override { return transformation.nnodes(); }

        const IDX *nodes() const override { return _nodes; }

        void transform(FE::NodalFEFunction<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys)
        const override {
            return transformation.transform(node_coords, _nodes, pt_ref, pt_phys);
        }

        void Jacobian(
            FE::NodalFEFunction< T, ndim > &node_coords,
            const Point &xi,
            T J[ndim][ndim]
        ) const override {
            return transformation.Jacobian(node_coords, _nodes, xi, J);
        }

        void Hessian(
            FE::NodalFEFunction<T, ndim> &node_coords,
            const Point &xi,
            T hess[ndim][ndim][ndim]
        ) const override {
            return transformation.Hessian(node_coords, _nodes, xi, hess);
        }

        /** @brief set the node index at idx to value */
        void setNode(int idx, int value){_nodes[idx] = value; }
    };
}
