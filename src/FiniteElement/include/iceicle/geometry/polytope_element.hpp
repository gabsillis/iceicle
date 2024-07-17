#include "iceicle/basis/lagrange_1d.hpp"
#include "iceicle/transformations/polytope_transformations.hpp"
#include "geo_element.hpp"
namespace iceicle {
   

    template<polytope::geo_code auto t, class T, class IDX, std::size_t geo_order>
    class PolytopeElement : public GeometricElement<T, IDX, get_ndim(t)> {

        static constexpr int ndim = get_ndim(t);
        static constexpr std::size_t _n_nodes = polytope::get_n_node(t, polytope::full_extrusion<ndim>, geo_order);
        std::array<IDX, _n_nodes> node_indices;

        static UniformLagrangeInterpolation<T, geo_order> interpolation{};
        // type aliases
        using Point = MATH::GEOMETRY::Point<T, ndim>;
        using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;
        using JacobianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim>;

        // =============================
        // = Coordinate Transformation =
        // =============================
        auto transform(
            NodeArray<T, ndim> &node_coords,
            const Point &pt_ref,
            Point &pt_phys
        ) const -> void override
        {
            std::array<T, n_nodes()> shp;
            polytope::fill_shp<T, t>(interpolation, pt_ref, shp.data());
            std::ranges::fill(pt_phys, 0.0);
            for(int inode = 0; inode < n_nodes(); ++inode){
                IDX global_inode = node_indices[inode];
                for(int idim = 0; idim < ndim; ++idim)
                    pt_phys += node_coords[global_inode][idim] * shp[inode];
            }
        }

        auto Jacobian(
            NodeArray<T, ndim> &node_coords,
            const Point &xi
        ) const -> JacobianType override 
        {
            std::extents dshp_extents{n_nodes(), ndim};
            constexpr std::size_t dshp_size = std::max(1, n_nodes() * ndim);
            std::array<T, dshp_size> dshp_data;
            std::mdspan dshp{dshp_data, dshp_extents};
            polytope::fill_deriv<T, t>(interpolation, xi, dshp);

            JacobianType J;
            J = 0;

            for(int inode = 0; inode < n_nodes(); ++inode){
                IDX global_inode = node_indices[inode];
                const auto &node = node_coords[global_inode];
                for(int idim = 0; idim < ndim; ++idim){
                    for(int jdim = 0; jdim < ndim; ++jdim){
                        // add contribution to jacobian from this basis function
                        J[idim][jdim] += dshp[inode, jdim] * node[idim];
                    }
                }
            }
        }

        auto Hessian(
            const NodeArray<T, ndim> &node_coords,
            const IDX *node_indices,
            const Point &xi
        ) const noexcept -> HessianType override 
        {
            std::extents d2shp_extents{n_nodes(), ndim};
            constexpr std::size_t d2shp_size = std::max(1, n_nodes() * ndim);
            std::array<T, d2shp_size> d2shp_data;
            std::mdspan d2shp{d2shp_data, d2shp_extents};
            polytope::fill_hess<T, t>(interpolation, xi, d2shp);

            HessianType hess{};
            hess = 0;

            for(int inode = 0; inode < n_nodes(); ++inode){
                // get view to the node coordinates from the node coordinate array
                IDX global_inode = node_indices[inode];
                const auto & node = node_coords[global_inode];

                for(int kdim = 0; kdim < ndim; ++kdim){ // k corresponds to xi 
                    for(int idim = 0; idim < ndim; ++idim){
                        for(int jdim = idim; jdim < ndim; ++jdim){
                            hess[kdim][idim][jdim] += d2shp[inode, idim, jdim] * node[kdim];
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

        auto nodes() const -> const IDX* override 
        {
            return node_indices.data();
        }

        constexpr 
        auto n_nodes() const -> int override
        { return _n_nodes; } 

    };
}
