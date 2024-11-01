#include "iceicle/disc/projection.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/element/reference_element.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/fe_function/dglayout.hpp"
#include "iceicle/fe_function/el_layout.hpp"
#include "iceicle/fe_function/layout_right.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/mesh/mesh.hpp"
#include "iceicle/mesh/mesh_utils.hpp"
#include "iceicle/element_linear_solve.hpp"
#include "iceicle/tmp_utils.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/fe_utils.hpp>

#include <gtest/gtest.h>
#include <memory>

using namespace iceicle;

TEST(test_fespace, test_element_construction){
    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 2;
    static constexpr int pn_basis = 3;

    // create a uniform mesh
    AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {2, 2}, pn_geo);

    FESpace<T, IDX, ndim> fespace{
        &mesh, FESPACE_ENUMS::LAGRANGE,
        FESPACE_ENUMS::GAUSS_LEGENDRE, 
        tmp::compile_int<pn_basis>()
    };

    ASSERT_EQ(fespace.elements.size(), 4);

    ASSERT_EQ(fespace.dg_map.calculate_size_requirement(2), 4 * 2 * std::pow(pn_basis + 1, ndim));
}

class test_geo_el : public GeometricElement<double, int, 2>{
    static constexpr int ndim = 2;
    using T = double;
    using IDX = int;

    using Point = MATH::GEOMETRY::Point<T, ndim>;
    using HessianType = NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim, ndim>;

public:
    
    constexpr int n_nodes() const override {return 0;};

    constexpr DOMAIN_TYPE domain_type() const noexcept override {return DOMAIN_TYPE::DYNAMIC;};

    constexpr int geometry_order() const noexcept override {return 1;};

    const IDX *nodes() const override { return nullptr; }

    void transform(
        NodeArray<T, ndim> &node_coords,
        const Point &pt_ref,
        Point &pt_phys
    ) const override {
        T xi = pt_ref[0];
        T eta = pt_ref[1];

        pt_phys[0] = xi * eta;
        pt_phys[1] = xi + eta;
    }

    NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> Jacobian(
        NodeArray<T, ndim> &node_coords,
        const Point &xi_arg
    ) const override {
        T xi = xi_arg[0];
        T eta = xi_arg[1];
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<T, ndim, ndim> jac = {{
            {eta, xi},
            {1, 1}
        }};
        return jac;
    }

    HessianType Hessian(
        NodeArray<T, ndim> &node_coords,
        const Point &xi
    ) const override {
        HessianType hess;
        hess[0][0][0] = 0;
        hess[0][0][1] = 1;
        hess[0][1][0] = 1;
        hess[0][1][1] = 0;
        hess[1][0][0] = 0;
        hess[1][0][1] = 0;
        hess[1][1][0] = 0;
        hess[1][1][1] = 0;
        return hess;
    }
    auto regularize_interior_nodes(
        NodeArray<T, ndim>& coord /// [in/out] the node coordinates array 
    ) const -> void override{
        // do nothing: no interior nodes
    }

        /// @brief get the number of vertices in a face
        virtual 
        auto n_face_vert(
            int face_number /// [in] the face number
        ) const -> int override {return -1;}

        /// @brief get the vertex indices on the face
        /// NOTE: These vertices must be in the same order as if get_element_vert() 
        /// was called on the transformation corresponding to the face
        virtual 
        auto get_face_vert(
            int face_number,      /// [in] the face number
            index_type* vert_fac  /// [out] the indices of the vertices of the given face
        ) const -> void override {}

        auto n_face_nodes(int face_number) const -> int override { return 0; }

        /// @brief get the node indices on the face
        ///
        /// NOTE: Nodes are all the points defining geometry (vertices are endpoints)
        ///
        /// NOTE: These vertices must be in the same order as if get_nodes
        /// was called on the transformation corresponding to the face
        virtual 
        auto get_face_nodes(
            int face_number,      /// [in] the face number
            index_type* nodes_fac /// [out] the indices of the nodes of the given face
        ) const -> void override {
            
        }

        /// @brief get the face number of the given vertices 
        /// @return the face number of the face with the given vertices
        virtual 
        auto get_face_nr(
            index_type* vert_fac /// [in] the indices of the vertices of the given face
        ) const -> int override { return -1; };


        auto n_faces() const -> int override { return 0; }

        auto face_domain_type(int face_number) const -> DOMAIN_TYPE override { return DOMAIN_TYPE::N_DOMAIN_TYPES; }

        /// @brief clone this element
        auto clone() const -> std::unique_ptr<GeometricElement<double, int, 2>> override {
            return std::make_unique<test_geo_el>(*this);
        }

};

class test_basis : public Basis<double, 2> {
    static constexpr int ndim = 2;
    using T = double;
public:

    int nbasis() const override {return 1;}

    constexpr DOMAIN_TYPE domain_type() const noexcept override { return DOMAIN_TYPE::DYNAMIC;}

    void evalBasis(const T *xi_vec, T *Bi) const override {
        T xi = xi_vec[0];
        T eta = xi_vec[1];

        Bi[0] = eta * xi * xi + eta * eta * xi;
    }

    void evalGradBasis(const T *xi_vec, T *dBidxj) const override {
        T xi = xi_vec[0];
        T eta = xi_vec[1];
        dBidxj[0] = 2 * xi * eta + eta * eta;
        dBidxj[1] = xi * xi + 2 * xi * eta;

    }

    void evalHessBasis(const T*xi_vec, T *Hessian) const override {
        T xi = xi_vec[0];
        T eta = xi_vec[1];
       
        Hessian[0] = 2 * eta;
        Hessian[1] = 2 * xi + 2 * eta;
        Hessian[2] = Hessian[1];
        Hessian[3] = 2 * xi;
    };

    bool isOrthonormal() const override { return false; }

    bool isNodal() const override { return false; }

    inline int getPolynomialOrder() const override {return 2;}

};

TEST(test_fespace, test_dg_projection){

    using T = double;
    using IDX = int;

    static constexpr int ndim = 2;
    static constexpr int pn_geo = 1;
    static constexpr int pn_basis = 2;
    static constexpr int neq = 1;

    // create a uniform mesh
    int nx = 50;
    int ny = 10;
    AbstractMesh<T, IDX, ndim> mesh({-1.0, -1.0}, {1.0, 1.0}, {nx, ny}, pn_geo);
    auto fixed_nodes = flag_boundary_nodes(mesh);
    
    PERTURBATION_FUNCTIONS::random_perturb<T, ndim> perturb_fcn(-0.4 * 1.0 / std::max(nx, ny), 0.4*1.0/std::max(nx, ny));
    perturb_nodes(mesh, perturb_fcn, fixed_nodes);

//    // taylor vortex warped mesh
//    TODO: investigate hessian accuracy with this case 
//
//    int nx = 8;
//    int ny = 8;
//    AbstractMesh<T, IDX, ndim> mesh({0.0, 0.0}, {1.0, 1.0}, {nx, ny}, pn_geo);
//    std::function< void(std::span<T, ndim>, std::span<T, ndim>) > perturb_fcn;
//    perturb_fcn = PERTURBATION_FUNCTIONS::TaylorGreenVortex<T, ndim>{
//        .v0 = 0.5,
//        .xmin = { 0.0, 0.0 },
//        .xmax = { 1.0, 1.0 },
//        .L = 1
//    };
//    std::vector<bool> fixed_nodes = flag_boundary_nodes(mesh);
//    perturb_nodes(mesh, perturb_fcn, fixed_nodes);


    FESpace<T, IDX, ndim> fespace{
        &mesh, FESPACE_ENUMS::LAGRANGE,
        FESPACE_ENUMS::GAUSS_LEGENDRE, 
        tmp::compile_int<pn_basis>()
    };

    // define a pn_basis order polynomial function to project onto the space 
    auto projfunc = [](const double *xarr, double *out){
        double x = xarr[0];
        double y = xarr[1];
        out[0] = std::pow(x, pn_basis) + std::pow(y, pn_basis);
    };

    auto dprojfunc = [](const double *xarr) {
        double x = xarr[0];
        double y = xarr[1];
        NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim> deriv = { 
            pn_basis * std::pow(x, pn_basis - 1),
            pn_basis * std::pow(y, pn_basis - 1)
        };
        return deriv;
    };

    auto hessfunc = [](const double *xarr){
        double x = xarr[0];
        double y = xarr[1];
        int n = pn_basis;
        if (pn_basis < 2){
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim, ndim> hess = {{
                {0, 0},
                {0, 0}
            }};
            return hess;
        } else {
            NUMTOOL::TENSOR::FIXED_SIZE::Tensor<double, ndim, ndim> hess = {{
                {n * (n-1) *std::pow(x, n-2), 0.0},
                {0.0, n * (n-1) *std::pow(y, n-2)}
            }};
            return hess;
        }
    };


    // create the projection discretization
    Projection<double, int, ndim, neq> projection{projfunc};

    T *u = new T[fespace.ndof_dg() * neq](); // 0 initialized
    fe_layout_right felayout{fespace.dg_map, tmp::to_size<neq>{}};
    fespan u_span{u, felayout};

    // solve the projection 
    std::for_each(fespace.elements.begin(), fespace.elements.end(),
        [&](const FiniteElement<T, IDX, ndim> &el){
            compact_layout_right<IDX, 1> el_layout{el};
            T *u_local = new T[el_layout.size()](); // 0 initialized 
            dofspan<T, compact_layout_right<IDX, 1>> u_local_span(u_local, el_layout);

            T *res_local = new T[el_layout.size()](); // 0 initialized 
            dofspan<T, compact_layout_right<IDX, 1>> res_local_span(res_local, el_layout);
            
            // projection residual
            projection.domain_integral(el, res_local_span);

            // solve 
            solvers::ElementLinearSolver<T, IDX, ndim, neq> solver{el};
            solver.solve(u_local_span, res_local_span);

            // test a bunch of random locations
            for(int k = 0; k < 50; ++k){
                MATH::GEOMETRY::Point<T, ndim> ref_pt = random_domain_point(el.trans);
                MATH::GEOMETRY::Point<T, ndim> phys_pt = el.transform(ref_pt);
                
                // get the actual value of the function at the given point in the physical domain
                T act_val;
                projfunc(phys_pt, &act_val);

                T projected_val = 0;
                T *basis_vals = new double[el.nbasis()];
                el.eval_basis(ref_pt, basis_vals);
                u_local_span.contract_dofs(basis_vals, &projected_val);

                ASSERT_NEAR(projected_val, act_val, 1e-8);

                // test the derivatives
                std::vector<double> grad_basis_data(el.nbasis() * ndim);
                auto grad_basis = el.eval_phys_grad_basis(ref_pt, grad_basis_data.data());
                static_assert(grad_basis.rank() == 2);
                static_assert(grad_basis.extent(1) == ndim);

                // get the derivatives for each equation by contraction
                std::vector<double> grad_eq_data(neq * ndim, 0);
                auto grad_eq = u_local_span.contract_mdspan(grad_basis, grad_eq_data.data());

                auto dproj = dprojfunc(phys_pt);
                ASSERT_NEAR(dproj[0], (grad_eq[0, 0]), 1e-10);
                ASSERT_NEAR(dproj[1], (grad_eq[0, 1]), 1e-10);

                // test hessian
                std::vector<double> hess_basis_data(el.nbasis() * ndim * ndim);
                auto hess_basis = el.eval_phys_hess_basis(ref_pt, hess_basis_data.data());

                // get the hessian for each equation by contraction 
                std::vector<double> hess_eq_data(neq * ndim * ndim, 0);
                auto hess_eq = u_local_span.contract_mdspan(hess_basis, hess_eq_data.data());
                auto hess_proj = hessfunc(phys_pt);
                ASSERT_NEAR(hess_proj[0][0], (hess_eq[0, 0, 0]), 1e-8);
                ASSERT_NEAR(hess_proj[0][1], (hess_eq[0, 0, 1]), 1e-8);
                ASSERT_NEAR(hess_proj[1][0], (hess_eq[0, 1, 0]), 1e-8);
                ASSERT_NEAR(hess_proj[1][1], (hess_eq[0, 1, 1]), 1e-8);

                delete[] basis_vals;
            }

            delete[] u_local;
            delete[] res_local;
        }
    );

    delete[] u;
}
