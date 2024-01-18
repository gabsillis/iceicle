/**
 * @file finite_element.hpp
 * @brief A finite element which can get quadrature, evaluate basis functions, and perform transformations
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */

#pragma once

#include <Numtool/matrixT.hpp>
#include <Numtool/matrix/dense_matrix.hpp>
#include <iceicle/geometry/geo_element.hpp>
#include <iceicle/basis/basis.hpp>
#include <iceicle/quadrature/QuadratureRule.hpp>
#include <vector>

namespace ELEMENT {
    
    template<
        typename T,
        typename IDX,
        int ndim
    >
    class FEEvaluation {
        private:

        /** @brief the basis functions evaluated at each quadrature point */
        std::vector<std::vector<T>> data;

        public:
        /** @brief the basis functions */
        const BASIS::Basis<T, ndim> *basis; 

        /** @brief the quadraature rule */
        const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule;

        FEEvaluation() = default;

        FEEvaluation(
            BASIS::Basis<T, ndim> *basisptr,
            QUADRATURE::QuadratureRule<T, IDX, ndim> *quadruleptr
        ) : basis(basisptr), quadrule(quadruleptr)
        {
            // call eval basis for each quadrature point and prestore
            for(int igauss = 0; igauss < quadrule->npoints(); ++igauss){
                std::vector<T> &eval_vec = data[igauss];
                eval_vec.resize(basis->nbasis());
                basis->evalBasis(quadrule->getPoint(igauss).abscisse, eval_vec.data());
            }

            // TODO: gradient of basis (make a DenseMatrix class)

        }

        FEEvaluation(
            BASIS::Basis<T, ndim> &basis,
            QUADRATURE::QuadratureRule<T, IDX, ndim> &quadrule
        ) : FEEvaluation(&basis, &quadrule){}

        /* @brief get the evaluations of the basis functions at the igaussth quadrature pt */
        const std::vector<T> &operator[](int igauss) const { return data[igauss]; }
    };

    template<
        typename T,
        typename IDX,
        int ndim
    >
    class FiniteElement {

        using Point = MATH::GEOMETRY::Point<T, ndim>;

        public:

        /** @brief the geometric element (container for node indices and basic geometric operations) */
        const GeometricElement<T, IDX, ndim> *geo_el;

        /** @brief the basis functions */
        const BASIS::Basis<T, ndim> *basis;

        /** @brief the quadraature rule */
        const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule;

        /** @brief precomputed evaluations of the basis functions at the quadrature points */
        const FEEvaluation<T, IDX, ndim> *qp_evals;

        /** @brief the element index in the mesh */
        const IDX elidx;

        /**
         * @brief construct a FiniteElement
         * @param geo_el_arg the geometric element
         * @param basis_arg the basis function
         * @param quadrule_arg the qudarature rule to use
         * @param qp_evals evaluations of the basis functions at the quadrature points
         * @param elidx_arg the element index in the mesh
         */
        FiniteElement(
            const GeometricElement<T, IDX, ndim> *geo_el_arg,
            const BASIS::Basis<T, ndim> *basis_arg,
            const QUADRATURE::QuadratureRule<T, IDX, ndim> *quadrule_arg,
            const FEEvaluation<T, IDX, ndim> *qp_evals,
            IDX elidx_arg
        ) : geo_el(geo_el_arg), basis(basis_arg),
            quadrule(quadrule_arg), qp_evals(qp_evals), elidx(elidx_arg) {}

        /** 
         * @brief precompute any stored quantities
         * @param node_list the global node list which geo_el has indices into
         */
        void precompute(FE::NodalFEFunction<T, ndim> &node_list){}

        // =============================
        // = Basis Function Operations =
        // =============================

        /* @brief get the number of basis functions */
        int nbasis() const { return basis->nbasis(); }

        /**
         * @brief evaluate the values of the basis functions at the given point
         * cost: cost of evaluating all basis functions at a point
         * @param [in] xi the point in the reference domain [size = ndim]
         * @param [out] Bi the values of the basis functions at the point [size = nbasis]
         */
        void evalBasis(const T *xi, T *Bi) const {
            basis->evalBasis(xi, Bi);
        }

        /**
         * @brief get the values of the basis functions at the given quadrature point
         * cost: one memcpy of size nbasis
         * @param [in] quadrature_pt_idx the index of the quadrature point in [0, ngauss()]
         * @param [out] Bi the values of the basis functions at the point [size = nbasis]
         */
        void evalBasisQP(int quadrature_pt_idx, T *Bi) const {
            const std::vector<T> &eval = qp_evals[quadrature_pt_idx];
            std::copy(eval.begin(), eval.end(), Bi);
        }

        /**
         * @brief directly access the evaluation of the given basis function at the given quadrature point
         * cost: pointer dereference
         * @param [in] quadrature_pt_idx the index of the quadrature point in [0, ngauss()]
         * @param [in] ibasis the index of the basis function
         * @return the evaluation of the basis function at the given quadrature point
         */
        inline T basisQP(int quadrature_pt_idx, int ibasis) const {
            return qp_evals[quadrature_pt_idx][ibasis];
        }
       
        /**
         * @brief evaluate the first derivatives of the basis functions
         * 
         * @param [in] xi  the point in the reference domain [size = ndim]
         * @param [out] dBidxj the values of the first derivatives of the basis functions
         *                with respect to the reference domain at that point
         *                This is in the form of a 2d pointer array that must be preallocated
         *                row major order and contiguous
         *                \frac{dB_i}{d\xi_j} where i is ibasis
         *                takes a pointer to the first element of this data structure
         *                [size = [nbasis : i][ndim : j]] 
         */
        void evalGradBasis(const T *xi, T **dBidxj) const {
            return basis->evalGradBasis(xi, dBidxj);
        }
       
        /**
         * @brief evaluate the first derivatives of the basis functions at a quadrature point
         *
         * TODO: precompute evaluation to reduce cost to a memcpy
         * @param [in] quadrature_pt_idx the index of the quadrature point [0, ngauss()]
         * @param [out] dBidxj the values of the first derivatives of the basis functions
         *                with respect to the reference domain at that point
         *                This is in the form of a 2d pointer array that must be preallocated
         *                row major order and contiguous
         *                \frac{dB_i}{d\xi_j} where i is ibasis
         *                takes a pointer to the first element of this data structure
         *                [size = [nbasis : i][ndim : j]] 
         */
        void evalGradBasisQP(int quadrature_pt_idx, T **dBidxj) const {
            return basis->evalGradBasis(quadrule[quadrature_pt_idx].abscisse, dBidxj);
        }

        /**
         * @brief evaluate the first derivatives of the basis functions
         *        with respect to physical domain coordinates
         * @param [in] xi the point in the reference domain (uses Point class)
         * @param [in] transformation the transformation from the reference domain to the physical domain
         *                            (must be compatible with the geometric element)
         * @param [in] node_list the list of global node coordinates
         * @param [out] dBidxj the values of the first derivatives wrt the physical domain
         *                This is in the form of a 2d pointer array that must be preallocated
         *                row major order and contiguous
         *                \frac{dB_i}{dx_j} where i is ibasis
         *                takes a pointer to the first element of this data structure
         *                [size = [nbasis : i][ndim : j]] 
         */
        void evalPhysGradBasis(
            const Point &xi,
            FE::NodalFEFunction<T, ndim> &node_list,
            T **dBidxj
        ) const {
            //  fill with zero
            std::fill_n(dBidxj[0], nbasis() * ndim, 0.0);

            // get the Jacobian
            T J[ndim][ndim];
            geo_el->Jacobian(node_list, geo_el->nodes(), xi, J);

            // the inverse of J = adj(J) / det(J)
            T adjJ[ndim][ndim]; // note: this will be a symmetric matrix
            MATH::MATRIX_T::adjugate<ndim>(J[0], adjJ[0]);
            T detJ = MATH::MATRIX_T::determinant<ndim>(J[0]);

            // Evaluate dBi in reference domain
            using namespace MATH::MATRIX;
            DenseMatrixSetWidth<T, ndim> dBi(nbasis());
            evalGradBasis(xi, dBi);

            // dBidxj =  Jadj_{jk} * dBidxk
            for(int i = 0; i < nbasis(); ++i){
                for(int j = 0; j < ndim; ++j){
                    dBidxj[i][j] = 0.0;
                    for(int k = 0; k < ndim; ++k){
                        dBidxj[i][j] += dBi[i][k] * adjJ[j][k];
                    }
                }
            }
        }

        /**
         * @brief evaluate the first derivatives of the basis functions
         *        with respect to physical domain coordinates at the given quadrature point
         *
         * TODO: prestore
         * @param [in] quadrature_pt_idx the quadrature point index
         * @param [in] transformation the transformation from the reference domain to the physical domain
         *                            (must be compatible with the geometric element)
         * @param [in] node_list the list of global node coordinates
         * @param [out] dBidxj the values of the first derivatives wrt the physical domain
         *                This is in the form of a 2d pointer array that must be preallocated
         *                row major order and contiguous
         *                \frac{dB_i}{dx_j} where i is ibasis
         *                takes a pointer to the first element of this data structure
         *                [size = [nbasis : i][ndim : j]] 
         */
        void evalPhysGradBasisQP(
            int quadrature_pt_idx,
            FE::NodalFEFunction<T, ndim> &node_list,
            T **dBidxj
        ) const {
            return evalPhysGradBasis((*quadrule)[quadrature_pt_idx].abscisse, node_list, dBidxj);
        }

        // =========================
        // = Quadrature Operations =
        // =========================

        /** @brief get the number of quadrature points in the quadrature rule */
        int nQP() const { return quadrule->npoints(); }

        /** @brief get the "QuadraturePoint" (contains point and weight) at the given quadrature point index */
        const QUADRATURE::QuadraturePoint<T, ndim> getQP(int qp_idx) const { return (*quadrule)[qp_idx]; }

        // ========================
        // = Geometric Operations =
        // ========================

        
        /**
         * @brief transform from the reference domain to the physical domain
         * @param [in] node_coords the coordinates of all the nodes
         * @param [in] pt_ref the point in the refernce domain
         * @param [out] pt_phys the point in the physical domain
         */
        inline void transform(FE::NodalFEFunction<T, ndim> &node_coords, const Point &pt_ref, Point &pt_phys) const {
            return geo_el->transform(node_coords, pt_ref,  pt_phys);
        }
    };

}
