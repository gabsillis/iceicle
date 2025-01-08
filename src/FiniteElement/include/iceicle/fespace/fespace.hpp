/**
 * @brief A finite element space represents a collection of finite elements 
 * and trace spaces (if applicable)
 * that provides a general interface to a finite element discretization of the domain
 * and simple generation utilities
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once

#include "iceicle/basis/basis.hpp"
#include "iceicle/crs.hpp"
#include "iceicle/element/finite_element.hpp"
#include <iceicle/element/reference_element.hpp>
#include "iceicle/fe_definitions.hpp"
#include <iceicle/basis/dof_mapping.hpp>
#include "iceicle/geometry/face.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include "iceicle/iceicle_mpi_utils.hpp"
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/tmp_utils.hpp>
#include <Numtool/tmp_flow_control.hpp>
#include <map>
#include <type_traits>

#ifdef ICEICLE_USE_MPI 
#include <mpi.h>
#endif

namespace iceicle {
    /**
     * Key to define the surjective mapping from an element 
     * to the corresponding evaluation
     */
    struct FETypeKey {
        DOMAIN_TYPE domain_type;

        int basis_order;

        int geometry_order;

        FESPACE_ENUMS::FESPACE_QUADRATURE qtype;

        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype;


        friend bool operator<(const FETypeKey &l, const FETypeKey &r){
            using namespace FESPACE_ENUMS;
            if(l.qtype != r.qtype){
                return (int) l.qtype < (int) r.qtype;
            } else if(l.btype != r.btype){
                return (int) l.btype < (int) r.btype;
            } else if(l.domain_type != r.domain_type) {
                return (int) l.domain_type < (int) r.domain_type;
            } else if (l.geometry_order != r.geometry_order){
                return l.geometry_order < r.geometry_order;
            } else if( l.basis_order != r.basis_order) {
                return l.basis_order < r.basis_order;
            } else {
                // they are equal so less than is false for both cases
                return false;
            }
        }
    };

    /**
     * key to define surjective mapping from trace space to 
     * corresponding evaluation 
     */
    struct TraceTypeKey {

        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype_l;

        FESPACE_ENUMS::FESPACE_BASIS_TYPE btype_r;

        int basis_order_l;

        int basis_order_r;

        int basis_order_trace;

        int geometry_order;

        DOMAIN_TYPE domain_type;

        FESPACE_ENUMS::FESPACE_QUADRATURE qtype;

        unsigned int face_info_l;

        unsigned int face_info_r;

        auto operator <=>(const TraceTypeKey&) const = default;
    };

    /**
     * @brief Collection of FiniteElements and TraceSpaces
     * to form a unified interface for a discretization of a domain 
     *
     * @tparam T the numeric type 
     * @tparam IDX the index type 
     * @tparam ndim the number of dimensions
     * @tparam conformity the conformity of degrees of freedom between elements 
     *         index corresponds to position in exact sequence
     */
    template<typename T, typename IDX, int ndim, int conformity = l2_conformity(ndim)>
    class FESpace {
    public:

        using ElementType = FiniteElement<T, IDX, ndim>;
        using TraceType = TraceSpace<T, IDX, ndim>;
        using GeoElementType = GeometricElement<T, IDX, ndim>;
        using GeoFaceType = Face<T, IDX, ndim>;
        using MeshType = AbstractMesh<T, IDX, ndim>;
        using BasisType = Basis<T, ndim>;

        /// @brief get the conformity of the degrees of freedom between elements 
        /// The index corresponds to the position in the exact sequence
        static constexpr int conformity_class()
        { return conformity; } 

        /// @brief pointer to the mesh used
        MeshType *meshptr;

    private:

        // ========================================
        // = Maps to Basis, Quadrature, and Evals =
        // ========================================

        using ReferenceElementType = ReferenceElement<T, IDX, ndim>;
        using ReferenceTraceType = ReferenceTraceSpace<T, IDX, ndim>;
        std::map<FETypeKey, ReferenceElementType> ref_el_map;
        std::map<TraceTypeKey, ReferenceTraceType> ref_trace_map;

    public:
        /// @brief Array of finite elements in the space including 
        /// ones owned by neighboring processes 
        /// NOTE: these are needed for computing integrals on interprocess faces
        std::vector<ElementType> all_elements;

        /// @brief view over the finite elements in the space elements
        /// owned by this process
        std::span<ElementType> elements;

        /// @brief Array of trace spaces in the space 
        std::vector<TraceType> traces;

        /// @brief the start index of the interior traces 
        std::size_t interior_trace_start;
        /// @brief the end index of the interior traces (exclusive) 
        std::size_t interior_trace_end;

        /// @brief the start index of the boundary traces 
        std::size_t bdy_trace_start;
        /// @brief the end index of the boundary traces (exclusive)
        std::size_t bdy_trace_end;

        /** @brief maps local dofs to global dofs */
        dof_map<IDX, ndim, conformity> dofs;

        /// @brief the parallel index partitioning of the dofs
        pindex_map<IDX> dof_partitioning;

        /** @brief the mapping of faces connected to each node */
        util::crs<IDX> fac_surr_nodes;

        /** @brief the mapping of elements connected to each node */
        util::crs<IDX, IDX> el_surr_nodes;


        // default constructor
        FESpace() = default;

        // delete copy semantics
        FESpace(const FESpace &other) = delete;
        FESpace<T, IDX, ndim, conformity>& operator=(const FESpace &other) = delete;

        // keep move semantics
        FESpace(FESpace &&other) = default;
        FESpace<T, IDX, ndim, conformity>& operator=(FESpace &&other) = default;

    private:
        
        template<int basis_order>
        [[nodiscard]] inline constexpr 
        auto generate_uniform_order_elements(
            FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
            FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            tmp::compile_int<basis_order> basis_order_arg
        ) -> std::vector<ElementType>
        {
            std::vector<ElementType> elements;
            // Generate the Finite Elements
            elements.reserve(meshptr->nelem());
            for(ElementTransformation<T, IDX, ndim>* geo_trans : meshptr->el_transformations){
                // create the Element Domain type key
                FETypeKey fe_key = {
                    .domain_type = geo_trans->domain_type,
                    .basis_order = basis_order,
                    .geometry_order = geo_trans->order,
                    .qtype = quadrature_type,
                    .btype = basis_type
                };

                // check if an evaluation doesn't exist yet
                if(ref_el_map.find(fe_key) == ref_el_map.end()){
                    ref_el_map[fe_key] = ReferenceElementType(geo_trans->domain_type, geo_trans->order, basis_type, quadrature_type, basis_order_arg);
                }
                ReferenceElementType &ref_el = ref_el_map[fe_key];

                // this will be the index of the new element
                IDX ielem = elements.size();

                // create the finite element
                ElementType fe{
                    .trans = geo_trans, 
                    .basis = ref_el.basis.get(),
                    .quadrule = ref_el.quadrule.get(),
                    .qp_evals = std::span<const BasisEvaluation<T, ndim>>{ref_el.evals},
                    .inodes = meshptr->conn_el.rowspan(ielem), // NOTE: meshptr cannot invalidate anymore
                    .coord_el = meshptr->coord_els.rowspan(ielem),
                    .elidx = ielem
                };

                // add to the elements list
                elements.push_back(fe);
            }
            return elements;
        }

    public:

        /**
         * @brief construct an FESpace with uniform 
         * quadrature rules, and basis functions over all elements 
         *
         * @tparam basis_order the polynomial order of 1D basis functions
         *
         * @param meshptr pointer to the mesh 
         * @param basis_type enumeration of what basis to use 
         * @param quadrature_type enumeration of what quadrature rule to use 
         * @param basis_order_arg for template argument deduction of the basis order
         * @param serial_tag set to true to build an fespace only on this process
         *        and ignore communication
         */
        template<int basis_order>
        FESpace(
            MeshType *meshptr,
            FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
            FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
            tmp::compile_int<basis_order> basis_order_arg,
            bool serial_tag = false
        ) requires( conformity == l2_conformity(ndim) ) 
        // the only case we currently have general mappings for
        : meshptr(meshptr), ref_el_map{}, ref_trace_map{}, 
          all_elements{generate_uniform_order_elements(basis_type, quadrature_type, basis_order_arg)},
          elements{all_elements.begin(), all_elements.begin() 
              + meshptr->element_partitioning.owned_range_size(mpi::mpi_world_rank())},
          dofs{elements}
        {
            // create a partitioning for the dofs
            if (serial_tag){
                dof_partitioning = pindex_map<IDX>::create_serial(elements.size());
            } else {
                IDX my_ndof = dofs.size();
                std::vector<IDX> offsets{0};
                std::vector<IDX> p_indices(my_ndof);
                std::unordered_map< IDX, IDX > inv_p_indices{};
                for(int irank = 0; irank < mpi::mpi_world_size(); ++irank){
                    IDX ndof = my_ndof;
#ifdef ICEICLE_USE_MPI
                    MPI_Bcast(&ndof, 1, mpi_get_type(ndof), irank, MPI_COMM_WORLD);
                    offsets.push_back(offsets[irank] + ndof);
#endif
                    if(irank == mpi::mpi_world_rank()){
                        std::iota(p_indices.begin(), p_indices.end(), offsets[irank]);
                        for(IDX lidx = 0; lidx < p_indices.size(); ++lidx){
                            inv_p_indices[p_indices[lidx]] = lidx;
                        }
                    }
                }
                dof_partitioning = pindex_map{p_indices, inv_p_indices, offsets};
            }

            // Generate the Trace Spaces
            traces.reserve(meshptr->faces.size());
            for(const auto& fac : meshptr->faces){
                // NOTE: assuming element indexing is the same as the mesh still

                // parallel bdy faces are essentially also interior faces 
                // aside from being a bit *special* :3
                bool is_interior = 
                    fac->bctype == BOUNDARY_CONDITIONS::INTERIOR 
                    or fac->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM;
                ElementType *elptrL = &elements[fac->elemL];
                ElementType *elptrR = (is_interior) ? &elements[fac->elemR] : &elements[fac->elemL];

                ElementType& elL = *elptrL;
                ElementType& elR = *elptrR;

                int geo_order = std::max(elL.trans->order, elR.trans->order);

                auto geo_order_dispatch = [&]<int geo_order>() -> int{
                    TraceTypeKey trace_key = { 
                        .basis_order_l = elL.basis->getPolynomialOrder(),
                        .basis_order_r = elR.basis->getPolynomialOrder(),
                        .basis_order_trace = std::max(elL.basis->getPolynomialOrder(), elR.basis->getPolynomialOrder()), 
                        .geometry_order = geo_order,
                        .domain_type = fac->domain_type(),
                        .qtype = quadrature_type,
                        .face_info_l = fac->face_infoL,
                        .face_info_r = fac->face_infoR
                    };

                    if(ref_trace_map.find(trace_key) == ref_trace_map.end()){
                        ref_trace_map[trace_key] = ReferenceTraceType(fac.get(),
                            basis_type, quadrature_type, 
                            *(elL.basis), *(elR.basis),
                            std::integral_constant<int, basis_order>{},
                            std::integral_constant<int, geo_order>{});
                    }
                    ReferenceTraceType &ref_trace = ref_trace_map[trace_key];
                    
                    if(is_interior || fac->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM){
                        // parallel bdy faces are essentially also interior faces 
                        // aside from being a bit *special* :3
                        TraceType trace{ fac.get(), &elL, &elR, ref_trace.trace_basis.get(),
                            ref_trace.quadrule.get(),
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_l},
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_r},
                            (IDX) traces.size() };
                        traces.push_back(trace);
                    } else {
                        TraceType trace = TraceType::make_bdy_trace_space(
                            fac.get(), &elL, ref_trace.trace_basis.get(), 
                            ref_trace.quadrule.get(), 
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_l},
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_r},
                            (IDX) traces.size());
                        traces.push_back(trace);
                    }

                    return 0;
                };

                NUMTOOL::TMP::invoke_at_index(
                    NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                    geo_order,
                    geo_order_dispatch                    
                );
            }
            // reuse the face indexing from the mesh
            interior_trace_start = meshptr->interiorFaceStart;
            interior_trace_end = meshptr->interiorFaceEnd;
            bdy_trace_start = meshptr->bdyFaceStart;
            bdy_trace_end = meshptr->bdyFaceEnd;

            // ===================================
            // = Build the connectivity matrices =
            // ===================================

            // generate the face surrounding nodes connectivity matrix 
            std::vector<std::vector<IDX>> connectivity_ragged(meshptr->n_nodes());
            for(int itrace = 0; itrace < traces.size(); ++itrace){
                const TraceType& trace = traces[itrace];
                for(IDX inode : trace.face->nodes_span()){
                    connectivity_ragged[inode].push_back(itrace);
                }
            }
            fac_surr_nodes = util::crs{connectivity_ragged};

            el_surr_nodes = util::crs{meshptr->elsup};
        } 

        /// @brief construct an FESpace that represents an isoparametric CG space
        /// to the given mesh 
        /// @param meshptr pointer to the mesh
        FESpace(MeshType *meshptr) 
        requires(conformity == h1_conformity(ndim))
        : meshptr(meshptr), elements{}, dofs{meshptr->conn_el}, dof_partitioning{meshptr->node_partitioning}{
            
            // Generate the Finite Elements
            elements.reserve(meshptr->nelem());
            for(ElementTransformation<T, IDX, ndim>* geo_trans : meshptr->el_transformations){
                // create the Element Domain type key
                FETypeKey fe_key = {
                    .domain_type = geo_trans->domain_type,
                    .basis_order = geo_trans->order,
                    .geometry_order = geo_trans->order,
                    .qtype = FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE,
                    .btype = FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE 
                };

                // check if an evaluation doesn't exist yet
                if(ref_el_map.find(fe_key) == ref_el_map.end()){
                    ref_el_map[fe_key] = ReferenceElementType(geo_trans->domain_type, geo_trans->order);
                }
                ReferenceElementType &ref_el = ref_el_map[fe_key];
               
                // this will be the index of the new element
                IDX ielem = all_elements.size();

                // create the finite element
                ElementType fe{
                    .trans = geo_trans, 
                    .basis = ref_el.basis.get(),
                    .quadrule = ref_el.quadrule.get(),
                    .qp_evals = std::span<const BasisEvaluation<T, ndim>>{ref_el.evals},
                    .inodes = meshptr->conn_el.rowspan(ielem), // NOTE: meshptr cannot invalidate anymore
                    .coord_el = meshptr->coord_els.rowspan(ielem),
                    .elidx = ielem
                };

                // add to the elements list
                all_elements.push_back(fe);
            }

            /// 
            elements = std::span{all_elements.begin(),
                all_elements.begin() + meshptr->element_partitioning.owned_range_size(mpi::mpi_world_rank())};

            // Generate the Trace Spaces
            traces.reserve(meshptr->faces.size());
            for(const auto& fac : meshptr->faces){
                // NOTE: assuming element indexing is the same as the mesh still
                bool is_interior = fac->bctype == BOUNDARY_CONDITIONS::INTERIOR;
                ElementType *elptrL;
                ElementType *elptrR; 

                if(fac->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM) {
                    auto [jrank, imleft] = decode_mpi_bcflag(fac->bcflag);
                    IDX neighbor_el_pidx = (imleft) ? fac->elemR : fac->elemL;

                    IDX neighbor_el_lidx = meshptr->element_partitioning.inv_p_indices[neighbor_el_pidx];

                    if(imleft){
                        elptrL = &elements[fac->elemL];
                        elptrR = &all_elements[neighbor_el_lidx];
                    } else {
                        elptrL = &all_elements[neighbor_el_lidx];
                        elptrR = &elements[fac->elemR];
                    }
                } else {
                    elptrL = &all_elements[fac->elemL];
                    elptrR = (is_interior) ? &all_elements[fac->elemR] : &all_elements[fac->elemL];
                }
                ElementType& elL = *elptrL;
                ElementType& elR = *elptrR;

                int geo_order = std::max(elL.trans->order, elR.trans->order);

                auto geo_order_dispatch = [&]<int geo_order>() -> int{
                    TraceTypeKey trace_key = { 
                        .basis_order_l = elL.basis->getPolynomialOrder(),
                        .basis_order_r = elR.basis->getPolynomialOrder(),
                        .basis_order_trace = std::max(elL.basis->getPolynomialOrder(), elR.basis->getPolynomialOrder()), 
                        .geometry_order = geo_order,
                        .domain_type = fac->domain_type(),
                        .qtype = FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE,
                        .face_info_l = fac->face_infoL,
                        .face_info_r = fac->face_infoR
                    };

                    if(ref_trace_map.find(trace_key) == ref_trace_map.end()){
                        ref_trace_map[trace_key] = ReferenceTraceType(
                            fac.get(),
                            FESPACE_ENUMS::FESPACE_BASIS_TYPE::LAGRANGE,
                            FESPACE_ENUMS::FESPACE_QUADRATURE::GAUSS_LEGENDRE,
                            *(elL.basis),
                            *(elR.basis),
                            std::integral_constant<int, geo_order>{},
                            std::integral_constant<int, geo_order>{});
                    }
                    ReferenceTraceType &ref_trace = ref_trace_map[trace_key];
                    
                    if(is_interior || fac->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM){
                        // parallel bdy faces are essentially also interior faces 
                        // aside from being a bit *special* :3
                        TraceType trace{ fac.get(), &elL, &elR, ref_trace.trace_basis.get(),
                            ref_trace.quadrule.get(), 
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_l},
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_r},
                            (IDX) traces.size() };
                        traces.push_back(trace);
                    } else {
                        TraceType trace = TraceType::make_bdy_trace_space(
                            fac.get(), &elL, ref_trace.trace_basis.get(), 
                            ref_trace.quadrule.get(),
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_l},
                            std::span<const BasisEvaluation<T, ndim>>{ref_trace.evals_r},
                            (IDX) traces.size());
                        traces.push_back(trace);
                    }

                    return 0;
                };

                NUMTOOL::TMP::invoke_at_index(
                    NUMTOOL::TMP::make_range_sequence<int, 1, MAX_DYNAMIC_ORDER>{},
                    geo_order,
                    geo_order_dispatch                    
                );
            }
            // reuse the face indexing from the mesh
            interior_trace_start = meshptr->interiorFaceStart;
            interior_trace_end = meshptr->interiorFaceEnd;
            bdy_trace_start = meshptr->bdyFaceStart;
            bdy_trace_end = meshptr->bdyFaceEnd;

            // ===================================
            // = Build the connectivity matrices =
            // ===================================

            // generate the face surrounding nodes connectivity matrix 
            std::vector<std::vector<IDX>> connectivity_ragged(meshptr->n_nodes());
            for(int itrace = 0; itrace < traces.size(); ++itrace){
                const TraceType& trace = traces[itrace];
                for(IDX inode : trace.face->nodes_span()){
                    connectivity_ragged[inode].push_back(itrace);
                }
            }
            fac_surr_nodes = util::crs{connectivity_ragged};

            el_surr_nodes = util::crs{meshptr->elsup};

            std::vector<std::vector<IDX>> fac_surr_el_ragged(elements.size());
            for(int itrace = 0; itrace < traces.size(); ++itrace) {
                const TraceType& trace = traces[itrace];
                if(trace.face->bctype == BOUNDARY_CONDITIONS::PARALLEL_COM){
                    // take some extra care to not add the wrong element index
                    auto [jrank, imleft] = decode_mpi_bcflag(trace.face->bcflag);
                    if(imleft){
                        fac_surr_el_ragged[trace.elL.elidx].push_back(itrace);
                    } else {
                        fac_surr_el_ragged[trace.elR.elidx].push_back(itrace);
                    }
                } else {
                    fac_surr_el_ragged[trace.elL.elidx].push_back(itrace);
                    fac_surr_el_ragged[trace.elR.elidx].push_back(itrace);
                }
            }
        }

        /**
         * @brief get the number of degrees of freedom in the entire fespace 
         * multiply this by the nummber of components to get the size requirement for 
         * an fespan or use the built_in function in the dof_map member
         * @return the number of degrees of freedom
         */
        constexpr std::size_t ndof() const noexcept
        {
            return dofs.calculate_size_requirement(1);
        }

        /**
         * @brief get the span that is the subset of the trace space list 
         * that only includes interior traces 
         * @return span over the interior traces 
         */
        std::span<TraceType> get_interior_traces(){
            return std::span<TraceType>{traces.begin() + interior_trace_start,
                traces.begin() + interior_trace_end};
        }

        /**
         * @brief get the span that is the subset of the trace space list 
         * that only includes boundary traces 
         * @return span over the boundary traces 
         */
        std::span<TraceType> get_boundary_traces(){
            return std::span<TraceType>{traces.begin() + bdy_trace_start,
                traces.begin() + bdy_trace_end};
        }

        /**
         * @brief get the element partitioning map 
         */
        [[nodiscard]] inline constexpr 
        auto element_partitioning() const noexcept
        -> pindex_map<IDX>&
        { return meshptr->element_partitioning; }

        auto print_info(std::ostream& out)
        -> std::ostream& {
            out << "Finite Element Space" << std::endl;
            switch(conformity){
                case l2_conformity(ndim):
                    out << "Space Type: ";
                    out << "L2" << std::endl;
                    out << "ndof: " << ndof() << std::endl;
                    break;
                case l2_conformity(ndim):
                    out << "Space Type: ";
                    out << "H1 (isoparametric)" << std::endl;
                    out << "ndof: " << ndof() << std::endl;
                    break;
            }
            return out;
        }

    };

    // Deduction Guides 

    // Isoparametric CG constructor
    template<class T, class IDX, int ndim>
    FESpace(AbstractMesh<T, IDX, ndim>*)
    -> FESpace<T, IDX, ndim, h1_conformity(ndim)>;

    // Uniform basis order DG constructor
    template<class T, class IDX, int ndim, int basis_order>
    FESpace(
        AbstractMesh<T, IDX, ndim> *meshptr,
        FESPACE_ENUMS::FESPACE_BASIS_TYPE basis_type,
        FESPACE_ENUMS::FESPACE_QUADRATURE quadrature_type,
        tmp::compile_int<basis_order> basis_order_arg
    ) -> FESpace<T, IDX, ndim, l2_conformity(ndim)>;
}
