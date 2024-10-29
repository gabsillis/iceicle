/**
 * @brief Class to write pvd files
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include "iceicle/anomaly_log.hpp"
#include "iceicle/element/finite_element.hpp"
#include "iceicle/fe_definitions.hpp"
#include "iceicle/fe_function/fespan.hpp"
#include "iceicle/geometry/geo_element.hpp"
#include <algorithm>
#include <ostream>
#include <stdexcept>
#include <vector>
#include <string>
#include <iceicle/mesh/mesh.hpp>
#include <iceicle/fespace/fespace.hpp>

#include <cassert>
#include <iomanip>
#include <fstream>
#include <filesystem>
namespace iceicle::io {

    namespace impl{

        struct XMLField{
            std::string name;
            std::string value;
        };

        /** @brief write an xml field */
        void write(const XMLField &field, std::ostream &out);

        struct XMLTag {
            std::string name;
            std::vector<XMLField> fields{}; 
        };

        /** @brief write an opening XML tag <tagname fields...>*/
        void write_open(const XMLTag &tag, std::ostream &out);

        /**@brief write an empty XML tag <tagname fields... />*/
        void write_empty(const XMLTag &tag, std::ostream &out);

        /**@brief write a closing XML tag </tagname> */
        void write_close(const XMLTag &tag, std::ostream &out);

        /** write the VTKFile header tag */
        void write_vtu_header(std::ostream &out);

        /** close the VTKFile header tag */
        void write_vtu_footer(std::ostream &out);

        // =====================
        // = VTK Element Types =
        // =====================

        template<typename T, int ndim>
        struct VTKElement {
            // @brief the nodes in the reference domain
            std::vector<MATH::GEOMETRY::Point<T, ndim>> nodes;

            /// @brief the id of the element in VTK
            int vtk_id;

            /// @brief permutation of local nodes for H1 elements
            std::vector<int> permutation;
        };

        // === Linear 2D ===

        template<typename T>
        static VTKElement<T, 2> VTK_TRIANGLE = {
            .nodes = {
                { 0.0,  0.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
            },
            .vtk_id = 5,
            .permutation = {0, 1, 2}
        };


        template<typename T>
        static VTKElement<T, 2> VTK_QUADRATIC_TRIANGLE = {
            .nodes = {
                { 0.0,  0.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
                { 0.5,  0.0},
                { 0.5,  0.5},
                { 0.0,  0.5},
            },
            .vtk_id = 22
        };

        template<typename T>
        static VTKElement<T, 2> VTK_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
            },
            .vtk_id = 9,
            .permutation = { 0, 2, 3, 1 }
        };

        template<typename T>
        static VTKElement<T, 2> VTK_QUADRATIC_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
                { 0.0, -1.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
                {-1.0,  0.0}
            },
            .vtk_id = 23
        };

        template<typename T>
        static VTKElement<T, 2> VTK_BIQUADRATIC_QUAD = {
            .nodes = {
                {-1.0, -1.0},
                { 1.0, -1.0},
                { 1.0,  1.0},
                {-1.0,  1.0},
                { 0.0, -1.0},
                { 1.0,  0.0},
                { 0.0,  1.0},
                {-1.0,  0.0},
                { 0.0,  0.0}
            },
            .vtk_id = 28
        };
        // === Linear 3D ===

        template<typename T>
        static VTKElement<T, 3> VTK_TETRA = {
            .nodes = {
                { 1.0, 0.0, 0.0},
                { 0.0, 1.0, 0.0},
                { 0.0, 0.0, 1.0},
                { 0.0, 0.0, 0.0},
            },
            .vtk_id = 10
        };


        template<typename T>
        static VTKElement<T, 3> VTK_HEXAHEDRON  = {
            .nodes = {
                {-1.0,-1.0,-1.0},
                { 1.0,-1.0,-1.0},
                { 1.0, 1.0,-1.0},
                {-1.0, 1.0,-1.0},
                {-1.0,-1.0, 1.0},
                { 1.0,-1.0, 1.0},
                { 1.0, 1.0, 1.0},
                {-1.0, 1.0, 1.0},
            },
            .vtk_id = 10
        };

        /**
         * @brief get the VTKElement based on the Element 
         * @param el the element 
         * @param basis_order an optional basis order argument
         *        uses the maximum polynomial order between the element and basis order 
         */
        template<typename T, typename IDX, int ndim>
        VTKElement<T, ndim> &get_vtk_element(const ElementTransformation<T, IDX, ndim> *el, int basis_order = 1){

            static VTKElement<T, ndim> NO_ELEMENT{};

            int max_order = std::max(el->order, basis_order);
            if constexpr (ndim == 2){
                switch(el->domain_type){

                    // Triangle type elements
                    case DOMAIN_TYPE::SIMPLEX:
                        switch(max_order){
                            case 1:
                                return VTK_TRIANGLE<T>;
                            case 2:
                                return VTK_QUADRATIC_TRIANGLE<T>;
                            default:
                                return VTK_QUADRATIC_TRIANGLE<T>;
                        }
                   
                    // Quad type elements 
                    case DOMAIN_TYPE::HYPERCUBE:
                        switch (max_order) {
                            case 0: // use case 1
                            case 1:
                                return VTK_QUAD<T>;
                            case 2:
                                return VTK_BIQUADRATIC_QUAD<T>;
                            default:
                                return VTK_QUADRATIC_QUAD<T>;
                        }
                    // default case
                    default:
                        throw std::logic_error("Unsupported element domain for vtu writer.");
                        return VTK_QUAD<T>;
                }

            } else if constexpr(ndim == 3) {

                switch(el->domain_type){

                    // Triangle type elements
                    case DOMAIN_TYPE::SIMPLEX:
                        switch(max_order){
                            case 1:
                                return VTK_TETRA<T>;
                            default:
                                return VTK_TETRA<T>;
                        }
                   
                    // Quad type elements 
                    case DOMAIN_TYPE::HYPERCUBE:
                        switch (max_order) {
                            case 1:
                                return VTK_HEXAHEDRON<T>;
                            default:
                                return VTK_HEXAHEDRON<T>;
                        }
                    // default case
                    default:
                        throw std::logic_error("Unsupported element domain for vtu writer.");
                        return VTK_HEXAHEDRON<T>;
                }
            } else {
                throw std::logic_error("Unsupported dimension for pvd writer.");
                return NO_ELEMENT; 
            }
        }
    }

    template<typename T, typename IDX, int ndim>
    class PVDWriter{

        /**
        * @brief a data field for output 
        * overridable write_data function
        */
        struct writeable_field {
            /// @brief adds the xml DataArray tags and data to a given vtu file 
            virtual void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim> &fespace) const = 0;

            /// @brief if this data is in dg format and requires duplicated mesh nodes
            virtual auto is_dg_format() const -> bool { return true; }

            virtual auto clone() const -> std::unique_ptr<writeable_field> = 0;

            virtual ~writeable_field(){};
        };

        /**
         * @brief a data field for outputting residuals 
         * 
         * given a list of field names and the fespan that points to the data 
         * i.e {"mass_conservation", "mom_conservation_u", "mom_conservation_v", "energy_conservation"}
         * output total pde residuals summed over test functions
         * and residuals summed over test functions broken down as 
         * - domain integrals 
         * - trace integrals
         *
         * NOTE: This treats the residuals as a function in the test function space 
         * This will be most useful with nodal basis functions where the 
         * output at nodes will correspond to the same test function
         *
         * @tparam LayoutPolicy layout policy of data 
         * @tparam AccessorPolicy accessor policy of the data 
         * @tparam disc_t the type of the discretization
         */
        template< class LayoutPolicy, class AccessorPolicy, class disc_t >
        struct ResidualsField final : public writeable_field {

            /// the data view
            fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// names for each residual
            std::vector< std::string > residual_names;

            /// the discretization
            disc_t& disc;

            /// Constructor
            ResidualsField(fespan<T, LayoutPolicy, AccessorPolicy> fedata, 
                std::vector<std::string> residual_names, disc_t& disc)
            : fedata{fedata}, residual_names{residual_names}, disc{disc}
            {}

            void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim>& fespace) const override 
            {
                using namespace impl;
                using Element = FiniteElement<T, IDX, ndim>;
                // compute the number of vtk points 
                std::size_t n_vtk_poin = 0;
                std::vector<std::size_t> el_vtk_offsets{0};
                IDX iel = 0;
                for(Element &el : fespace.elements){
                    // NOTE: using the vtk el based on basis polynomial order
                    VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                    n_vtk_poin += vtk_el.nodes.size();
                    el_vtk_offsets.push_back(el_vtk_offsets[iel] + vtk_el.nodes.size());
                    ++iel;
                }

                std::vector<T> res_storage( residual_names.size() * n_vtk_poin );
                std::mdspan res_mat{ res_storage.data(), n_vtk_poin, residual_names.size() };
                std::vector<T> domain_storage( residual_names.size() * n_vtk_poin );
                std::mdspan domain_mat{ domain_storage.data(), n_vtk_poin, residual_names.size() };
                std::vector<T> trace_storage( residual_names.size() * n_vtk_poin );
                std::mdspan trace_mat{ trace_storage.data(), n_vtk_poin, residual_names.size() };
                std::ranges::fill(res_storage, 0.0);
                std::ranges::fill(domain_storage, 0.0);
                std::ranges::fill(trace_storage, 0.0);

                // storage for element data
                std::vector<T> el_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                std::vector<T> res_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                std::array<T, disc_t::nv_comp> res_poin{}; // residuals at the point

                // === domain integral ===
                std::size_t i_vtk_poin = 0;
                iel = 0;
                for(Element& el : fespace.elements) {
                    dofspan u_el{el_data, fedata.create_element_layout(iel)};
                    dofspan res_el{res_data, fedata.create_element_layout(iel)};
                    extract_elspan(iel, fedata, u_el);

                    res_el = 0;
                    disc.domain_integral(el, u_el, res_el);

                    // sum over test functions at each point
                    // NOTE: using the vtk el based on basis polynomial order
                    VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                    // storage for basis functions 
                    std::vector<T> basis_data(el.nbasis());

                    // get the solution for each point in the vtk element
                    for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                        el.eval_basis(refnode, basis_data.data());

                        // compute residual contribution at given node
                        std::ranges::fill(res_poin, 0.0);
                        for(int ieq = 0; ieq < fedata.nv(); ++ieq){
                            for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                res_poin[ieq] += res_el[idof, ieq] 
                                    * basis_data[idof];
                            }

                            res_mat[i_vtk_poin, ieq] += res_poin[ieq];
                            domain_mat[i_vtk_poin, ieq] += res_poin[ieq];
                        } 
                        ++i_vtk_poin; // increment the point index
                    }

                    ++iel; // increment the element index
                }

                // === interior faces ===
                std::vector<T> uL_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                std::vector<T> uR_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                std::vector<T> resL_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                std::vector<T> resR_data(fespace.dg_map.max_el_size_reqirement(disc_t::nv_comp));
                for(const auto& trace : fespace.get_interior_traces()) {
                    // compact data views 
                    dofspan uL{uL_data.data(), fedata.create_element_layout(trace.elL.elidx)};
                    dofspan uR{uR_data.data(), fedata.create_element_layout(trace.elR.elidx)};

                    // compact residual views
                    dofspan resL{resL_data.data(), fedata.create_element_layout(trace.elL.elidx)};
                    dofspan resR{resR_data.data(), fedata.create_element_layout(trace.elL.elidx)};

                    resL = 0; 
                    resR = 0;

                    // extract the compact values from the global u view 
                    extract_elspan(trace.elL.elidx, fedata, uL);
                    extract_elspan(trace.elR.elidx, fedata, uR);

                    disc.trace_integral(trace, fespace.meshptr->coord, uL, uR, resL, resR);

                    { // left
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(trace.elL.trans, trace.elL.basis->getPolynomialOrder());
                        std::vector<T> basis_data(trace.elL.nbasis());

                        for(int inode = 0; inode < vtk_el.nodes.size(); ++inode){
                            auto& refnode = vtk_el.nodes[inode];
                            trace.elL.eval_basis(refnode, basis_data.data());

                            // compute residual contribution at given node
                            std::ranges::fill(res_poin, 0.0);
                            for(int ieq = 0; ieq < fedata.nv(); ++ieq){
                                for(std::size_t idof = 0; idof < trace.elL.nbasis(); ++idof){
                                    res_poin[ieq] += resL[idof, ieq] 
                                        * basis_data[idof];
                                }

                                res_mat[el_vtk_offsets[trace.elL.elidx] + inode, ieq] += res_poin[ieq];
                                trace_mat[el_vtk_offsets[trace.elL.elidx] + inode, ieq] += res_poin[ieq];
                            } 

                        }
                    }
                    { // right
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(trace.elR.trans, trace.elR.basis->getPolynomialOrder());
                        std::vector<T> basis_data(trace.elR.nbasis());

                        for(int inode = 0; inode < vtk_el.nodes.size(); ++inode){
                            auto& refnode = vtk_el.nodes[inode];
                            trace.elR.eval_basis(refnode, basis_data.data());

                            // compute residual contribution at given node
                            std::ranges::fill(res_poin, 0.0);
                            for(int ieq = 0; ieq < fedata.nv(); ++ieq){
                                for(std::size_t idof = 0; idof < trace.elR.nbasis(); ++idof){
                                    res_poin[ieq] += resR[idof, ieq] 
                                        * basis_data[idof];
                                }

                                res_mat[el_vtk_offsets[trace.elR.elidx] + inode, ieq] += res_poin[ieq];
                                trace_mat[el_vtk_offsets[trace.elR.elidx] + inode, ieq] += res_poin[ieq];
                            } 
                        }
                    }
                }

                // === Boundary Faces ===
                for(const auto& trace : fespace.get_boundary_traces()) {
                    // compact data views 
                    dofspan uL{uL_data.data(), fedata.create_element_layout(trace.elL.elidx)};
                    dofspan uR{uR_data.data(), fedata.create_element_layout(trace.elR.elidx)};

                    // compact residual views
                    dofspan resL{resL_data.data(), fedata.create_element_layout(trace.elL.elidx)};

                    resL = 0; 

                    // extract the compact values from the global u view 
                    extract_elspan(trace.elL.elidx, fedata, uL);
                    extract_elspan(trace.elR.elidx, fedata, uR);

                    disc.boundaryIntegral(trace, fespace.meshptr->coord, uL, uR, resL);
                    { // left
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(trace.elL.trans, trace.elL.basis->getPolynomialOrder());
                        std::vector<T> basis_data(trace.elL.nbasis());

                        for(int inode = 0; inode < vtk_el.nodes.size(); ++inode){
                            auto& refnode = vtk_el.nodes[inode];
                            trace.elL.eval_basis(refnode, basis_data.data());

                            // compute residual contribution at given node
                            std::ranges::fill(res_poin, 0.0);
                            for(int ieq = 0; ieq < fedata.nv(); ++ieq){
                                for(std::size_t idof = 0; idof < trace.elL.nbasis(); ++idof){
                                    res_poin[ieq] += resL[idof, ieq] 
                                        * basis_data[idof];
                                }

                                res_mat[el_vtk_offsets[trace.elL.elidx] + inode, ieq] += res_poin[ieq];
                                trace_mat[el_vtk_offsets[trace.elL.elidx] + inode, ieq] += res_poin[ieq];
                            } 
                        }
                    }
                }

                // === write full residuals ===
                for(std::size_t ifield = 0; ifield < residual_names.size(); ++ifield){
                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", residual_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    std::size_t i_vtk_poin = 0;
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());
                        for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                            vtu_file << " " << res_mat[i_vtk_poin, ifield];
                            ++i_vtk_poin;
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }

                // === write domain residuals ===
                for(std::size_t ifield = 0; ifield < residual_names.size(); ++ifield){
                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", std::string{"domain_"} + residual_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    std::size_t i_vtk_poin = 0;
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());
                        for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                            vtu_file << " " << domain_mat[i_vtk_poin, ifield];
                            ++i_vtk_poin;
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }

                // === write trace residuals ===
                for(std::size_t ifield = 0; ifield < residual_names.size(); ++ifield){
                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", std::string{"trace_"} + residual_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    std::size_t i_vtk_poin = 0;
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());
                        for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                            vtu_file << " " << trace_mat[i_vtk_poin, ifield];
                            ++i_vtk_poin;
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<ResidualsField<LayoutPolicy, AccessorPolicy, disc_t>>(*this);
            }
        };

        // TODO: L2 mdg residuals (corrigan formulation)

        /**
        * @brief a data field for output 
        * contains a name list that will be the field names in VTK 
        * and an fespan that points to the data 
        * and a callback function and name list for derived fields 
        */
        template< class LayoutPolicy, class AccessorPolicy>
        struct PVDDataField final : public writeable_field {
            /// the data view
            fespan<T, LayoutPolicy, AccessorPolicy> fedata;

            /// the field name for each vector component of fespan 
            std::vector<std::string> field_names;

            // Function to get derived fields from the current state
            std::function<std::vector<T>(const T *)> derived_fields_callback;

            /// the field name for each derived field
            std::vector<std::string> derived_field_names;

            PVDDataField(fespan<T, LayoutPolicy, AccessorPolicy> fedata,
                std::vector<std::string> field_names, 
                std::function<std::vector<T>(const T *)> derived_fields_callback,
                std::vector<std::string> derived_field_names
            ) : fedata{fedata}, field_names{field_names}, derived_fields_callback{derived_fields_callback},
                derived_field_names{derived_field_names}
            {
                if(field_names.size() != fedata.nv()) 
                    util::AnomalyLog::log_anomaly("field names size does not match number of fields");
            }

            /// @brief adds the xml DataArray tags and data to a given vtu file 
            void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim> &fespace) const override
            {
                using namespace impl;
                using Element = FiniteElement<T, IDX, ndim>;

                // compute the number of vtk points 
                std::size_t n_vtk_poin = 0;
                for(Element &el : fespace.elements){
                    // NOTE: using the vtk el based on basis polynomial order
                    VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                    n_vtk_poin += vtk_el.nodes.size();
                }

                // storage for the fields 
                std::vector<T> output_storage( (field_names.size() + derived_field_names.size()) * n_vtk_poin );
                std::mdspan output_mat{output_storage.data(), n_vtk_poin, field_names.size() + derived_field_names.size()};

                std::size_t i_vtk_poin = 0;
                // loop over the elements
                for(Element &el : fespace.elements){

                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());

                        // get the solution for each point in the vtk element
                        for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                            el.eval_basis(refnode, basis_data.data());

                            // compute the pde variables
                            std::vector<T> u(fedata.nv());
                            std::ranges::fill(u, 0.0);
                            for(int ieq = 0; ieq < fedata.nv(); ++ieq){
                                for(std::size_t idof = 0; idof < el.nbasis(); ++idof){
                                    u[ieq] += fedata[el.elidx, idof, ieq] 
                                        * basis_data[idof];
                                }

                                output_mat[i_vtk_poin, ieq] = u[ieq];
                            } 

                            // compute derived variables 
                            std::vector<T> u_derived = derived_fields_callback(u.data());
                            for(int ifield = 0; ifield < derived_field_names.size(); ++ifield){
                                output_mat[i_vtk_poin, fedata.nv() + ifield] = u_derived[ifield];
                            }
                            ++i_vtk_poin; // increment the point index
                        }
                }

                // print out pde variables
                for(std::size_t ifield = 0; ifield < field_names.size(); ++ifield){
                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", field_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    std::size_t i_vtk_poin = 0;
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());
                        for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                            vtu_file << " " << output_mat[i_vtk_poin, ifield];
                            ++i_vtk_poin;
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }

                // print out derived variables
                for(std::size_t ifield = 0; ifield < derived_field_names.size(); ++ifield){
                    // point data tag
                    write_open(XMLTag{"DataArray", {
                        {"type", "Float64"},
                        {"Name", derived_field_names[ifield]}, 
                        {"format", "ascii"}
                    }}, vtu_file);

                    // loop over the elements
                    std::size_t i_vtk_poin = 0;
                    for(Element &el : fespace.elements){
                        // NOTE: using the vtk el based on basis polynomial order
                        VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());

                        // storage for basis functions 
                        std::vector<T> basis_data(el.nbasis());
                        for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                            vtu_file << " " << output_mat[i_vtk_poin, fedata.nv() + ifield];
                            ++i_vtk_poin;
                        }

                        // line break for each element
                        vtu_file << std::endl;
                    }

                    // close the data array tag
                    write_close(XMLTag{"DataArray"}, vtu_file);
                }
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<PVDDataField<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };

        /**
         * @brief a data field for MDG nodal vector data output 
         * contains field_name
         * and a node_selection_span for the data 
         *
         * writes nodal data for a nodal version of the mesh 
         * WARNING: do not couple with dg mesh
         */
        template< class LayoutPolicy, class AccessorPolicy>
        struct MDGVectorDataField final : public writeable_field {
            using value_type = T; 
            using index_type = LayoutPolicy::index_type;
            ///  the data view 
            mutable dofspan<T, LayoutPolicy, AccessorPolicy> mdgdata;
            static_assert(node_selection_span<decltype(mdgdata)>, 
                    "Must be a node selection span to be mdg data");

            /// the field name 
            std::string field_name;

            /// @brief constructor with argument forwarding for the vector constructor
            MDGVectorDataField(dofspan<value_type, LayoutPolicy, AccessorPolicy> mdgdata, std::string field_name)
            : mdgdata(mdgdata), field_name(field_name){}

            void write_data(std::ofstream &vtu_file, FESpace<T, IDX, ndim>& fespace) const override {
                using namespace impl;
                const nodeset_dof_map<index_type>& nodeset = mdgdata.get_layout().nodeset;

                write_open(XMLTag{"DataArray", {
                    {"type", "Float64"},
                    {"Name", field_name}, 
                    {"NumberOfComponents", std::to_string(mdgdata.nv())},
                    {"format", "ascii"}
                }}, vtu_file);

                for(index_type inode = 0; inode < fespace.meshptr->n_nodes(); ++inode){
                    index_type idof = nodeset.inv_selected_nodes[inode];
                    if(idof == nodeset.selected_nodes.size()){
                        for(index_type iv = 0; iv < mdgdata.nv(); ++iv){
                            vtu_file << "0.0 ";
                        }
                        vtu_file << std::endl;
                    } else {
                        for(index_type iv = 0; iv < mdgdata.nv(); ++iv){
                            vtu_file << mdgdata[idof, iv] << " ";
                        }
                        vtu_file << std::endl;
                    }
                }
                write_close(XMLTag{"DataArray"}, vtu_file);
            }

            auto is_dg_format() const -> bool override {
                return false;
            }

            auto clone() const -> std::unique_ptr<writeable_field> override {
                return std::make_unique<MDGVectorDataField<LayoutPolicy, AccessorPolicy>>(*this);
            }
        };

        private:

        AbstractMesh<T, IDX, ndim> *meshptr = nullptr;
        FESpace<T, IDX, ndim> *fespace_ptr = nullptr;
        std::vector<std::unique_ptr<writeable_field>> fields;

        // @brief callback function for when no derived fields are used
        static constexpr 
        auto empty_derived_fields_callback(const T *)
        -> std::vector<T>
        { return std::vector<T>{}; }

        public:
        using value_type = T;

        int print_precision = 8;
        std::string collection_name = "data";
        std::filesystem::path data_directory;

        PVDWriter() : data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        PVDWriter(AbstractMesh<T, IDX, ndim> *meshptr)
        : meshptr(meshptr), data_directory(std::filesystem::current_path()) {
            data_directory /= "iceicle_data";
        }

        PVDWriter(const PVDWriter<T, IDX, ndim>& other)
            : meshptr(other.meshptr), fespace_ptr(other.fespace_ptr), fields{}, print_precision(other.print_precision),
              collection_name(other.collection_name), data_directory(other.data_directory)
        {
            for(const std::unique_ptr<writeable_field>& field : other.fields){
                fields.push_back(field->clone());
            }
        }

        PVDWriter(PVDWriter<T, IDX, ndim>&& other) = default;
              

        void register_mesh(AbstractMesh<T, IDX, ndim> *newptr){
            meshptr = newptr;
        }

        /// @brief register an fespace to this writer 
        /// will overwrite the registered mesh to the one in the fespace
        void register_fespace(FESpace<T, IDX, ndim> &fespace){
            fespace_ptr = &fespace;
            meshptr = fespace.meshptr;
        }

        /**
         * @brief register a set of fields represented in an fespan 
         * @param fedata the global data view to write to files 
         * @param field_names the names for each field in fe_data 
         */
        template< class LayoutPolicy, class AccessorPolicy >
        void register_fields(
            fespan<T, LayoutPolicy, AccessorPolicy> &fedata,
            std::vector<std::string> field_names,
            std::function<std::vector<T>(const T*)> derived_fields_callback = empty_derived_fields_callback,
            std::vector<std::string> derived_field_names = std::vector<std::string>{}
        ) {
            // create the field handle and add it to the list
            auto field_ptr = std::make_unique<PVDDataField<LayoutPolicy, AccessorPolicy>>(
                    fedata, field_names, derived_fields_callback, derived_field_names);
            fields.push_back(std::move(field_ptr));
        }

        /**
         * @brief register a set of residuals over an fespan with given discretization 
         * @param fedata the solution data 
         * @param residual_names the names for each residual 
         * @param disc the discretiation 
         */
        template< class LayoutPolicy, class AccessorPolicy, class disc_t >
        void register_residuals(
            fespan<T, LayoutPolicy, AccessorPolicy> &fedata,
            std::vector<std::string> residual_names,
            disc_t& disc
        ) {
            auto field_ptr = std::make_unique<ResidualsField<LayoutPolicy, AccessorPolicy, disc_t>>(fedata, residual_names, disc);
            fields.push_back(std::move(field_ptr));
        }

        template< class LayoutPolicy, class AccessorPolicy>
        void register_fields(dofspan<T, LayoutPolicy, AccessorPolicy>& nodal_data, std::string_view field_name){
            auto field_ptr = std::make_unique<MDGVectorDataField<LayoutPolicy, AccessorPolicy>>(
                    nodal_data, std::string{field_name});
            fields.push_back(std::move(field_ptr));
        }

        private:
        auto write_cg_unstructured_grid(std::ofstream& out){
            using namespace impl;
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;
            write_open(XMLTag{"Piece", {
                {"NumberOfPoints", std::to_string(meshptr->n_nodes())},
                {"NumberOfCells", std::to_string(meshptr->nelem())}
            }}, out);

            // ===================
            // = write the nodes =
            // ===================
            write_open(XMLTag{"Points"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Float64"},
                {"Name", "Points"},
                {"NumberOfComponents", "3"},
                {"format", "ascii"}
            }}, out);

            for(IDX inode = 0; inode < meshptr->n_nodes(); ++inode){
                for(int idim = 0; idim < ndim; ++idim){
                    out << meshptr->coord[inode][idim] << " ";
                }
                for(int idim = ndim; idim < 3; ++idim){
                    out << 0.0;
                }
                out << std::endl;
            }
            write_close(XMLTag{"DataArray"}, out);
            write_close(XMLTag{"Points"}, out);

            // ===============
            // = write cells =
            // ===============
            
            // === connectivity ===
            write_open(XMLTag{"Cells"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "connectivity"},
                {"format", "ascii"}
            }}, out);
            
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.trans->order);
                if(vtk_el.nodes.size() != el.trans->nnode){
                    AnomalyLog::log_anomaly(Anomaly{"must have a matching vtk element for cg data", general_anomaly_tag{}});
                }

                // TODO: FIX AND GENERALIZE TO OTHER ELEMENT TYPES 
                // convert to paraview ordering
                std::span<const IDX> nodes = el.inodes;
                out << std::to_string(nodes[0]) << " "
                    << std::to_string(nodes[2]) << " "
                    << std::to_string(nodes[3]) << " "
                    << std::to_string(nodes[1]) << " ";

//                for(IDX inode : el.geo_el->nodes_span()){
//                    out << std::to_string(inode) << " ";
//                }
                out << std::endl;
            }
            write_close(XMLTag{"DataArray"}, out);

            // === offsets ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "offsets"},
                {"format", "ascii"}
            }}, out);
            std::size_t goffset = 0;
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.trans->order);
                goffset += vtk_el.nodes.size();
                out << std::to_string(goffset) << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            // === cell types ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "types"},
                {"format", "ascii"}
            }}, out);
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.trans->order);
                out << vtk_el.vtk_id  << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            write_close(XMLTag{"Cells"}, out);

            // ===================
            // = write PointData =
            // ===================
            write_open(XMLTag{"PointData"}, out);
            for(auto &field_ptr : fields){
                field_ptr->write_data(out, *fespace_ptr);
            }
            write_close(XMLTag{"PointData"}, out);
            write_close(XMLTag{"Piece"}, out);
        }

        auto write_dg_unstructured_grid(std::ofstream &out) -> void {
            using namespace impl;
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;
            // count the number of nodes (duplicate for each element)
            std::size_t nodecount = 0;
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                nodecount += vtk_el.nodes.size();
            }
            write_open(XMLTag{"Piece", {
                {"NumberOfPoints", std::to_string(nodecount)},
                {"NumberOfCells", std::to_string(meshptr->nelem())}
            }}, out);


            // ===================
            // = write the nodes =
            // ===================

            write_open(XMLTag{"Points"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Float64"},
                {"Name", "Points"},
                {"NumberOfComponents", "3"},
                {"format", "ascii"}
            }}, out);

            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                MATH::GEOMETRY::Point<T, ndim> point_phys;
                for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                    // interpolate the point from the vtk element 
                    point_phys = el.transform(refnode);

                    // print the physical node
                    for(int idim = 0; idim < ndim; ++idim)
                    { out << point_phys[idim] << " "; }

                    // fill with zeros
                    for(int idim = ndim; idim < 3; ++idim)
                    { out << 0.0 << " "; }
                    out << std::endl;
                }
            }
            write_close(XMLTag{"DataArray"}, out);
            write_close(XMLTag{"Points"}, out);

            // ===============
            // = write cells =
            // ===============
            
            // === connectivity ===
            write_open(XMLTag{"Cells"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "connectivity"},
                {"format", "ascii"}
            }}, out);
            
            std::size_t ignode = 0;
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                    out << std::to_string(ignode++) << " ";
                }
                out << std::endl;
            }
            write_close(XMLTag{"DataArray"}, out);

            // === offsets ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "offsets"},
                {"format", "ascii"}
            }}, out);
            std::size_t goffset = 0;
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                goffset += vtk_el.nodes.size();
                out << std::to_string(goffset) << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            // === cell types ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "types"},
                {"format", "ascii"}
            }}, out);
            for(Element &el : fespace_ptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(el.trans, el.basis->getPolynomialOrder());
                out << vtk_el.vtk_id  << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            write_close(XMLTag{"Cells"}, out);

            // ===================
            // = write PointData =
            // ===================
            write_open(XMLTag{"PointData"}, out);
            for(auto &field_ptr : fields){
                field_ptr->write_data(out, *fespace_ptr);
            }
            write_close(XMLTag{"PointData"}, out);

            write_close(XMLTag{"Piece"}, out);

        }

        public:
        /**
         * @brief write the mesh and field values in a .vtu file 
         * @param itime the timestep
         * @param time the time value
         * NOTE: the user is responsible for making sure itime and time are unique 
         * (aside from parallel case, separate file names are generated per process in parallel)
         */
        void write_vtu(int itime, T time){

            if(fespace_ptr == nullptr) {
                throw std::logic_error("fespace pointer not set");
            }

            using namespace impl;
            using namespace util;
            using Element = FiniteElement<T, IDX, ndim>;

            // create the path if it doesn't exist
            std::filesystem::create_directories(data_directory);

            std::filesystem::path mesh_path = data_directory;
            mesh_path /= (collection_name + std::to_string(itime) + ".vtu");

            std::ofstream out{mesh_path};
            if(!out) {
                throw std::logic_error("could not open mesh file for writing.");
            }

            if(!meshptr){
                throw std::logic_error("mesh doesn't exist");
            }

            // setup the output stream 
            out << std::setprecision(print_precision);

            write_vtu_header(out);
            
            bool use_dg_mesh = fields[0]->is_dg_format();

            for(int ifield = 1; ifield < fields.size(); ++ifield){
                if(fields[ifield]->is_dg_format() != use_dg_mesh){
                    AnomalyLog::log_anomaly(Anomaly{"Cannot mix dg and cg fields", general_anomaly_tag{}});
                }
            }

            write_open(XMLTag{"UnstructuredGrid"}, out);

            if(use_dg_mesh) write_dg_unstructured_grid(out);
            else write_cg_unstructured_grid(out);

            write_close(XMLTag{"UnstructuredGrid"}, out);
            write_vtu_footer(out);
        }

        

        /**
         * @brief writes a .vtu file called "mesh.vtu"
         * that only contains the mesh 
         *
         * NOTE: this has no context of the finite element space 
         * so will use the minimum polynomial order to capture the 
         * mesh geometry (may not match up to solutions)
         */
        void write_mesh(){
            using namespace impl;
            using Element = GeometricElement<T, IDX, ndim>;

            // create the path if it doesn't exist
            std::filesystem::create_directories(data_directory);

            // create the mesh file 
            std::filesystem::path mesh_path = data_directory;
            mesh_path /= "mesh.vtu";

            std::ofstream out{mesh_path};
            if(!out) {
                throw std::logic_error("could not open mesh file for writing.");
            }

            if(!meshptr){
                throw std::logic_error("mesh doesn't exist");
            }

            // setup the output stream 
            out << std::setprecision(print_precision);

            write_vtu_header(out);
            
            // count the number of nodes (duplicate for each element)
            std::size_t nodecount = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                nodecount += vtk_el.nodes.size();
            }

            write_open(XMLTag{"UnstructuredGrid"}, out);
            write_open(XMLTag{"Piece", {
                {"NumberOfPoints", std::to_string(nodecount)},
                {"NumberOfCells", std::to_string(meshptr->elements.size())}
            }}, out);

            // ===================
            // = write the nodes =
            // ===================

            write_open(XMLTag{"Points"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Float64"},
                {"Name", "Points"},
                {"NumberOfComponents", "3"},
                {"format", "ascii"}
            }}, out);

            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                MATH::GEOMETRY::Point<T, ndim> point_phys;
                for(const MATH::GEOMETRY::Point<T, ndim> &refnode : vtk_el.nodes){
                    // interpolate the point from the vtk element 
                    elptr->transform(meshptr->nodes, refnode, point_phys);

                    // print the physical node
                    for(int idim = 0; idim < ndim; ++idim)
                    { out << point_phys[idim] << " "; }

                    // fill with zeros
                    for(int idim = ndim; idim < 3; ++idim)
                    { out << 0.0 << " "; }
                    out << std::endl;
                }
            }
            write_close(XMLTag{"DataArray"}, out);
            write_close(XMLTag{"Points"}, out);

            // ===============
            // = write cells =
            // ===============
            
            // === connectivity ===
            write_open(XMLTag{"Cells"}, out);
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "connectivity"},
                {"format", "ascii"}
            }}, out);
            
            std::size_t ignode = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                for(int ilnode = 0; ilnode < vtk_el.nodes.size(); ++ilnode){
                    out << std::to_string(ignode++) << " ";
                }
                out << std::endl;
            }
            write_close(XMLTag{"DataArray"}, out);

            // === offsets ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "offsets"},
                {"format", "ascii"}
            }}, out);
            std::size_t goffset = 0;
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                goffset += vtk_el.nodes.size();
                out << std::to_string(goffset) << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            // === cell types ===
            write_open(XMLTag{"DataArray", {
                {"type", "Int64"},
                {"Name", "types"},
                {"format", "ascii"}
            }}, out);
            for(Element *elptr : meshptr->elements){
                VTKElement<T, ndim> &vtk_el = get_vtk_element(elptr);
                out << vtk_el.vtk_id  << " ";
            }
            out << "\n";
            write_close(XMLTag{"DataArray"}, out);

            write_close(XMLTag{"Cells"}, out);

            write_close(XMLTag{"Piece"}, out);
            write_close(XMLTag{"UnstructuredGrid"}, out);
            write_vtu_footer(out);
        }
    };


}
