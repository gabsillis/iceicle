/// @brief type erasure writer class 
/// @author Gianni Absillis (gabsill@ncsu.edu)

#pragma once
#include "iceicle/pvd_writer.hpp"
#include <iceicle/fespace/fespace.hpp>
#include <iceicle/dat_writer.hpp>
#include <memory>
namespace iceicle::io {


    /// @brief external function interface for type erasure to write a file 
    /// writes the file with the given time index and time values 
    template<class T, class IDX, int ndim>
    auto write_file(DatWriter<T, IDX, ndim>& writer, int itime, T time) -> void {
        writer.write_dat(itime, time);
    }

    /// @brief external function interface for type erasure to write a file 
    /// writes the file with the given time index and time values 
    template<class T, class IDX, int ndim>
    auto write_file(PVDWriter<T, IDX, ndim>& writer, int itime, T time) -> void {
        writer.write_vtu(itime, time);
    }

    namespace impl {
        /// @brief external function interface for type erasure to rename the collection
        template<class T, class IDX, int ndim>
        auto rename_collection(DatWriter<T, IDX, ndim>& writer, std::string_view new_name)-> void {
            writer.collection_name = new_name;
        }

        /// @brief external function interface for type erasure to rename the collection
        template<class T, class IDX, int ndim>
        auto rename_collection(PVDWriter<T, IDX, ndim>& writer, std::string_view new_name) -> void {
            writer.collection_name = new_name;
        }
    }

    /// @brief Type erasure class for things that can write values to file
    /// given a time index and time value
    ///
    /// see Klaus Iglberger cppcon 2022 type erasure the implementation details
    class Writer {
        private:

        struct WriterConcept {
            public:
                virtual ~WriterConcept() = default;

                virtual void do_write_file(int itime, double time) = 0;

                virtual auto clone() const -> std::unique_ptr<WriterConcept> = 0;

                virtual void do_rename_collection(std::string_view new_name) = 0;
        };

        template< typename WriterT >
        struct WriterModel : public WriterConcept {
            public:
            using value_type = WriterT::value_type;

            // the writer that is being type erased
            WriterT _writer;

            /// @brief construct from a writer 
            WriterModel(WriterT writer) : _writer{std::move(writer)}{} ;

            /// @brief write to file 
            /// @param itime the time index 
            /// @param time the time 
            auto do_write_file(int itime, double time) -> void override {
                write_file(_writer, itime, time);
            }

            /// @brief rename the collection of a writer
            void do_rename_collection(std::string_view new_name) override {
                impl::rename_collection(_writer, new_name);
            }

            auto clone() const -> std::unique_ptr<WriterConcept> override {
                return std::make_unique<WriterModel>(*this);
            }
        };

        std::unique_ptr<WriterConcept> pimpl;


        public:
        Writer() = default;

        template< typename WriterT>
        Writer(WriterT writer) 
         : pimpl{std::make_unique<WriterModel<WriterT>>(std::move(writer))}
        {}

        /// @brief check if the writer contains an actual writer value
        operator bool() const { return (bool) pimpl; }

        /// @brief write to file free function interface
        /// @param writer the writer
        /// @param itime the time index 
        /// @param time the time 
        void write(
            int itime,
            double time
        ) {
            pimpl->do_write_file(itime, time);
        }

        /// @brief rename the collection of file names output
        /// @param new_name the new name to set the collection to
        void rename_collection(std::string_view new_name) 
        { pimpl->do_rename_collection(new_name); }

        // copy
        Writer( const Writer& other ) 
            : pimpl( other.pimpl->clone() ) 
        {}
        Writer& operator=(const Writer& other) {
            // copy and swap
            Writer tmp(other);
            std::swap(pimpl, tmp.pimpl);
            return *this;
        }

        // move 
        Writer( Writer&& other) = default;
        Writer& operator=( Writer&& other ) = default;

    };


}
