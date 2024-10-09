/**
 * @brief utility to log anomalies 
 * (Errors, Warnings, Unexpected situations)
 * without incurring the overhead of exceptions,
 * but withouut the added stack unwinding capabilities of exceptions 
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#pragma once
#include <algorithm>
#include <cstddef>
#include <iostream>
#include <limits>
#include <vector>
#include <memory>
#include <ostream>
#include <source_location>
#include <string>
namespace iceicle::util {

    class AbstractAnomaly {
        friend class AnomalyLog;

        public:
        virtual ~AbstractAnomaly() = default;
        private:
        virtual void handle_self(std::ostream &log_out) = 0;

    };

    template<class Data>
    class Anomaly : public AbstractAnomaly {
    private:
        std::string desc;
        Data user_data;
        std::source_location loc;

    public:
        Anomaly(
            std::string_view desc, 
            const Data &data,
            const std::source_location &loc = std::source_location::current()
        ) : desc{desc}, user_data{data}, loc{loc}
        {}

        const std::string &what() const noexcept { return desc; }
        const std::source_location &where() const noexcept { return loc; }
        const Data& data() const noexcept { return user_data; }

        void handle_self (std::ostream &log_out) override;
    };

    template<class Data>
    Anomaly(std::string_view, const Data&) -> Anomaly<Data>;
    template<class Data>
    Anomaly(std::string_view, const Data &, std::source_location) -> Anomaly<Data>;

    /** @brief overload for stream out of std::source_location */
    inline std::ostream& operator<<(std::ostream& os, const std::source_location &loc){
        os << loc.file_name() << "("
           << loc.line() << ":"
           << loc.column() << "), function `"
           << loc.function_name() << "`";
        return os;
    }

//    /** @brief overload for stream out of std::stacktrace */
//    inline std::ostream& operator<<(std::ostream& os, const std::stacktrace& backtrace){
//        for(auto &trace_item : backtrace){
//            os << trace_item.source_file() << "(" << trace_item.source_line()
//               << "):" << trace_item.description() << std::endl;
//        }
//        return os;
//    } 

    /** @brief anomaly tag to mark a failed expect() statement */
    struct expectation_anomaly_tag{ using anomaly_tag = expectation_anomaly_tag; };

    /** @brief anomaly tag for bounds issues */ 
    template<class index_type, class bounds_type>
    struct bounds_anomaly_tag{
        using anomaly_tag = bounds_anomaly_tag<index_type, bounds_type>;
        static constexpr bounds_type unbounded = std::numeric_limits<bounds_type>::max();
        index_type index;
        bounds_type bounds_lower;
        bounds_type bounds_upper;
    };

    /// @brief Anomaly for user input text that doesn't match any implementation
    struct text_not_found_tag { 
        using anomaly_tag = text_not_found_tag;
        std::string text;
    };

    /// @brief Anomaly when parsing a file storing the file number
    struct file_parse_tag {
        using anomaly_tag = file_parse_tag;
        std::size_t line_number;
    };

    /** @brief tag to mark general anomalies */
    struct general_anomaly_tag{ using anomaly_tag = general_anomaly_tag; };

    /** @brief anomaly tag to mark an anomaly that can be communicated as a warning */
    struct warning_anomaly_tag{ using anomaly_tag = warning_anomaly_tag; };

    /**
     * @brief default way to handle an anomaly 
     * log to output for every type of anomaly
     *
     * Override based on the data class to customize
     */
    template<class Data>
    void handle_anomaly(const Anomaly<Data> &anomaly, std::ostream &log_out) {
        log_out << "Error: " << anomaly.what() << std::endl 
                << anomaly.where() << std::endl << std::endl;
//                << anomaly.stack() << std::endl << std::endl;
    }

    template<>
    inline void handle_anomaly(const Anomaly<text_not_found_tag>& anomaly, std::ostream& log_out){
        log_out << "Error: " << anomaly.what() << "\"" << anomaly.data().text << "\" at:" << std::endl 
                << anomaly.where() << std::endl << std::endl;
    }

    template<>
    inline void handle_anomaly(const Anomaly<file_parse_tag>& anomaly, std::ostream& log_out){
        log_out << "Error on line " << anomaly.data().line_number << ": " << anomaly.what() << "\" at:" << std::endl 
                << anomaly.where() << std::endl << std::endl;
    }

    template<>
    inline void handle_anomaly(const Anomaly<warning_anomaly_tag> &anomaly, std::ostream &log_out) {
        log_out << "Warning: " << anomaly.what() << std::endl 
                << anomaly.where() << std::endl << std::endl;
    }

    template<class index_type, class bounds_type>
    inline void handle_anomaly(const Anomaly<bounds_anomaly_tag<index_type, bounds_type>> &anomaly,
            std::ostream &log_out) {
        log_out << "Index out of bounds: " << anomaly.what() << std::endl 
                << "index " << anomaly.data().index << "is not between "
                << anomaly.data().bounds_lower << " and " << anomaly.data().bounds_upper
                << anomaly.where() << std::endl << std::endl;
    }
    template<class Data>
    void Anomaly<Data>::handle_self(std::ostream &log_out){
        handle_anomaly(*this, log_out);
    }

    class AnomalyLog {
        private: 
            inline static std::vector<std::unique_ptr<AbstractAnomaly>> anomalies;
            AnomalyLog() = default;
            AnomalyLog(const AnomalyLog&) = delete;
            AnomalyLog& operator=(const AnomalyLog&) = delete;
        public:

            /**
             * @brief get the instance of this singleton
             */
            static AnomalyLog &instance() {
                static AnomalyLog instance_;
                return instance_;
            }

            template<class Data>
            static void log_anomaly(Anomaly<Data> anomaly){
                anomalies.push_back(std::make_unique<Anomaly<Data>>(std::move(anomaly)));
            }

            static void log_anomaly(std::string_view message){
                AnomalyLog::log_anomaly(Anomaly{message, general_anomaly_tag{}});
            }

            template<class Data>
            static void check(bool condition, Anomaly<Data> anomaly_on_failure){
                if(!condition) log_anomaly(std::move(anomaly_on_failure));
            }

            // can't get forwarding and source location to play nice
//            template<class... anomaly_argsT>
//            static void log_anomaly(anomaly_argsT&&... args, std::source_location loc = std::source_location::current()){
//                Anomaly anomaly{std::forward<anomaly_argsT>(args)..., loc};
//                using anomalyT = decltype(anomaly);
//                auto anomalyptr = std::make_unique<anomalyT>(anomaly);
//                anomalies.push_back(std::move(anomalyptr));
//            }

            static void handle_anomalies(std::ostream &os = std::cerr){
                for(auto &anomaly : anomalies){
                    anomaly->handle_self(os);
                }
                anomalies.clear();
            }

            ~AnomalyLog(){
                // Log any remaining anommalies in the error stream
                for(auto &anomaly : anomalies){
                    anomaly->handle_self(std::cerr);
                }
                anomalies.clear();
            }

            static auto size() -> std::size_t 
            { return anomalies.size(); }
    };


    /**
     * @brief expect an expression to be true 
     * if false creates an anomaly to express this and logs it
     * @param expect_true the expression that is expected to be true
     * @param message_if_false the message to describe the situation 
     *                         if the expectation is not met 
     */
    inline static void expect(
        bool expect_true,
        std::string_view message_if_false,
        std::source_location loc = std::source_location::current()
    ){
        if(!expect_true){
            AnomalyLog::log_anomaly(Anomaly{"Expectation failed", expectation_anomaly_tag{}, loc});
        }
    }

    // ==============
    // = Usage Goal =
    // ==============

    namespace USAGE_EX {
        struct convergence_anomaly{};

        struct convergence_info{
            using anomaly_tag = convergence_anomaly;
            double norm_value;
            double norm_converged;
            std::size_t ntime;
        };

        inline void example_a(){
            double norm = 1e-6;
            double converged = 1e-8;
            std::size_t ntime = 100;
            if(norm > converged) {
                AnomalyLog::log_anomaly(Anomaly{
                    "Not Converged: reason - maximum timesteps reached",
                    convergence_info{norm, converged, ntime}
                });
            }
        }

        inline void example_b(){
            struct dimensionality_mismatch{};

            struct dimensionality_info{
                using anomaly_tag = dimensionality_mismatch;
                int dimensionality_a;
                int dimensionality_b;
            };

            // log an anomaly
            AnomalyLog::log_anomaly(Anomaly{
                "Dimensionality mismatch",
                dimensionality_info{2, 3}
            });

            // or handle it right away 
            handle_anomaly(Anomaly{"dimensionality_mismatch", dimensionality_info{2, 3}}, std::cerr);
        }

        inline void example_c(){
            example_a();
            example_a();
            example_b();

            AnomalyLog::handle_anomalies();
        }
    }
}
