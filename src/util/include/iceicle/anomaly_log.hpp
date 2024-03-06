/**
 * @brief utility to log anomalies 
 * (Errors, Warnings, Unexpected situations)
 * without incurring the overhead of exceptions,
 * but withouut the added stack unwinding capabilities of exceptions 
 *
 * @author Gianni Absillis (gabsill@ncsu.edu)
 */
#include <cstddef>
#include <iostream>
#include <vector>
#include <memory>
#include <ostream>
#include <source_location>
#include <stacktrace>
#include <string>
namespace ICEICLE::UTIL {

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
        std::stacktrace trace;

    public:
        Anomaly(
            std::string_view desc, 
            const Data &data,
            const std::source_location &loc = std::source_location::current(),
            const std::stacktrace &trace = std::stacktrace::current()
        ) : desc{desc}, user_data{data}, loc{loc}, trace{trace}
        {}

        const std::string &what() const noexcept { return desc; }
        const std::source_location &where() const noexcept { return loc; }
        const std::stacktrace &stack() const noexcept { return trace; }
        const Data &data() const noexcept { return data; }

        void handle_self (std::ostream &log_out) override;
    };

    template<class Data>
    Anomaly(std::string_view, const Data&) -> Anomaly<Data>;
    template<class Data>
    Anomaly(std::string_view, const Data &, std::source_location) -> Anomaly<Data>;
    template<class Data>
    Anomaly(std::string_view, const Data &, std::source_location, std::stacktrace) -> Anomaly<Data>;

    /** @brief overload for stream out of std::source_location */
    inline std::ostream& operator<<(std::ostream& os, const std::source_location &loc){
        os << loc.file_name() << "("
           << loc.line() << ":"
           << loc.column() << "), function `"
           << loc.function_name() << "`";
        return os;
    }

    /** @brief overload for stream out of std::stacktrace */
    inline std::ostream& operator<<(std::ostream& os, const std::stacktrace& backtrace){
        for(auto &trace_item : backtrace){
            os << trace_item.source_file() << "(" << trace_item.source_line()
               << "):" << trace_item.description() << std::endl;
        }
        return os;
    } 

    /**
     * @brief default way to handle an anomaly 
     * log to output for every type of anomaly
     *
     * Override based on the data class to customize
     */
    template<class Data>
    void handle_anomaly(const Anomaly<Data> &anomaly, std::ostream &log_out) {
        log_out << "Error: " << anomaly.what() << std::endl 
                << anomaly.where() << std::endl 
                << anomaly.stack() << std::endl << std::endl;
    }

    template<class Data>
    void Anomaly<Data>::handle_self(std::ostream &log_out){
        handle_anomaly(*this, log_out);
    }

    class AnomalyLog {
        private: 
            static std::vector<std::unique_ptr<AbstractAnomaly>> anomalies;
            AnomalyLog() = default;
            ~AnomalyLog() = default;
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

            static void handle_anomalies(std::ostream &os = std::cerr){
                for(auto &anomaly : anomalies){
                    anomaly->handle_self(os);
                }
            }
    };


    struct expectation_anomaly_tag{ using anomaly_tag = expectation_anomaly_tag; };

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
        std::source_location loc = std::source_location::current(),
        std::stacktrace trace = std::stacktrace::current()
    ){
        if(!expect_true){
            AnomalyLog::log_anomaly(Anomaly{"Expectation failed", expectation_anomaly_tag{}, loc, trace});
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
